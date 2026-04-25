import argparse
import json
import re
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf

WEIGHT_PATH = Path("austen_subword_model.weights.h5")
VOCAB_PATH = Path("bpe_vocabulary.txt")

D_MODEL = 512
NUM_HEADS = 8
DFF = 2048
NUM_LAYERS = 6
DROPOUT_RATE = 0.3
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 512

TOKEN_PATTERN = re.compile(r"\n|--|'s|'t|'re|'m|'d|[A-Za-z]+|[0-9]+|[^\w\s]")


def load_vocab(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        raw = [line.rstrip("\n") for line in f]

    vocab = []
    for token in raw:
        if token == "<NEWLINE>":
            vocab.append("\n")
        else:
            vocab.append(token)
    return vocab


class WordPieceTokenizer:
    def __init__(self, vocabulary: List[str]):
        self.vocabulary = vocabulary
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(vocabulary)}
        self.id_to_token: Dict[int, str] = {i: tok for i, tok in enumerate(vocabulary)}

        self.unk_id = self.token_to_id.get("[UNK]", 0)
        self.pad_id = self.token_to_id.get("[PAD]", 1)
        self.cap_id = self.token_to_id.get("^", 2)

    def normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        out = []
        prev = ""
        for ch in text:
            if ch.isupper() and (not prev.isalpha()):
                out.append("^ ")
                out.append(ch.lower())
            else:
                out.append(ch.lower())
            prev = ch
        return "".join(out)

    def encode(self, text: str) -> List[int]:
        pieces = TOKEN_PATTERN.findall(text)
        ids: List[int] = []

        for piece in pieces:
            if piece in self.token_to_id:
                ids.append(self.token_to_id[piece])
                continue

            if piece.isalpha():
                ids.extend(self._encode_word(piece))
            else:
                ids.append(self.unk_id)

        return ids or [self.unk_id]

    def _encode_word(self, word: str) -> List[int]:
        if word in self.token_to_id:
            return [self.token_to_id[word]]

        tokens: List[int] = []
        cursor = 0

        while cursor < len(word):
            found = False
            for end in range(len(word), cursor, -1):
                segment = word[cursor:end]
                candidate = segment if cursor == 0 else f"##{segment}"
                token_id = self.token_to_id.get(candidate)
                if token_id is not None:
                    tokens.append(token_id)
                    cursor = end
                    found = True
                    break

            if not found:
                return [self.unk_id]

        return tokens


@tf.keras.utils.register_keras_serializable()
class TiedDense(tf.keras.layers.Layer):
    def __init__(self, embedding_layer: tf.keras.layers.Embedding):
        super().__init__()
        self.embedding_layer = embedding_layer

    def call(self, x: tf.Tensor) -> tf.Tensor:
        embedding_matrix = self.embedding_layer.embeddings
        logits = tf.einsum("bld,vd->blv", x, embedding_matrix)
        return logits

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "embedding_layer_name": getattr(self.embedding_layer, "name", None),
            }
        )
        return config


@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation=lambda t: tf.keras.activations.gelu(t, approximate=True)),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(d_model),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool = False, causal_mask: tf.Tensor | None = None) -> tf.Tensor:
        norm_x = self.layernorm1(x)
        attn_output = self.attention(
            query=norm_x,
            value=norm_x,
            key=norm_x,
            attention_mask=causal_mask,
            training=training,
        )
        x = x + self.dropout1(attn_output, training=training)

        norm_x = self.layernorm2(x)
        ffn_output = self.ffn(norm_x, training=training)
        x = x + self.dropout2(ffn_output, training=training)
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable()
class AustenTextModel(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        dff: int = 512,
        num_layers: int = 4,
        dropout_rate: float = 0.3,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_embedding = tf.keras.layers.Embedding(max_seq_len, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)
        ]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.tied_dense = TiedDense(self.token_embedding)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(seq_len)

        x = self.token_embedding(inputs)
        x *= tf.cast(tf.math.sqrt(tf.cast(self.d_model, tf.float32)), x.dtype)
        x += self.pos_embedding(positions)
        x = self.dropout(x, training=training)

        row_ids = tf.range(seq_len)[:, tf.newaxis]
        col_ids = tf.range(seq_len)[tf.newaxis, :]
        causal_mask = tf.cast(row_ids >= col_ids, tf.float32)
        causal_mask = causal_mask[tf.newaxis, :, :]

        for block in self.transformer_blocks:
            x = block(x, training=training, causal_mask=causal_mask)

        x = self.layernorm(x)
        return self.tied_dense(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "num_layers": self.num_layers,
                "dropout_rate": self.dropout_rate,
                "max_seq_len": self.max_seq_len,
            }
        )
        return config


class TextGenerator:
    def __init__(self, model: AustenTextModel, tokenizer: WordPieceTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.id_to_token = tokenizer.id_to_token

    def generate(
        self,
        prompt: str,
        max_tokens: int = 120,
        temperature: float = 0.7,
        top_k: int = 50,
        repetition_penalty: float = 2.5,
    ) -> Tuple[str, str]:
        normalized = self.tokenizer.normalize_text(prompt)
        input_ids = self.tokenizer.encode(normalized)
        generated_ids: List[int] = []

        for _ in range(max_tokens):
            context_ids = input_ids[-SEQUENCE_LENGTH:]
            context = tf.constant([context_ids], dtype=tf.int32)

            logits = self.model(context, training=False)
            logits = tf.cast(logits[:, -1, :], tf.float32)
            logits /= max(temperature, 1e-4)

            banned = [self.tokenizer.unk_id, self.tokenizer.pad_id]
            for token_id in banned:
                if 0 <= token_id < VOCAB_SIZE:
                    logits = tf.tensor_scatter_nd_update(logits, [[0, token_id]], [-1e9])

            if top_k > 0:
                top_values, _ = tf.math.top_k(logits, k=min(top_k, VOCAB_SIZE))
                kth = top_values[:, -1:]
                logits = tf.where(logits < kth, tf.constant(-1e9, dtype=logits.dtype), logits)

            recent = input_ids[-128:]
            frequencies = Counter(recent)
            for token_id, count in frequencies.items():
                if token_id not in banned and token_id != self.tokenizer.cap_id:
                    penalty = repetition_penalty * float(count)
                    logits = tf.tensor_scatter_nd_sub(
                        logits,
                        indices=[[0, token_id]],
                        updates=tf.constant([penalty], dtype=logits.dtype),
                    )

            predicted = int(tf.random.categorical(logits, num_samples=1)[0, 0].numpy())
            input_ids.append(predicted)
            generated_ids.append(predicted)

        continuation = self._decode_ids(generated_ids)
        full_text = self._merge_prompt_and_completion(prompt, continuation)
        return continuation, full_text

    def _decode_ids(self, token_ids: List[int]) -> str:
        raw = ""
        punctuation = {".", ",", ";", "!", "?", ":", "'", '"', ")", "(", "-", "--"}

        for tok_id in token_ids:
            token = self.id_to_token.get(int(tok_id), "[UNK]")
            if token in {"[UNK]", "[PAD]", ""}:
                continue

            if token.startswith("##"):
                raw += token[2:]
            elif token == "\n":
                raw = raw.rstrip() + "\n"
            elif token == "^":
                if raw and not raw.endswith((" ", "\n", "(")):
                    raw += " "
                raw += "^"
            elif token in punctuation:
                raw = raw.rstrip() + token
            else:
                if raw and not raw.endswith((" ", "\n", "(")):
                    raw += " "
                raw += token

        raw = self._apply_cap_markers(raw)
        raw = re.sub(r"\s+([?.!,;:'\")])", r"\1", raw)
        raw = re.sub(r"([(])\s+", r"\1", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n +", "\n", raw)
        raw = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", raw)
        return raw.strip()

    @staticmethod
    def _apply_cap_markers(text: str) -> str:
        out = []
        i = 0
        while i < len(text):
            if text[i] == "^":
                i += 1
                while i < len(text) and text[i] == " ":
                    i += 1
                if i < len(text):
                    out.append(text[i].upper())
                    i += 1
                continue

            out.append(text[i])
            i += 1
        return "".join(out)

    @staticmethod
    def _merge_prompt_and_completion(prompt: str, continuation: str) -> str:
        if not continuation:
            return prompt
        if not prompt:
            return continuation

        if continuation[0] in ".,;:!?')\"]":
            return f"{prompt.rstrip()}{continuation}"
        if prompt.endswith((" ", "\n", "(")):
            return f"{prompt}{continuation}"
        return f"{prompt} {continuation}"


MODEL_LOCK = threading.Lock()
GENERATOR: TextGenerator | None = None


def build_generator() -> TextGenerator:
    vocabulary = load_vocab(VOCAB_PATH)
    tokenizer = WordPieceTokenizer(vocabulary)

    model = AustenTextModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        max_seq_len=SEQUENCE_LENGTH,
    )

    model(tf.zeros((1, 1), dtype=tf.int32), training=False)
    model.load_weights(str(WEIGHT_PATH))

    return TextGenerator(model=model, tokenizer=tokenizer)


def get_generator() -> TextGenerator:
    global GENERATOR
    with MODEL_LOCK:
        if GENERATOR is None:
            GENERATOR = build_generator()
    return GENERATOR


def run_cli(args: argparse.Namespace) -> None:
    generator = get_generator()
    continuation, full_text = generator.generate(
        prompt=args.prompt,
        max_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.penalty,
    )

    print("\n=== Continuation ===")
    print(continuation)
    print("\n=== Full Text ===")
    print(full_text)


def make_html() -> str:
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Austen Transformer Demo</title>
  <style>
    :root {
      --bg: #0d1b2a;
      --panel: rgba(255, 255, 255, 0.08);
      --panel-strong: rgba(255, 255, 255, 0.14);
      --text: #f7fafc;
      --subtext: #cbd5e1;
      --accent: #f59e0b;
      --accent-2: #0ea5e9;
      --ok: #10b981;
      --danger: #ef4444;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", "SF Pro Text", "Trebuchet MS", sans-serif;
      color: var(--text);
      background:
        radial-gradient(1200px 700px at 10% -10%, #1b263b 0%, transparent 55%),
        radial-gradient(900px 600px at 100% 10%, #0f172a 0%, transparent 50%),
        linear-gradient(145deg, #0d1b2a 0%, #1b263b 45%, #0b132b 100%);
      display: grid;
      place-items: center;
      padding: 24px;
    }

    .app {
      width: min(980px, 100%);
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 18px;
      backdrop-filter: blur(10px);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
      overflow: hidden;
    }

    .header {
      padding: 24px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.15);
      background: linear-gradient(90deg, rgba(245, 158, 11, 0.18), rgba(14, 165, 233, 0.18));
    }

    .title {
      margin: 0;
      font-size: clamp(1.5rem, 3vw, 2rem);
      letter-spacing: 0.5px;
    }

    .subtitle {
      margin: 8px 0 0;
      color: var(--subtext);
      line-height: 1.5;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
      padding: 20px;
    }

    label {
      display: block;
      font-size: 0.9rem;
      color: var(--subtext);
      margin-bottom: 6px;
    }

    textarea,
    input {
      width: 100%;
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid rgba(255, 255, 255, 0.2);
      background: rgba(2, 6, 23, 0.5);
      color: var(--text);
      outline: none;
      transition: border-color 0.18s ease, box-shadow 0.18s ease;
    }

    textarea:focus,
    input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.2);
    }

    textarea {
      min-height: 150px;
      resize: vertical;
      line-height: 1.5;
    }

    .controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }

    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }

    button {
      padding: 12px 18px;
      border: none;
      border-radius: 10px;
      font-weight: 600;
      letter-spacing: 0.2px;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .primary {
      color: #0b132b;
      background: linear-gradient(90deg, var(--accent), #fbbf24);
      box-shadow: 0 8px 26px rgba(245, 158, 11, 0.35);
    }

    .primary:hover {
      transform: translateY(-1px);
    }

    .ghost {
      color: var(--text);
      background: var(--panel-strong);
    }

    .status {
      font-size: 0.95rem;
      color: var(--subtext);
      min-height: 1.2em;
    }

    .status.ok { color: var(--ok); }
    .status.err { color: var(--danger); }

    .outputs {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }

    .output-box {
      padding: 14px;
      border-radius: 10px;
      background: rgba(2, 6, 23, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.15);
      min-height: 90px;
      white-space: pre-wrap;
      line-height: 1.5;
    }

    .footer {
      padding: 14px 20px 20px;
      color: var(--subtext);
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <main class=\"app\">
    <section class=\"header\">
      <h1 class=\"title\">Austen Transformer Text Generator</h1>
      <p class=\"subtitle\">Type a prompt and the model writes a continuation. Built with TensorFlow and a local HTTP API.</p>
    </section>

    <section class=\"grid\">
      <div>
        <label for=\"prompt\">Prompt</label>
        <textarea id=\"prompt\" placeholder=\"Example: The world seemed peaceful until the strange tree appeared in London...\"></textarea>
      </div>

      <div class=\"controls\">
        <div>
          <label for=\"tokens\">Max Tokens</label>
          <input id=\"tokens\" type=\"number\" value=\"120\" min=\"1\" max=\"500\" />
        </div>
        <div>
          <label for=\"temperature\">Temperature</label>
          <input id=\"temperature\" type=\"number\" step=\"0.05\" value=\"0.7\" min=\"0.1\" max=\"2.0\" />
        </div>
        <div>
          <label for=\"topk\">Top-K</label>
          <input id=\"topk\" type=\"number\" value=\"50\" min=\"1\" max=\"200\" />
        </div>
        <div>
          <label for=\"penalty\">Repetition Penalty</label>
          <input id=\"penalty\" type=\"number\" step=\"0.1\" value=\"2.5\" min=\"0\" max=\"10\" />
        </div>
      </div>

      <div class=\"actions\">
        <button class=\"primary\" id=\"generate\">Generate</button>
        <button class=\"ghost\" id=\"fill\">Use Sample Prompt</button>
        <span id=\"status\" class=\"status\"></span>
      </div>

      <div class=\"outputs\">
        <div>
          <label>Generated Continuation</label>
          <div id=\"continuation\" class=\"output-box\"></div>
        </div>
        <div>
          <label>Full Output</label>
          <div id=\"full\" class=\"output-box\"></div>
        </div>
      </div>
    </section>

    <section class=\"footer\">
      Local-only demo suitable for a portfolio walkthrough. Start server with: <code>python main.py --serve</code>
    </section>
  </main>

  <script>
    const statusEl = document.getElementById('status');
    const continuationEl = document.getElementById('continuation');
    const fullEl = document.getElementById('full');

    function setStatus(text, kind = '') {
      statusEl.textContent = text;
      statusEl.className = 'status ' + kind;
    }

    async function generate() {
      const prompt = document.getElementById('prompt').value.trim();
      const max_tokens = Number(document.getElementById('tokens').value || 120);
      const temperature = Number(document.getElementById('temperature').value || 0.7);
      const top_k = Number(document.getElementById('topk').value || 50);
      const repetition_penalty = Number(document.getElementById('penalty').value || 2.5);

      if (!prompt) {
        setStatus('Please enter a prompt.', 'err');
        return;
      }

      setStatus('Generating...');
      continuationEl.textContent = '';
      fullEl.textContent = '';

      try {
        const response = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, max_tokens, temperature, top_k, repetition_penalty })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || 'Generation failed.');
        }

        continuationEl.textContent = data.continuation;
        fullEl.textContent = data.full_text;
        setStatus('Done.', 'ok');
      } catch (error) {
        setStatus(error.message, 'err');
      }
    }

    document.getElementById('generate').addEventListener('click', generate);
    document.getElementById('fill').addEventListener('click', () => {
      document.getElementById('prompt').value = 'The world seemed like such a peaceful place until the magic tree was discovered in London.';
    });
  </script>
</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, html: str) -> None:
        raw = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_html(make_html())
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self._send_json({"error": "Not found"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8"))
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("Prompt is required.")

            max_tokens = int(payload.get("max_tokens", 120))
            temperature = float(payload.get("temperature", 0.7))
            top_k = int(payload.get("top_k", 50))
            repetition_penalty = float(payload.get("repetition_penalty", 2.5))

            max_tokens = max(1, min(500, max_tokens))
            top_k = max(1, min(200, top_k))
            temperature = max(0.1, min(2.0, temperature))
            repetition_penalty = max(0.0, min(10.0, repetition_penalty))

            generator = get_generator()
            with MODEL_LOCK:
                continuation, full_text = generator.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )

            self._send_json(
                {
                    "prompt": prompt,
                    "continuation": continuation,
                    "full_text": full_text,
                }
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args) -> None:
        return


def run_server(host: str, port: int) -> None:
    _ = get_generator()
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"Server running at http://{host}:{port}")
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Austen transformer text generation demo")
    parser.add_argument("--prompt", type=str, default="The world seemed peaceful until", help="Prompt text")
    parser.add_argument("--tokens", type=int, default=120, help="Number of generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--penalty", type=float, default=2.5, help="Repetition penalty")
    parser.add_argument("--serve", action="store_true", help="Run local web app")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.serve:
        run_server(args.host, args.port)
    else:
        run_cli(args)


if __name__ == "__main__":
    main()

