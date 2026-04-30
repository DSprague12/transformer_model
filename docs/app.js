const promptEl = document.getElementById("prompt");
const outputEl = document.getElementById("output");
const statusEl = document.getElementById("status");
const generateBtn = document.getElementById("generateBtn");
const sampleBtn = document.getElementById("sampleBtn");
const maxTokensEl = document.getElementById("maxTokens");
const temperatureEl = document.getElementById("temperature");
const topPEl = document.getElementById("topP");
const repetitionPenaltyEl = document.getElementById("repetitionPenalty");

const MODEL_URL = "./model/model.json";
const VOCAB_URL = "./model/vocab.txt";
const SEQUENCE_LENGTH = 512;

const TOKEN_PATTERN = /\n|--|'s|'t|'re|'m|'d|[A-Za-z]+|[0-9]+|[^\w\s]/g;

let model = null;
let tokenizer = null;
const tf = window.tf;

function setStatus(text, tone = "") {
  statusEl.textContent = text;
  statusEl.className = tone ? `status ${tone}` : "status";
}

function clampNumber(value, min, max, fallback) {
  const n = Number(value);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

class WordPieceTokenizer {
  constructor(vocabulary) {
    this.vocabulary = vocabulary;
    this.tokenToId = new Map(vocabulary.map((t, i) => [t, i]));
    this.idToToken = vocabulary;
    this.unkId = this.tokenToId.get("[UNK]") ?? 0;
    this.padId = this.tokenToId.get("[PAD]") ?? 1;
    this.capId = this.tokenToId.get("^") ?? 2;
  }

  normalizeText(text) {
    const normalized = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    let out = "";
    let prev = "";
    for (const ch of normalized) {
      if (/[A-Z]/.test(ch) && !/[A-Za-z]/.test(prev)) {
        out += "^ " + ch.toLowerCase();
      } else {
        out += ch.toLowerCase();
      }
      prev = ch;
    }
    return out;
  }

  encode(text) {
    const pieces = text.match(TOKEN_PATTERN) ?? [];
    if (pieces.length === 0) return [this.unkId];

    const ids = [];
    for (const piece of pieces) {
      if (this.tokenToId.has(piece)) {
        ids.push(this.tokenToId.get(piece));
        continue;
      }

      if (/^[a-z]+$/.test(piece)) {
        ids.push(...this.encodeWord(piece));
      } else {
        ids.push(this.unkId);
      }
    }

    return ids.length > 0 ? ids : [this.unkId];
  }

  encodeWord(word) {
    if (this.tokenToId.has(word)) {
      return [this.tokenToId.get(word)];
    }

    const ids = [];
    let cursor = 0;

    while (cursor < word.length) {
      let found = false;
      for (let end = word.length; end > cursor; end -= 1) {
        const segment = word.slice(cursor, end);
        const candidate = cursor === 0 ? segment : `##${segment}`;
        if (this.tokenToId.has(candidate)) {
          ids.push(this.tokenToId.get(candidate));
          cursor = end;
          found = true;
          break;
        }
      }

      if (!found) {
        return [this.unkId];
      }
    }

    return ids;
  }

  decode(ids) {
    let raw = "";
    const punctuation = new Set([".", ",", ";", "!", "?", ":", "'", '"', ")", "(", "-", "--"]);

    for (const id of ids) {
      const token = this.idToToken[id] ?? "[UNK]";
      if (token === "[UNK]" || token === "[PAD]" || token === "") {
        continue;
      }

      if (token.startsWith("##")) {
        raw += token.slice(2);
      } else if (token === "\n") {
        raw = raw.trimEnd() + "\n";
      } else if (token === "^") {
        if (raw && !raw.endsWith(" ") && !raw.endsWith("\n") && !raw.endsWith("(")) {
          raw += " ";
        }
        raw += "^";
      } else if (punctuation.has(token)) {
        raw = raw.trimEnd() + token;
      } else {
        if (raw && !raw.endsWith(" ") && !raw.endsWith("\n") && !raw.endsWith("(")) {
          raw += " ";
        }
        raw += token;
      }
    }

    raw = applyCapMarkers(raw);
    raw = raw.replace(/\s+([?.!,;:'")])/g, "$1");
    raw = raw.replace(/([(])\s+/g, "$1");
    raw = raw.replace(/[ \t]+/g, " ");
    raw = raw.replace(/\n +/g, "\n");
    raw = raw.replace(/([.!?])([A-Za-z])/g, "$1 $2");

    return raw.trim();
  }
}

function applyCapMarkers(text) {
  let out = "";
  let i = 0;

  while (i < text.length) {
    if (text[i] === "^") {
      i += 1;
      while (i < text.length && text[i] === " ") i += 1;
      if (i < text.length) {
        out += text[i].toUpperCase();
        i += 1;
      }
    } else {
      out += text[i];
      i += 1;
    }
  }

  return out;
}

function mergePromptAndCompletion(prompt, completion) {
  if (!completion) return prompt;
  if (!prompt) return completion;

  if (/[.,;:!?')"\]]/.test(completion[0])) {
    return `${prompt.trimEnd()}${completion}`;
  }

  if (prompt.endsWith(" ") || prompt.endsWith("\n") || prompt.endsWith("(")) {
    return `${prompt}${completion}`;
  }

  return `${prompt} ${completion}`;
}

function sampleFromLogits(logits, temperature, topP, topK, repetitionPenalty, recentIds, bannedIds, capId) {
  const adjusted = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i += 1) {
    adjusted[i] = logits[i] / Math.max(temperature, 1e-4);
  }

  for (const id of bannedIds) {
    if (id >= 0 && id < adjusted.length) adjusted[id] = -1e9;
  }

  const freq = new Map();
  for (const id of recentIds) {
    freq.set(id, (freq.get(id) ?? 0) + 1);
  }

  for (const [id, count] of freq.entries()) {
    if (!bannedIds.has(id) && id !== capId && id >= 0 && id < adjusted.length) {
      adjusted[id] -= repetitionPenalty * count;
    }
  }

  const scored = [];
  for (let i = 0; i < adjusted.length; i += 1) {
    scored.push([i, adjusted[i]]);
  }
  scored.sort((a, b) => b[1] - a[1]);

  const k = Math.max(1, Math.min(topK, scored.length));
  const topKSlice = scored.slice(0, k);

  const maxLogit = topKSlice[0][1];
  const exps = topKSlice.map(([id, score]) => [id, Math.exp(score - maxLogit)]);
  const total = exps.reduce((s, [, v]) => s + v, 0);
  const probs = exps.map(([id, v]) => [id, v / (total || 1)]);

  let cum = 0;
  const nucleus = [];
  for (const [id, p] of probs) {
    nucleus.push([id, p]);
    cum += p;
    if (cum >= topP) break;
  }

  const nucleusTotal = nucleus.reduce((s, [, p]) => s + p, 0) || 1;
  let r = Math.random();
  for (const [id, p] of nucleus) {
    r -= p / nucleusTotal;
    if (r <= 0) return id;
  }

  return nucleus[nucleus.length - 1][0];
}

async function ensureLocalModel() {
  if (!tf) {
    throw new Error("TensorFlow.js failed to load. Check internet/CDN access and refresh.");
  }
  if (model && tokenizer) return;

  generateBtn.disabled = true;
  setStatus("Loading local model files (docs/model)... first run may take a while.");

  const [loadedModel, vocabText] = await Promise.all([
    tf.loadGraphModel(MODEL_URL),
    fetch(VOCAB_URL).then((r) => {
      if (!r.ok) throw new Error("Could not load docs/model/vocab.txt");
      return r.text();
    }),
  ]);

  const vocabulary = vocabText
    .split(/\r?\n/)
    .filter((line) => line.length > 0)
    .map((line) => (line === "<NEWLINE>" ? "\n" : line));

  model = loadedModel;
  tokenizer = new WordPieceTokenizer(vocabulary);

  setStatus("Local Austen model loaded. Ready.", "ok");
  generateBtn.disabled = false;
}

async function generateWithLocalModel() {
  const prompt = promptEl.value.trim();
  if (!prompt) {
    setStatus("Add a prompt first.", "err");
    return;
  }

  try {
    await ensureLocalModel();

    generateBtn.disabled = true;
    setStatus("Generating with local model...");

    const maxNewTokens = clampNumber(maxTokensEl.value, 10, 300, 90);
    const temperature = clampNumber(temperatureEl.value, 0.1, 2.0, 0.9);
    const topP = clampNumber(topPEl.value, 0.1, 1.0, 0.92);
    const repetitionPenalty = clampNumber(repetitionPenaltyEl.value, 1.0, 2.0, 1.12);
    const topK = 50;

    const startedAt = performance.now();

    const normalized = tokenizer.normalizeText(prompt);
    const promptIds = tokenizer.encode(normalized);
    const inputIds = [...promptIds];

    for (let step = 0; step < maxNewTokens; step += 1) {
      const contextIds = inputIds.slice(-SEQUENCE_LENGTH);
      const contextTensor = tf.tensor2d([contextIds], [1, contextIds.length], "int32");

      let logits = null;
      try {
        logits = model.execute({ "inputs:0": contextTensor }, "Identity:0");
      } catch {
        // Fallback names used by some tfjs graph exports.
        logits = model.execute({ inputs: contextTensor }, "logits");
      }
      const lastLogits = logits.slice([0, contextIds.length - 1, 0], [1, 1, -1]).squeeze();
      const logitsArray = await lastLogits.data();

      const predictedId = sampleFromLogits(
        logitsArray,
        temperature,
        topP,
        topK,
        repetitionPenalty,
        inputIds.slice(-128),
        new Set([tokenizer.unkId, tokenizer.padId]),
        tokenizer.capId,
      );

      inputIds.push(predictedId);

      tf.dispose([contextTensor, logits, lastLogits]);
      await tf.nextFrame();
    }

    const generatedIds = inputIds.slice(promptIds.length);
    const continuation = tokenizer.decode(generatedIds);
    const merged = mergePromptAndCompletion(prompt, continuation);

    outputEl.textContent = merged;
    promptEl.value = merged;

    const elapsedMs = Math.max(1, performance.now() - startedAt);
    const tokPerSec = ((maxNewTokens * 1000) / elapsedMs).toFixed(1);
    setStatus(`Done. ~${tokPerSec} tok/s`, "ok");
  } catch (error) {
    outputEl.textContent = "";
    setStatus(`Generation failed: ${error.message}`, "err");
  } finally {
    generateBtn.disabled = false;
  }
}

sampleBtn.addEventListener("click", () => {
  promptEl.value = "The city had been quiet for years, until one letter arrived at dawn.";
  promptEl.focus();
});

generateBtn.addEventListener("click", generateWithLocalModel);

window.addEventListener("DOMContentLoaded", async () => {
  try {
    await ensureLocalModel();
  } catch (error) {
    setStatus(
      `Local model not found. Run: python scripts/convert_local_model_to_tfjs.py (${error.message})`,
      "err",
    );
  }
});




