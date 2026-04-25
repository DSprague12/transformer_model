from pathlib import Path
import shutil
import sys

import numpy as np

# tensorflowjs 3.x expects deprecated numpy aliases.
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "bool"):
    np.bool = np.bool_

import tensorflow as tf
from tensorflowjs.converters import converter as tfjs_converter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main


def run() -> None:
    repo_root = REPO_ROOT
    saved_model_dir = repo_root / "tmp" / "austen_saved_model"
    output_dir = repo_root / "docs" / "model"

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    saved_model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = main.AustenTextModel(
        vocab_size=main.VOCAB_SIZE,
        d_model=main.D_MODEL,
        num_heads=main.NUM_HEADS,
        dff=main.DFF,
        num_layers=main.NUM_LAYERS,
        dropout_rate=main.DROPOUT_RATE,
        max_seq_len=main.SEQUENCE_LENGTH,
    )
    model(tf.zeros((1, 1), dtype=tf.int32), training=False)

    weights_path = repo_root / main.WEIGHT_PATH
    model.load_weights(str(weights_path))

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None], dtype=tf.int32, name="inputs")])
    def serving_fn(inputs: tf.Tensor) -> dict:
        logits = model(inputs, training=False)
        return {"logits": logits}

    tf.saved_model.save(model, str(saved_model_dir), signatures={"serving_default": serving_fn})

    tfjs_converter.convert(
        [
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            "--signature_name=serving_default",
            "--saved_model_tags=serve",
            str(saved_model_dir),
            str(output_dir),
        ]
    )

    vocab_src = repo_root / main.VOCAB_PATH
    vocab_dst = output_dir / "vocab.txt"
    shutil.copyfile(vocab_src, vocab_dst)

    print(f"Export complete: {output_dir}")
    print("Model: docs/model/model.json")
    print("Vocab: docs/model/vocab.txt")


if __name__ == "__main__":
    run()
