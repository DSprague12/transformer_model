import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

env.allowLocalModels = false;
env.useBrowserCache = true;

env.backends.onnx.wasm.numThreads = 1;

const promptEl = document.getElementById("prompt");
const outputEl = document.getElementById("output");
const statusEl = document.getElementById("status");
const generateBtn = document.getElementById("generateBtn");
const sampleBtn = document.getElementById("sampleBtn");
const maxTokensEl = document.getElementById("maxTokens");
const temperatureEl = document.getElementById("temperature");
const topPEl = document.getElementById("topP");
const repetitionPenaltyEl = document.getElementById("repetitionPenalty");

const MODEL_ID = "Xenova/distilgpt2";
let generator = null;

function setStatus(text, tone = "") {
  statusEl.textContent = text;
  statusEl.className = tone ? `status ${tone}` : "status";
}

async function ensureGenerator() {
  if (generator) {
    return generator;
  }

  setStatus("Loading model... first run can take up to a minute.");
  generateBtn.disabled = true;

  const supportsWebGPU = typeof navigator !== "undefined" && !!navigator.gpu;
  if (supportsWebGPU) {
    try {
      generator = await pipeline("text-generation", MODEL_ID, { device: "webgpu" });
      setStatus("Model loaded on WebGPU. Ready.", "ok");
      generateBtn.disabled = false;
      return generator;
    } catch {
      // Fallback to WASM if WebGPU init fails.
    }
  }

  generator = await pipeline("text-generation", MODEL_ID);

  setStatus("Model loaded. Ready.", "ok");
  generateBtn.disabled = false;
  return generator;
}

function clampNumber(value, min, max, fallback) {
  const n = Number(value);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

async function runGeneration() {
  const prompt = promptEl.value.trim();
  if (!prompt) {
    setStatus("Add a prompt first.", "err");
    return;
  }

  try {
    generateBtn.disabled = true;
    setStatus("Generating continuation...");

    const pipe = await ensureGenerator();

    const maxNewTokens = clampNumber(maxTokensEl.value, 10, 300, 90);
    const temperature = clampNumber(temperatureEl.value, 0.1, 2.0, 0.9);
    const topP = clampNumber(topPEl.value, 0.1, 1.0, 0.92);
    const repetitionPenalty = clampNumber(repetitionPenaltyEl.value, 1.0, 2.0, 1.12);
    const startedAt = performance.now();

    const out = await pipe(prompt, {
      max_new_tokens: maxNewTokens,
      temperature,
      top_p: topP,
      top_k: 50,
      do_sample: true,
      repetition_penalty: repetitionPenalty,
      return_full_text: true,
    });

    const mergedText = out?.[0]?.generated_text?.trim() ?? "(No output)";
    outputEl.textContent = mergedText;
    if (mergedText !== "(No output)") {
      promptEl.value = mergedText;
    }

    const elapsedMs = Math.max(1, performance.now() - startedAt);
    const approxTokensPerSec = ((maxNewTokens * 1000) / elapsedMs).toFixed(1);
    setStatus(`Done. ~${approxTokensPerSec} tok/s`, "ok");
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

generateBtn.addEventListener("click", runGeneration);

window.addEventListener("DOMContentLoaded", async () => {
  setStatus("Preparing model runtime...");
  try {
    await ensureGenerator();
  } catch (error) {
    setStatus(`Model load failed: ${error.message}`, "err");
  }
});
