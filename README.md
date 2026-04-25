# transformer_model
A text-generation portfolio project with two runnable modes:

1. Python/TensorFlow local generator (`main.py`)
2. Static JavaScript website (`docs/`) that runs your **local Austen model** in-browser

## GitHub Pages website (uses your local model)
The website loads model files from `docs/model/` (not from an external model ID).

### Files
- `docs/index.html` - app layout
- `docs/styles.css` - visual styling
- `docs/app.js` - browser inference logic (TensorFlow.js)
- `scripts/convert_local_model_to_tfjs.py` - converts local `.h5` weights to browser model files
- `.github/workflows/deploy-pages.yml` - automatic Pages deployment workflow

### Build browser model files
Run this from repo root:

```bash
python scripts/convert_local_model_to_tfjs.py
```

This generates:
- `docs/model/model.json`
- `docs/model/group1-shard*.bin`
- `docs/model/vocab.txt`

### Publish steps
1. Run conversion script above.
2. Commit `docs/model/*` so Pages can serve your model.
3. Push to GitHub (`main`).
4. In GitHub: `Settings` -> `Pages` and set source to **GitHub Actions**.
5. Workflow deploys to:
   - `https://<your-username>.github.io/<repo-name>/`

### UX behavior
- Connected generation (`prompt + completion`)
- Generated output is written back into the prompt box for iterative continuation
- Controls:
  - `Max New Tokens`
  - `Temperature`
  - `Top-P`
  - `Repetition Penalty`

### Local preview (static site)
```bash
python -m http.server 8080
```
Then visit: `http://127.0.0.1:8080/docs/`

## Python/TensorFlow local mode
This keeps your original direct-weights workflow.

### Run CLI
```bash
python main.py --prompt "The evening was calm until" --tokens 120 --temperature 0.7 --top-k 50 --penalty 2.5
```

### Run local web server (Python backend)
```bash
python main.py --serve --host 127.0.0.1 --port 8000
```
Then visit: `http://127.0.0.1:8000`
