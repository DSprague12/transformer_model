# transformer_model
A text-generation portfolio project with two runnable modes:

1. Python/TensorFlow local generator (`main.py`)
2. Static JavaScript website (`docs/`) that runs the **local model** in-browser

## GitHub Pages website
The website loads model files from `docs/model/` (not from an external model ID).

### Files
- `docs/index.html` - app layout
- `docs/styles.css` - visual styling
- `docs/app.js` - browser inference logic (TensorFlow.js)
- `scripts/convert_local_model_to_tfjs.py` - converts local `.h5` weights to browser model files
- `.github/workflows/deploy-pages.yml` - automatic Pages deployment workflow

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

### Run CLI
```bash
python main.py --prompt "The evening was calm until" --tokens 120 --temperature 0.7 --top-k 50 --penalty 2.5
```

### Run local web server (Python backend)
```bash
python main.py --serve --host 127.0.0.1 --port 8000
```
Then visit: `http://127.0.0.1:8000`
