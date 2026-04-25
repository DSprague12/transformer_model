# transformer_model
A text-generation portfolio project with two runnable modes:

1. Python/TensorFlow local generator (`main.py`)
2. Static JavaScript website for GitHub Pages (`docs/index.html`, `docs/app.js`, `docs/styles.css`)

## GitHub Pages website (recommended for portfolio)
This mode runs fully in the browser using Transformers.js and does not need a backend server.

### Files
- `docs/index.html` - app layout
- `docs/styles.css` - visual styling
- `docs/app.js` - browser inference logic
- `.github/workflows/deploy-pages.yml` - automatic deployment workflow

### UX behavior
- Generation now returns connected output (`prompt + completion`)
- The prompt box is auto-updated with generated text so users can keep extending naturally
- Controls include:
  - `Max New Tokens`
  - `Temperature`
  - `Top-P`
  - `Repetition Penalty`

### Publish steps
1. Push this repo to GitHub (default branch `main`).
2. In GitHub: `Settings` -> `Pages`.
3. Ensure source is **GitHub Actions**.
4. The included workflow (`Deploy static site to GitHub Pages`) will deploy on each push to `main`.
5. Your site URL will be:
   - `https://<your-username>.github.io/<repo-name>/`

### Local preview (optional)
Open `docs/index.html` in a browser, or run a local static server from repo root:

```bash
python -m http.server 8080
```

Then visit: `http://127.0.0.1:8080`

## Python/TensorFlow local mode
This keeps your original model + weights workflow.

### Run CLI
```bash
python main.py --prompt "The evening was calm until" --tokens 120 --temperature 0.7 --top-k 50 --penalty 2.5
```

### Run local web server (Python backend)
```bash
python main.py --serve --host 127.0.0.1 --port 8000
```

Then visit: `http://127.0.0.1:8000`

