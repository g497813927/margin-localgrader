# Margin

A local-only rubric grading workspace with customizable decision models and optional AI explanations.

Start with a **blank rubric and empty optional instructor guidance**. Add your criteria, paste text or extract text from PDF/Word, compare local decision models, and enter your own final scores.

**[Open the single setup guide](SETUP.html)** for installation, the four bundled model presets, custom local connections, and server controls. After setup:

```sh
.venv/bin/python run.py
```

Open **http://127.0.0.1:8765/**. On macOS you can also double-click **Start Margin.command**. If another copy is running, use `MARGIN_PORT=8766 .venv/bin/python run.py`.

- Configure up to eight local decision models. Search presets, add/remove entries, and start/stop bundled servers.
- Optional local Ollama explanations leave model and instructor scores unchanged.
- PDF and Word imports extract text only; images are ignored.
- Model workspace saves contain settings only. Submissions, rubrics, grades, and notes are not saved automatically. Exports are explicit local files.
- No cloud inference, Canvas API, analytics, or grading logs. Dependency and model downloads happen during setup, separately from grading.

## Development

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
node tests/frontend.cjs
```

Tests use synthetic fixtures and mocked inference; no model download is required. Test rubrics are fixtures only and never load as application defaults.

## Publishing this source

Keep model weights, environment files, saved configurations, submissions, and result exports outside version control. The included `.gitignore` covers the standard local directories and export filenames. Review the files selected for a commit before publishing; no runtime data is part of this source distribution.

Third-party code retains its license and notices under `vendor/d3/`. Model weights are separate downloads governed by their upstream licenses. This project does not assign a new license to those third-party materials.
