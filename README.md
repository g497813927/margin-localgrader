# Margin

A local-only rubric grading workspace with customizable decision models and optional AI explanations.

Start with a **blank rubric and empty optional instructor guidance**. Add your criteria, paste text, extract document text, or review local image/diagram descriptions, compare local decision models, and enter your own final scores.

**[Open the single setup guide](SETUP.html)** for Windows, Linux, and macOS installation, model presets, local vision setup, and server controls. With the project virtual environment activated:

```sh
python run.py
```

Open **http://127.0.0.1:8765/**. On macOS you can also double-click **Start Margin.command**. If another copy is running, the setup guide includes alternate-port commands for each platform.

- Configure up to eight local decision models. Search presets, add/remove entries, and start/stop bundled servers.
- Optional local Ollama explanations leave model and instructor scores unchanged.
- Upload a submission once: PDF/Word text extraction also detects visual content and opens a preview. Automatic detection is enabled by default and can be disabled in Model workspace for text-only document imports.
- Optional local vision preserves native PDF text, reads embedded PDF/Word images separately, and uses overlapping crops for detailed diagrams. PNG/JPEG/WebP and vector PDF pages are also supported. Analysis starts only when requested; review and correct its transcription/descriptions before adding them to the submission. Recognition never assigns scores. Convert Word vector drawings and charts to PDF first.
- Model workspace saves contain model, explanation, and vision settings only. Submissions, rubrics, grades, and notes are not saved automatically. Exports are explicit local files.
- No cloud inference, Canvas API, analytics, or grading logs. Dependency and model downloads happen during setup, separately from grading.

## Development

Use Node.js **26.12.1** (pinned in `.node-version`) for frontend tests. It includes the [October 2026 security fixes](https://nodejs.org/en/blog/vulnerability/october-2026-security-releases). Node and npm are not required to run the Python application; there are no npm packages to install.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
node tests/frontend.cjs
```

On Windows, create the environment with `py -3.12 -m venv .venv` and replace `.venv/bin/python` with `.venv\Scripts\python.exe`. Automated tests pass on macOS and in a fresh Linux container. Windows launcher branches have simulated tests and a native Windows CI job; native Windows execution still needs verification.

Tests use synthetic fixtures and mocked inference; no model download is required. Test rubrics are fixtures only and never load as application defaults.

Dependabot checks pinned Python packages and GitHub Actions daily and opens reviewable update PRs. CI tests the portable app and optional decision runtime on those PRs. A dependency audit also checks application, optional model, and maintenance packages on dependency changes and every Monday. These maintenance jobs query public package advisories; they never read submissions or connect grading inference to a remote service. Review and merge update PRs after their checks pass.

## Publishing this source

Keep model weights, environment files, saved configurations, submissions, and result exports outside version control. The included `.gitignore` covers the standard local directories and export filenames. Review the files selected for a commit before publishing; no runtime data is part of this source distribution.

Third-party code retains its license and notices under `vendor/d3/`. Model weights are separate downloads governed by their upstream licenses. This project does not assign a new license to those third-party materials.
