# Local Resume Tailor (FastAPI, Local Filesystem)

Local-only resume tailoring web app for Linux servers, designed to be accessed over Tailscale. The app binds to `0.0.0.0:8030`.

## What it does

- Upload resume (`PDF`/`DOCX`), extract text, convert to canonical YAML (`data/resume/base.yaml`) as source of truth.
- Auto-sync canonical resume sections into vault items, then tailor primarily from vault content.
- Maintain an editable local Experience Vault (`data/vault/items/*.yaml`).
- Ingest vault items from rough notes or uploaded legacy docs, then review/edit parsed YAML before save.
- Ingest jobs by URL (`POST /jobs/ingest`) via Playwright + readability; fallback to pasted JD text.
- Tailor a one-page ATS-friendly resume in two modes:
  - `HARD_TRUTH`: conservative, never invents, rejects unsupported claims.
  - `FUCK_IT`: aggressive phrasing/reordering but still no new facts.
- MVP flow uses a single tailoring pass with fixed opinionated defaults; no user tuning knobs in the main UI.
- Legacy per-job feedback and multi-pass optimization code paths have been removed from active backend flow.
- Auto-generate a concise summary line that includes the target role title (when detected) plus evidence terms from selected resume content.
- Render LaTeX via Jinja2 and compile using `latexmk` to `resume.pdf`.
- Enforce one-page output with compile-time control loop: trim when over one page, then add projects back while it still fits.
- Project selection enforces at least 2 bullets per selected project; if space is tight, fewer projects are kept.
- Save outputs locally in `data/outputs/<job_id>/<timestamp>/` with `resume.tex`, `resume.pdf`, `report.json`.
- Exposes deterministic ATS audit/extension APIs:
  - mirror parsing + parse quality scoring
  - ATS linter with structured issue codes
  - ATS-safe deterministic renderers (`DOCX`, text-layer `PDF`, `TXT`)
  - contact/sensitive data validation
  - timeline normalization + overlap/duration analysis
  - hard/soft skill normalization + JD requirement graph
  - explainable hybrid match scoring
  - grounded patch generation/application + version compare
- Includes an interactive audit page (`/audit`) to run the full pipeline from one screen.

## Tech stack

- Python 3.11+
- FastAPI + Jinja2 templates
- Playwright (Chromium) + readability-lxml
- pdfplumber (+ optional OCR fallback via pytesseract)
- python-docx
- OpenAI API integration (`app/services/llm.py`) with strict JSON validation
- Local YAML/JSON filesystem storage (no DB)

## File tree

```text
.
├── app
│   ├── config.py
│   ├── main.py
│   ├── models.py
│   ├── public_api.py
│   ├── services
│   │   ├── ats_engine.py
│   │   ├── extractors.py
│   │   ├── latex.py
│   │   ├── llm.py
│   │   ├── repository.py
│   │   ├── scraper.py
│   │   └── tailoring.py
│   ├── static
│   │   └── style.css
│   ├── storage.py
│   ├── templates
│   │   ├── audit.html
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── job_detail.html
│   │   ├── jobs_list.html
│   │   ├── jobs_new.html
│   │   ├── resume.tex.j2
│   │   ├── resume_upload.html
│   │   ├── tailor_result.html
│   │   ├── vault_form.html
│   │   ├── vault_ingest.html
│   │   └── vault_list.html
│   └── utils.py
├── data
│   ├── jobs
│   ├── outputs
│   ├── resume
│   ├── sample
│   │   ├── base.yaml
│   │   ├── jobs/sample-backend-role/jd.txt
│   │   └── vault/items/*.yaml
│   ├── uploads
│   └── vault/items
├── schemas
│   ├── canonical_resume.schema.yaml
│   ├── job_record.schema.yaml
│   └── vault_item.schema.yaml
├── tests
│   ├── test_ats_engine.py
│   ├── test_constraints.py
│   ├── test_integration_tailor.py
│   ├── test_matching_quality.py
│   ├── test_pruning.py
│   ├── test_public_api_routes.py
│   ├── test_schema_validation.py
│   └── test_scoring.py
├── .gitignore
├── Makefile
├── requirements-dev.txt
└── requirements.txt
```

## Quick start (Docker, one command)

### 1) Prerequisites

- Docker Engine
- Docker Compose (v2 plugin)

### 2) Configure environment

Create or update `.env` in the project root (same variables as local mode):

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
ENABLE_OCR=0
# Optional:
# RESUME_APP_TOKEN=change_me
```

### 3) Start the app

```bash
make docker-up
```

Open: `http://localhost:8030`

Helpful commands:

```bash
make docker-logs
make docker-down
```

First build can take several minutes because the image installs TeX and Playwright Chromium.

## Manual setup (Ubuntu/Debian)

### 1) System packages

```bash
sudo apt update
sudo apt install -y \
  python3 python3-pip python3-venv \
  latexmk texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended lmodern \
  tesseract-ocr poppler-utils
```

`OCR` fallback is optional; it only runs if `ENABLE_OCR=1`.

### 2) Python deps

```bash
make install
```

### 3) Playwright browser

```bash
python3 -m playwright install chromium
# If needed on fresh servers:
python3 -m playwright install-deps chromium
```

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | For tailoring | unset | Enables LLM-based tailoring/extraction |
| `OPENAI_MODEL` | No | `gpt-4.1-mini` | OpenAI model name |
| `ENABLE_OCR` | No | `0` | Enable pytesseract OCR fallback for short PDF extraction |
| `RESUME_APP_TOKEN` | No | unset | If set, all POST routes require header token |

If `RESUME_APP_TOKEN` is set, POST requests must include:
- Header: `X-Resume-Token: <token>`

## Run

```bash
make run
```

Server listens on:
- `0.0.0.0:8030`

## Access over Tailscale

1. Install and connect Tailscale on the Linux host.
2. From another device on your tailnet, open:
   - `http://<tailscale-hostname-or-ip>:8030`
3. Keep host firewall open for `8030/tcp` within tailnet policy.

## App routes

- `GET /` dashboard
- `GET/POST /resume/upload` upload + parse resume
- `POST /resume/save` save canonical resume YAML + sync into vault
- `POST /resume/sync-vault` manual base->vault resync
- `GET /audit` audit runner page
- `POST /audit/run` execute full ATS audit pipeline
- `GET /vault` list vault items
- `GET /vault/ingest` ingest from pasted notes/uploaded file
- `POST /vault/ingest/parse` parse to vault YAML (review before save)
- `GET/POST /vault/new` create vault item
- `GET/POST /vault/{item_id}` edit vault item
- `GET /jobs` list jobs
- `GET /jobs/new` ingest form
- `POST /jobs/ingest` ingest job URL or pasted JD text
- `GET /jobs/{job_id}` edit JD + tailor controls
- `POST /jobs/{job_id}/jd` save JD edits
- `POST /jobs/{job_id}/tailor` run tailoring
- `GET /outputs/{job_id}/{timestamp}/resume.pdf` download output PDF
- `GET /outputs/{job_id}/{timestamp}/{artifact}` download generated artifacts (`ats_resume.pdf`, `ats_resume.docx`, `ats_resume.txt`, `bundle.zip`, etc.)

### ATS extension API routes

- `POST /api/upload_resume`
- `POST /api/upload_job_description`
- `POST /api/lint_resume`
- `POST /api/parse_mirror`
- `POST /api/build_canonical`
- `POST /api/score_parse_quality`
- `POST /api/score_match`
- `POST /api/generate_patches`
- `POST /api/apply_patches`
- `POST /api/render_outputs`
- `GET /api/export_bundle/{job_id}/{timestamp}`
- `POST /api/compare_versions`

## Data layout

- Canonical resume: `data/resume/base.yaml`
- Vault items: `data/vault/items/<item_id>.yaml` (user-created UUIDs plus deterministic `base_*` IDs from base-resume sync)
- Jobs: `data/jobs/<job_id>/job.yaml` + `data/jobs/<job_id>/jd.txt`
- Tailored outputs: `data/outputs/<job_id>/<timestamp>/resume.tex|resume.pdf|report.json`

## Tests

```bash
make test
```

Includes:
- schema validation
- scoring behavior
- constraint enforcement (no new tech / metrics)
- one-page pruning by bullet caps
- integration tailoring in both modes + render + mock PDF compile

## Notes

- No authentication by default (assumes trusted Tailscale network).
- If OpenAI key is missing, app runs but tailoring is disabled with clear UI errors.
- LaTeX output filename is always `resume.pdf` inside each output run directory.
- Project sectioning: use `section: projects` or `section: minor_projects` on canonical projects, or add vault tag `section:minor_projects`.
