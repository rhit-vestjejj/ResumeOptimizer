# Local Resume Tailor (FastAPI, Local Filesystem)

Local-only resume tailoring web app for Linux servers, designed to be accessed over Tailscale. The app binds to `0.0.0.0:8030`.

## What it does

- Optionally upload/refresh a resume (`PDF`/`DOCX`) to update canonical profile data (`data/resume/base.yaml`).
- Tailoring selection is vault-only for the MVP flow; vault evidence drives what gets included.
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
│   ├── eval
│   │   ├── cases/*.yaml
│   │   ├── fixtures/base.yaml
│   │   ├── fixtures/vault/items/*.yaml
│   │   ├── jds/*.txt
│   │   └── results/*.json
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
│   ├── eval_case.schema.yaml
│   ├── eval_result.schema.yaml
│   ├── job_record.schema.yaml
│   └── vault_item.schema.yaml
├── tests
│   ├── test_ats_engine.py
│   ├── test_selection_benchmark.py
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
APP_SECRET_KEY=change_me_to_a_long_random_value
# Optional:
# APP_ENV=prod
# RESUME_APP_TOKEN=change_me
# SESSION_COOKIE_SECURE=1
# ALLOW_SELF_SIGNUP=0
# MAX_UPLOAD_MB=10
# BOOTSTRAP_USER_EMAIL=admin@example.com
# BOOTSTRAP_USER_PASSWORD=change_me_please
```

Keep `.env` placeholder-only in shared code and never commit real API keys/tokens.

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
| `APP_ENV` | No | `dev` | Runtime environment (`dev`/`test`/`prod`) used for safety validation |
| `APP_SECRET_KEY` | For auth sessions | `dev-change-me` | Signs login session cookies |
| `SESSION_COOKIE_NAME` | No | `resume_session` | Auth session cookie name |
| `SESSION_TTL_SECONDS` | No | `604800` | Session expiration in seconds |
| `SESSION_COOKIE_SECURE` | No | `0` | Set cookie `Secure` flag (enable in HTTPS prod) |
| `ALLOW_SELF_SIGNUP` | No | `0` | Allow users to self-register (`1`) or enforce invite-only (`0`) |
| `MAX_UPLOAD_MB` | No | `10` | Maximum upload size in MB for resume/vault/audit upload routes |
| `SQLITE_PATH` | No | `data/app.db` | SQLite path for user/auth data |
| `BOOTSTRAP_USER_EMAIL` | No | unset | Optional startup account bootstrap email |
| `BOOTSTRAP_USER_PASSWORD` | No | unset | Optional startup account bootstrap password |

If `RESUME_APP_TOKEN` is set, POST requests must include:
- Header: `X-Resume-Token: <token>`

When `APP_ENV` is set to a non-dev value (for example `prod`), `APP_SECRET_KEY` must be changed from `dev-change-me`.

### Invite-only account provisioning

Default mode is invite-only (`ALLOW_SELF_SIGNUP=0`). Create friend accounts with:

```bash
.venv/bin/python scripts/create_user.py --email friend@example.com
```

Or provide password inline:

```bash
.venv/bin/python scripts/create_user.py --email friend@example.com --password "change_me_please"
```

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

- `GET/POST /auth/login` sign in
- `GET/POST /auth/register` create account (only when `ALLOW_SELF_SIGNUP=1`)
- `POST /auth/logout` sign out
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
- `GET /healthz` lightweight liveness check
- `GET /readyz` readiness check (paths + sqlite connectivity)

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

- User-scoped canonical resume: `data/users/<user_id>/resume/base.yaml`
- User-scoped vault items: `data/users/<user_id>/vault/items/<item_id>.yaml`
- User-scoped jobs: `data/users/<user_id>/jobs/<job_id>/job.yaml` + `jd.txt`
- User-scoped outputs: `data/users/<user_id>/outputs/<job_id>/<timestamp>/resume.tex|resume.pdf|report.json`
- SQLite auth/user DB: `data/app.db` (override with `SQLITE_PATH`)

## Tests

```bash
make test
```

Run the full local release gate (tests + benchmark):

```bash
make check
```

Run offline selection benchmark gate:

```bash
make eval
```

`make eval` runs labeled cases from `data/eval/cases`, computes:
- selection precision/recall/F1
- required-term coverage
- unsupported-claim rate

and writes a timestamped JSON result file to `data/eval/results/`.

Default aggregate gate thresholds:
- precision `>= 0.72`
- recall `>= 0.82`
- F1 `>= 0.76`
- required-term coverage `>= 0.90`
- unsupported-claim rate `<= 0.00`

Includes:
- schema validation
- scoring behavior
- constraint enforcement (no new tech / metrics)
- one-page pruning by bullet caps
- integration tailoring in both modes + render + mock PDF compile

## Release Gate

- CI workflow: `.github/workflows/ci.yml` runs `make test` and `make eval` on push and pull request.
- Local equivalent before deploying: `make check`.

## Notes

- Session-based authentication is required for app and API routes (except `/auth/*` and static assets).
- Startup bootstrap uses FastAPI lifespan handlers (not deprecated startup events).
- If OpenAI key is missing, app runs but tailoring is disabled with clear UI errors.
- LaTeX output filename is always `resume.pdf` inside each output run directory.
- Project sectioning: use `section: projects` or `section: minor_projects` on canonical projects, or add vault tag `section:minor_projects`.

## Troubleshooting

- Login/signup confusion: if self-signup is disabled (`ALLOW_SELF_SIGNUP=0`), new users must be provisioned via `scripts/create_user.py`.
- Upload errors: ensure files are under `MAX_UPLOAD_MB` and in supported formats.
- Tailoring disabled: set `OPENAI_API_KEY` and restart.
- Debug support: include the request ID shown in the app footer when reporting issues.

## Known limitations (MVP)

- No background worker queue; heavy operations run inline per request.
- Invite-only provisioning is CLI-based (no admin UI yet).
- External scraping quality depends on source website structure and anti-bot defenses.
