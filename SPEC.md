You are Codex acting as a senior engineer. Build a local-only resume tailoring webapp running on a Linux server, accessed via browser over Tailscale. MUST listen on port 8030. Provide production-quality code, a clear file tree, and setup instructions.

GOALS
1) User uploads a resume (PDF or DOCX). The system extracts text, then converts it into a canonical structured format (YAML/JSON). This canonical representation is the source of truth.
2) The system maintains a local “Experience Vault” using a schema (YAML) where each item can be: project, job, club, coursework, skill cluster, award, etc. User can add/edit items via web UI.
3) User supplies a job posting by URL. The system scrapes the page, extracts the job description text, and uses that to tailor the resume.
4) The system generates a tailored one-page ATS-friendly resume as LaTeX and compiles to PDF. It drops output files locally (server filesystem) and provides a download link in the UI.
5) The system supports two modes:
   - HARD_TRUTH: never invent. Only uses facts from canonical resume + vault. If a bullet would require unsupported claims, it refuses/omits and logs warnings.
   - FUCK_IT: aggressive rewriting/reordering for ATS alignment but STILL NO NEW FACTS. It can phrase things more strongly, but must not fabricate metrics/technologies/roles.
6) The system does NOT require authentication because it’s behind Tailscale, but include a simple optional header token env var (RESUME_APP_TOKEN). If set, require it for all POST routes.

NON-GOALS
- Perfectly preserving original PDF layout. Use a default clean LaTeX template and keep it strictly one page by design.
- Git/versioning for now.

TECH CHOICES (use these)
- Python 3.11+
- FastAPI for the web server
- Jinja2 for LaTeX templating
- latexmk + TeX Live to compile
- PDF text extraction: try pdfplumber first; fallback to pytesseract OCR ONLY if extracted text is empty or too short. Make OCR optional via env (ENABLE_OCR=1).
- DOCX extraction: python-docx
- Web scraping: Playwright (chromium) + readability-lxml (or equivalent) to extract main content. If scraping fails, UI must allow user to paste JD text as fallback.
- LLM: call OpenAI via environment variables (OPENAI_API_KEY, OPENAI_MODEL). Use one helper module llm.py with strict system prompts and JSON schema validation. If no key is set, still run but disable tailoring with a clear UI error.

DATA MODEL
Store all data locally under ./data
- data/resume/base.yaml (canonical resume)
- data/vault/items/<uuid>.yaml (vault items)
- data/jobs/<job_id>/job.yaml and jd.txt (scraped or pasted)
- data/outputs/<job_id>/<timestamp>/resume.tex + resume.pdf + report.json

Canonical resume schema (base.yaml) MUST include:
- identity: name, email, phone, location, links[]
- summary (optional)
- education[]: school, degree, major, minors[], gpa (string), dates (start/end), coursework[]
- experience[]: company, title, location, dates, bullets[]
- projects[]: name, link (optional), dates (optional), tech[], bullets[]
- skills: categories{category_name: [skills]}
- awards (optional)

Vault item schema items/<uuid>.yaml MUST include:
- type: project|job|club|coursework|award|skillset|other
- title
- dates (optional)
- tags[] (keywords)
- tech[] (optional)
- bullets[] (STAR-like; allow outcome/impact fields)
- links[] (optional)
- source_artifacts[] (optional file paths)

JOB INGEST
- POST /jobs/ingest with URL. Scrape and extract JD text.
- Save jd.txt + job.yaml (url, title if detected, company if detected, scraped_at)
- Provide /jobs/<job_id> page to review/edit extracted JD text.

TAILORING ALGORITHM
1) Parse JD text. Extract:
   - target role keywords
   - required skills/tools
   - nice-to-haves
   - responsibilities
2) Build candidate content pool from base resume + vault.
3) Select projects/experience/coursework that match JD. Use simple scoring (keyword overlap + tag overlap + tech overlap).
4) Rewrite bullets for selected items to align with JD terms without inventing facts.
5) Enforce constraints:
   - Never change dates, GPA, titles.
   - Never add technologies not present in the underlying item’s tech/tags/bullets.
   - Never add metrics unless present in source bullets or explicitly in structured fields.
6) Keep one page: limit bullet count per section, shorten bullets, prune lowest-scoring items. Provide deterministic rules for pruning.
7) Produce:
   - tailored canonical YAML (in memory)
   - resume.tex via Jinja template
   - compiled resume.pdf
   - report.json with: chosen items, keywords covered/missed, any warnings, mode used.

LATEX TEMPLATE
- Provide a default ATS-friendly template (single column, no icons, no tables, no graphics).
- Must compile with latexmk.
- Must be designed to stay within one page for typical content. Include “pruning” logic rather than relying on LaTeX overflow.
- Use a consistent neutral filename at output: "resume.pdf" (and store inside output folder). For downloads, content-disposition can be "resume.pdf" too to avoid “machine-made” naming.

WEB UI
- Simple HTML using Jinja2 templates or minimal frontend (no React).
Pages:
- / : dashboard (base resume status, vault count, create job ingest)
- /resume/upload : upload PDF/DOCX, run extraction, show extracted preview, allow user to edit canonical YAML in a textbox, save
- /vault : list items + add/edit
- /vault/new : create item form
- /vault/<uuid> : edit item form
- /jobs : list jobs
- /jobs/new : ingest form (URL) + fallback paste
- /jobs/<job_id> : job details + JD text editor + “Tailor resume” button
- /jobs/<job_id>/tailor : runs tailoring, shows “Download PDF” link + warnings (only show warnings, not full rationale prose)

OPERATIONS
- Provide a Makefile or scripts:
  - make install (pip deps)
  - make run (uvicorn on 0.0.0.0:8030)
- Provide README with:
  - TeX Live install commands for Ubuntu/Debian
  - Playwright install commands
  - Environment variables list
  - How to run behind Tailscale

TESTS
- Provide unit tests for:
  - schema validation
  - keyword scoring
  - constraint enforcement (no new tech, no new metrics)
  - pruning to one page (simulate by max bullet count)
- Provide an integration test that:
  - loads a sample base.yaml + 3 vault items + sample JD text
  - runs tailoring in both modes
  - produces resume.tex and a mock compile step (skip latexmk in CI if not available, but keep compile function)

DELIVERABLES
- Full codebase with clear structure.
- All schemas and templates.
- A small sample dataset in ./data/sample to demonstrate.
- Do not leave TODOs. If something is optional (OCR), implement it with feature flag.

IMPORTANT
- Bind to port 8030, host 0.0.0.0.
- Store everything locally. No external DB required; use filesystem + YAML.
- Be strict about JSON/YAML validation and defensive coding.

Now generate the project with code, file tree, and README.
