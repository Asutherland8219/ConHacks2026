# Lost & Found Matcher (SAPChallenge)

Privacy‑first lost‑item intake and matching system with a zero‑dependency Python
backend and lightweight HTML/CSS frontend.

## Project Layout
- `backend/` – stdlib HTTP server, matching logic, JSON storage (`data/`), file
  uploads (`uploads/`).
- `frontend/` – user + assistant pages (`index.html`, `assistant.html`) and
  shared assets in `frontend/assets/`.
- `data/` – starter inventory JSON; public sample content for demos.
- `static/` – favicon and misc static assets.

## Quick Start
```bash
cd SAPChallenge/backend
cp config.example.json config.json   # edit values or use env vars
ASSISTANT_KEY=super-secret PORT=8000 python3 app.py
```
- User UI:            http://localhost:8000/
- Assistant console:  http://localhost:8000/assistant

### SMTP (optional, for status emails)
Set in `backend/config.json` or environment:
- `SMTP_HOST`, `SMTP_PORT` (587 or 465), `SMTP_USER`, `SMTP_PASS`, `MAIL_FROM`
- `ASSISTANT_KEY` (shared secret for assistant login)
- `PORT` (defaults to 8000)

## User Flow (frontend/index.html)
1) Visitor submits an inquiry with contact, category, description, and optional
   photos. A tracking code is returned.
2) If the system finds many similar items, it asks for one extra detail to
   disambiguate.
3) User can check status via the tracking code page.

## Assistant Flow (frontend/assistant.html)
- Login with `ASSISTANT_KEY` (sets an `assistant_token` cookie).
- Review inquiries with curated matches, inspect attached photos, and mark
  status (`matched`, `resolved`, `under_review`, `submitted`, `needs_info`).
- Add inventory items with verification prompts; browse all stored items.

## Matching Logic & Confidence Score
Defined in `backend/app.py::compute_match_score`:
- Token overlap between inquiry text (description, category, brand, color,
  location, follow‑up answer) and item text (name, category, brand, color,
  details, tags) → `+0.75` per shared token, recorded as “Shared terms”.
- Exact field boosts: brand `+2.0`, category `+1.5`, color `+1.0`.
- Location token overlap: `+0.5` per shared token (“Nearby location”).
- Score is normalized: `confidence = min(1.0, score / 6.0)`, rounded to 3
  decimals. Reasons for each boost accompany every match.
- `curate_matches` keeps matches with `confidence > 0`, sorts descending, and
  returns top 8 with their verification prompts.

## Fraud Signals
`detect_fraud_signals` flags:
- Reused contact info across inquiries.
- Heavy description overlap (>10 shared tokens) with previous inquiries.

## Data Files
- `backend/data/inventory.json` – sample catalog items with verification prompts.
- `backend/data/inquiries.json` – grows at runtime; gitignored in real use.
- `backend/uploads/` – saved photos, served only to authenticated assistants.

## API Endpoints (summary)
- `POST /api/inquiries` – submit inquiry; returns tracking `code` + `status`.
- `GET  /api/inquiries/{code}` – public status (PII stripped).
- `POST /api/inquiries/{code}/follow-up` – add extra detail, rerun matching.
- Assistant (cookie auth):
  - `POST /api/assistant/login` – body `{ key }`.
  - `GET  /api/assistant/inquiries` – list all.
  - `GET  /api/assistant/inquiries/{id}` – detail + enriched matches.
  - `POST /api/assistant/inquiries/{id}/status` – update status/notes/match id.
  - `GET/POST /api/assistant/inventory` – list or add items.

## Development Notes
- No external deps; runs on Python 3.8+.
- Persistence is JSON files; safe writes via temp file + move.
- Keep `ASSISTANT_KEY` strong; uploads are not virus‑scanned—use trusted hosts
  only.
- To reset demo data, clear `backend/data/inquiries.json` and `backend/uploads/`.

