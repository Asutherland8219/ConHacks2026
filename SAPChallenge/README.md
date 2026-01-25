# Lost & Found Matcher

Privacy-first lost-item matcher with a Python backend and Vue frontend.

## Project layout
- `backend/` – Python HTTP server + JSON storage (`backend/data/*`, uploads in `backend/uploads/`).
- `frontend/` – Vue pages (`index.html` for users, `assistant.html` for assistants, shared CSS in `assets/style.css`).

## Run the backend
```bash
cd SAPChallenge/backend
ASSISTANT_KEY=your-secret PORT=8000 python3 app.py
```
Then open `http://localhost:8000/` (user) or `http://localhost:8000/assistant` (assistant). The server also serves the frontend files.

## User flow (frontend/index.html)
- Submit inquiry with description + optional photos.
- Receive a tracking code and track status via the same page.
- If >5 matches, a follow-up question appears to narrow results.

## Assistant flow (frontend/assistant.html)
- Login with the shared `ASSISTANT_KEY`.
- Review inquiries and curated matches; approve/resolve with notes.
- Browse and add catalog items. Uploaded files (`/uploads/*`) are only accessible when logged in.

## Matching & fraud checks
- Attribute/term scoring (brand/category/color/location overlap) with confidence ranking.
- Duplicate contact/description flags for possible abuse.

## Notes
- Uses only Python stdlib; no external deps. Keep `ASSISTANT_KEY` strong if deployed.***
