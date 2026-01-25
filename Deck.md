# DeckChallenge repo overview

## Purpose
- Playwright-based login scraper for deckathon-concordia.com with multiple captcha-solving strategies (manual, OCR, 2captcha, browser automation, image-grid heuristics, logo template matching).

## Layout
- DeckChallenge/src/login_scraper.py: Playwright login flow, credential loading, captcha detection, solver integration.
- DeckChallenge/src/captcha_solver.py: solver registry + implementations (manual, OCR, 2captcha, browser auto-grid, intelligent logo match, similarity, size-based, OpenAI).
- DeckChallenge/src/captcha_image_analysis.py: grid detection + color-vs-image classification utilities.
- DeckChallenge/src/captcha_database.py: JSON-backed database of captcha solutions + stats.
- DeckChallenge/collect_captchas.py: repeated login attempts to collect samples + print stats.
- DeckChallenge/src/test_template_matching.py: OpenCV template matching used by the intelligent solver.
- DeckChallenge/src/debug_grid.py: renders debug overlays for grid analysis.
- DeckChallenge/ref/logo.png: reference logo for template matching.
- DeckChallenge/captcha_config.json.example: solver configuration example.
- DeckChallenge/requirements.txt and DeckChallenge/Makefile: dependencies and dev commands.
- Docs/Deck Dropout Challenge.pdf: challenge spec.

## How it runs
- Entry point: DeckChallenge/src/login_scraper.py or `make run`.
- Captcha snapshots default to `captcha_logs/` (relative to the working dir).
- Credentials come from CLI args, DeckChallenge/creds.json, or env vars (DECK_EMAIL/DECK_USERNAME + DECK_PASSWORD).

## Common commands
- `make install` (venv + deps)
- `make browsers` (Playwright Chromium)
- `make run` (login + solver)
- `make collect COUNT=...` (dataset capture)
- `make captcha-stats`

## Solver behavior
- login_scraper.handle_captcha detects captchas and calls solve_captcha_on_page.
- Browser-based solvers operate on the page and click tiles; others return a solution and get injected.

## Notes
- OpenAI solver requires an API key; 2captcha requires a key too.
- Grid analysis assumes 3x3 tiles and uses color uniformity to tag "logo" tiles vs solid color tiles.
