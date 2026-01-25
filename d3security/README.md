# D3 Security: Signal-to-Decision Challenge
_TUI for SOC triage_

## libraries / dependencies
- [`pandas`](https://pandas.pydata.org/) for data analysis
- [`textual`](https://github.com/Textualize/textual?tab=readme-ov-file) for tui interface
- [`fzf`](https://github.com/junegunn/fzf) for fuzzy finding files
- [`google-genai`](https://ai.google.dev/gemini-api/docs/quickstart) for the AI Analyst Assistant

## approaches used
### triage dashboard with priority & drill-down
- calculates a `priority_score` using weighted inputs from Severity, Confidence, and External Source status to automatically bubble critical threats to the top.
- Integrated `FilterPane` allows analysts to rapidly slice data by Entity, Source IP, or Time Range, reducing noise and focusing investigations.
### explainability-first triage
- The `ExplanationPanel` explicitly lists the contributing factors for a high score (e.g., "CRITICAL SEVERITY" or "EXTERNAL SOURCE"), ensuring the analyst understands why an event was flagged.
- Highlights whether defensive measures succeeded (e.g., "DEFENSIVE SUCCESS: Blocked"), preventing wasted cycles on already-mitigated threats.
### AI-analyst assistant
- model's context window is fed the full dataset (open .csv file), ensuring answers are grounded in real data (preventing hallucinations).
- Requires explicit user confirmation before executing API calls, preserving cost control and keeping the human as the decision-maker.
- System instructions enforce outputs that cite specific event IDs as evidence.

## operating application (linux)
- `export GEMINI_API_KEY="api_key_here"` to setup gemini api
- `python3 fzf.py` to run application

## related notes
- application will only detect (fuzzy-find) `.csv` files in current directory (local file-system) of python file
    - `CTRL-n / CTRL-p` to optionally navigate the `fzf` interface
- once file is selected, keymaps are indicated in TUI footer
- can use `TAB / SHIFT-TAB` and `h, j, k, l` to navigate (see TUI footer)
- can use arrow-keys for navigtaion or menu controls
- `f` to open filter pane
- `p` to go-to gemini prompt box
- `CTRL-q` to force-exit appliction

## future additions
- UTC to local time option for timestamps
- sorting functionality
- additional filters
- read from network/cloud services
- improved file-system navigation