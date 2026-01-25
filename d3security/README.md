## D3 Security: Signal-to-Decision Challenge
_TUI for SOC triage_

### libraries / dependencies
- [`pandas`](https://pandas.pydata.org/) for data analysis
- [`textual`](https://github.com/Textualize/textual?tab=readme-ov-file) for tui interface
- [`fzf`](https://github.com/junegunn/fzf) for fuzzy finding files
- [`google-genai`](https://ai.google.dev/gemini-api/docs/quickstart) for the AI Analyst Assistant

### approaches used
- triage dashboard with priority & drill-down
- explainability-first triage
- AI analyst assistant

---

### operating application (linux)
- `export GEMINI_API_KEY="api_key_here"` to setup gemini api
- `python3 fzf.py` to run application

### related notes
- application will only detect (fuzzy-find) `.csv` files in current directory (local file-system) of python file
    - `CTRL-n / CTRL-p` to optionally navigate the `fzf` interface
- once file is selected, keymaps are indicated in TUI footer
- can use `TAB / SHIFT-TAB` and `h, j, k, l` to navigate (see TUI footer)
- can use arrow-keys for navigtaion or menu controls
- `f` to open filter pane
- `p` to go-to gemini prompt box
- `CTRL-q` to force-exit appliction

### future additions
- UTC to local time option for timestamps
- sort functionality
- additional filters
- read from network/cloud services
- improved file-system navigation