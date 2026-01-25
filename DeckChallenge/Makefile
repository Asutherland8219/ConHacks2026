VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

CREDS ?= creds.json
HEADLESS ?= 0
# Default arguments for the login scraper.
# The default is to use the browser-based captcha solver.
ARGS ?= --captcha-solver browser
COUNT ?= 5

HEADLESS_FLAG := $(if $(filter 1 true yes on,$(HEADLESS)),--headless,--no-headless)

.PHONY: venv install browsers run clean collect captcha-stats

$(PYTHON):
	python3 -m venv $(VENV)

venv: $(PYTHON)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Install Playwright browser binaries (run once).
browsers: install
	$(PYTHON) -m playwright install chromium

# Run the scraper with creds; override CREDS/ARGS/HEADLESS to tweak behavior.
# Examples:
#   make run -s                  # show browser (default with browser captcha solver)
#   make run HEADLESS=1          # run headless with browser solver
#   make run ARGS="--screenshot login.png"
#   make run ARGS="--captcha-solver ocr"  # use OCR solver instead
run: install
	$(PYTHON) src/login_scraper.py --creds $(CREDS) $(HEADLESS_FLAG) $(ARGS)

# Collect captcha samples for training dataset
# Usage:
#   make collect COUNT=10        # collect 10 samples (default headless)
#   make collect COUNT=5 HEADLESS=0  # show browser while collecting
collect: install
	$(PYTHON) collect_captchas.py --count $(COUNT) $(HEADLESS_FLAG)

# Show captcha database statistics
captcha-stats: install
	$(PYTHON) -c "import sys; sys.path.insert(0, 'src'); from captcha_database import CaptchaDatabase; CaptchaDatabase().print_statistics()"

clean:
	rm -rf $(VENV)
