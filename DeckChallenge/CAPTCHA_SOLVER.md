# Captcha Solver Integration

This project now includes automatic and guided captcha solving capabilities integrated with the login scraper.

## Supported Captcha Types

- **Image-based Text CAPTCHAs**: Solved via OCR (Tesseract)
- **Image Selection CAPTCHAs**: Guided solving with browser display (e.g., "Select all logos")
- **reCAPTCHA v2**: Solved via 2captcha service
- **hCaptcha**: Solved via 2captcha service  
- **Manual**: Fallback option requiring user interaction

## Solving Methods

### 1. Browser-Assisted Solving (Default)

Automatically detects image selection captchas and guides you through solving them. The browser stays open for you to interact with the captcha.

```bash
make run
# or explicitly
make run ARGS="--captcha-solver browser --no-headless"
```

**Best for:**
- "Select all images matching..." style captchas
- Multi-select image grids
- Complex visual captchas requiring human judgment

### 2. OCR-Based Solving

Automatically solves simple image-based text CAPTCHAs using Tesseract OCR.

**Requirements:**
- Tesseract OCR engine installed on your system
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt-get install tesseract-ocr`
  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

**Usage:**

```bash
# Basic OCR solving
make run ARGS="--captcha-solver ocr"

# With custom Tesseract path
make run ARGS="--captcha-solver ocr --captcha-config captcha_config.json"
```

**Configuration file (captcha_config.json):**
```json
{
  "method": "ocr",
  "pytesseract_path": "/usr/local/bin/tesseract"
}
```

### 3. 2Captcha Service

Solve any captcha type automatically using the 2captcha API service.

**Requirements:**
- 2captcha account: https://2captcha.com
- API key from your 2captcha account

**Setup:**
1. Create a 2captcha account and get your API key
2. Set the API key in environment variable or config file

**Usage:**

```bash
# Using environment variable
export CAPTCHA_API_KEY="your_2captcha_api_key"
make run ARGS="--captcha-solver 2captcha"

# Or pass directly
make run ARGS="--captcha-solver 2captcha --captcha-api-key your_api_key"

# Or use config file
make run ARGS="--captcha-solver 2captcha --captcha-config captcha_config.json"
```

**Configuration file (captcha_config.json):**
```json
{
  "method": "2captcha",
  "api_key": "your_2captcha_api_key"
}
```

## Installation

The required dependencies are already in `requirements.txt`:

```bash
make install
```

For OCR support, also install Tesseract:
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt-get install tesseract-ocr`
- Windows: https://github.com/UB-Mannheim/tesseract/wiki

## Examples

### Run with OCR solver:
```bash
make run ARGS="--captcha-solver ocr --headless"
```

### Run with 2captcha service (from environment variable):
```bash
export CAPTCHA_API_KEY="your_key"
make run ARGS="--captcha-solver 2captcha"
```

### Run with manual solving on pause:
```bash
make run ARGS="--pause-on-captcha --headless"
```

### Use custom config file:
```bash
make run ARGS="--captcha-config my_captcha_config.json"
```

## How It Works

1. **Detection**: Scans the page for common captcha patterns (reCAPTCHA, hCaptcha, image elements)
2. **Extraction**: Extracts captcha details (sitekey, image data, etc.)
3. **Solving**: Uses the configured solver method to get the solution
4. **Injection**: Automatically fills the captcha solution into the form
5. **Submission**: Continues with form submission

## Solver Priority

The solving method is determined by CLI arguments:

1. `--captcha-solver` flag (manual, ocr, 2captcha)
2. `--captcha-config` JSON file
3. `CAPTCHA_API_KEY` environment variable (for 2captcha)

### 3. 2Captcha Service

Solve any captcha type automatically using the 2captcha API service.

**Requirements:**
- 2captcha account: https://2captcha.com
- API key from your 2captcha account

**Setup:**
1. Create a 2captcha account and get your API key
2. Set the API key in environment variable or config file

**Usage:**

```bash
# Using environment variable
export CAPTCHA_API_KEY="your_2captcha_api_key"
make run ARGS="--captcha-solver 2captcha"

# Or pass directly
make run ARGS="--captcha-solver 2captcha --captcha-api-key your_api_key"

# Or use config file
make run ARGS="--captcha-solver 2captcha --captcha-config captcha_config.json"
```

**Configuration file (captcha_config.json):**
```json
{
  "method": "2captcha",
  "api_key": "your_2captcha_api_key"
}
```

### 4. Manual Solving (Fallback)

If other solvers fail or are not available, the system falls back to manual solving with browser interaction.

```bash
make run ARGS="--captcha-solver manual"
```

## Installation

The required dependencies are already in `requirements.txt`:

```bash
make install
```

For OCR support, also install Tesseract:
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt-get install tesseract-ocr`
- Windows: https://github.com/UB-Mannheim/tesseract/wiki

## Examples

### Run with browser-assisted solver (default, shows browser):
```bash
make run
```

### Run with browser solver headless:
```bash
make run ARGS="--captcha-solver browser --headless"
```

### Run with OCR solver:
```bash
make run ARGS="--captcha-solver ocr --headless"
```

### Run with 2captcha service:
```bash
export CAPTCHA_API_KEY="your_key"
make run ARGS="--captcha-solver 2captcha --headless"
```

### Use custom config file:
```bash
make run ARGS="--captcha-config my_captcha_config.json"
```

## How It Works

1. **Detection**: Scans the page for common captcha patterns
   - Image selection prompts ("Select all logos", etc.)
   - reCAPTCHA/hCaptcha widgets
   - Text-based image captchas

2. **Extraction**: Extracts relevant details
   - Prompt text for image selection
   - Sitekeys for reCAPTCHA/hCaptcha
   - Image data for text captchas

3. **Selection Based on Type**:
   - **Image selection**: Browser automation + manual guidance
   - **Text images**: OCR analysis
   - **Widget captchas**: 2captcha API

4. **Injection**: Automatically fills the captcha solution into the form

5. **Submission**: Continues with form submission

## Solver Comparison

| Solver | Text CAPTCHA | Image Selection | reCAPTCHA | hCaptcha | Speed |
|--------|--------------|-----------------|-----------|----------|-------|
| Browser | ✓ (OCR) | ✓✓ (Guided) | ✓ | ✓ | Fast |
| OCR | ✓✓ (Automatic) | ✗ | ✗ | ✗ | Fast |
| 2Captcha | ✓✓ | ✓ | ✓✓ | ✓✓ | Slow |
| Manual | ✓ | ✓ | ✓ | ✓ | Very Slow |

## Configuration Options

### Browser Method
```json
{
  "method": "browser"
}
```
Recommended for image selection captchas that require visual judgment.

### OCR Method
```json
{
  "method": "ocr",
  "pytesseract_path": "/usr/local/bin/tesseract"
}
```
Fast but only works for text-based image captchas.

### 2Captcha Method
```json
{
  "method": "2captcha",
  "api_key": "your_api_key"
}
```
Works for all types but costs money.

## Environment Variables

- `CAPTCHA_API_KEY` - 2Captcha API key
- `DECK_EMAIL` - Login email
- `DECK_PASSWORD` - Login password
- Ensure Tesseract is installed correctly
- Check that image quality is adequate
- Try with better-quality captcha images

### 2captcha service errors
- Verify your API key is valid
- Check your 2captcha account balance
- Ensure your internet connection is stable

### Captcha not detected
- Use `--dump-candidates` flag to see available form fields
- Check if captcha is in an iframe (may need manual solving)
- Review the captcha_logs directory for debug screenshots

## API Usage

You can also use the captcha solver module programmatically:

```python
from captcha_solver import get_solver, solve_captcha_on_page

# Initialize solver
solver = get_solver({"method": "ocr"})

# Detect and solve on page
solved = solve_captcha_on_page(page, solver, inject_solution=True)
if solved:
    print("Captcha solved and injected!")
```

## Captcha Logs

All detected captchas are logged to the `captcha_logs/` directory with:
- Screenshot of the captcha
- JSON metadata (timestamp, URL, SHA256 hash)

This helps with debugging and auditing.
