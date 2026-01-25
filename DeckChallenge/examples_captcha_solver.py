#!/usr/bin/env python3
"""
Example script showing how to use the captcha solver independently.
"""

from pathlib import Path
from captcha_solver import (
    OCRCaptchaSolver,
    ManualCaptchaSolver,
    TwoCaptchaSolver,
    get_solver,
)

def demo_ocr():
    """Demo OCR-based solving."""
    print("=== OCR Captcha Solver Demo ===")
    try:
        solver = get_solver({"method": "ocr"})
        print(f"Solver type: {type(solver).__name__}")
        
        # Example: solver.solve({"type": "image", "image_path": "/path/to/captcha.png"})
        print("Use: solver.solve({'type': 'image', 'image_path': '/path/to/image.png'})")
    except ImportError as e:
        print(f"OCR not available: {e}")
        print("Install with: pip install pytesseract pillow")
        print("And: brew install tesseract (macOS) or apt-get install tesseract-ocr (Ubuntu)")

def demo_2captcha():
    """Demo 2captcha service."""
    print("\n=== 2Captcha Service Demo ===")
    try:
        solver = get_solver({"method": "2captcha", "api_key": "demo_key"})
        print(f"Solver type: {type(solver).__name__}")
        print("Set CAPTCHA_API_KEY environment variable for actual use")
        print("Sign up: https://2captcha.com")
    except ImportError as e:
        print(f"Requests library not available: {e}")
        print("Install with: pip install requests")

def demo_manual():
    """Demo manual solving."""
    print("\n=== Manual Captcha Solver Demo ===")
    solver = get_solver({"method": "manual"})
    print(f"Solver type: {type(solver).__name__}")
    print("This solver will prompt user for captcha solution")

if __name__ == "__main__":
    print("Captcha Solver Examples\n")
    demo_ocr()
    demo_2captcha()
    demo_manual()
    
    print("\n=== Using with Playwright ===")
    print("""
from captcha_solver import get_solver, solve_captcha_on_page
from playwright.sync_api import sync_playwright

solver = get_solver({"method": "ocr"})

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://example.com")
    
    # Detect and solve captcha
    solved = solve_captcha_on_page(page, solver, inject_solution=True)
    if solved:
        print("Captcha solved!")
    
    browser.close()
    """)
