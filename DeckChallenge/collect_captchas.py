#!/usr/bin/env python3
"""
Collect captcha samples for building a training dataset.
Runs multiple login attempts and logs all captchas encountered.

Usage:
    python3 collect_captchas.py --count 10 --headless
    python3 collect_captchas.py --count 5 --no-headless --pause-on-each
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from login_scraper import main as login_main, parse_args, load_creds
from captcha_database import CaptchaDatabase


def collect_captchas(count: int, headless: bool = True, pause_on_each: bool = False, **kwargs):
    """Collect multiple captcha samples."""
    db = CaptchaDatabase()
    
    print(f"\n=== Captcha Collection Mode ===")
    print(f"Collecting {count} login attempts")
    print(f"Headless: {headless}")
    print()
    
    collected = 0
    for i in range(1, count + 1):
        print(f"\n--- Attempt {i}/{count} ---")
        
        # Build args for login
        creds_path = Path(__file__).parent / "creds.json"
        args = [
            f"--creds", str(creds_path),
            "--headless" if headless else "--no-headless",
            "--captcha-solver", "similarity",
            "--captcha-dir", "captcha_logs",
        ]
        
        try:
            # Run login
            result = login_main(args)
            
            if result == 0:
                print(f"✓ Attempt {i}: Login successful")
                collected += 1
            else:
                print(f"✗ Attempt {i}: Login failed")
            
            if pause_on_each and i < count:
                input(f"Press Enter to continue with attempt {i+1}...")
        
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
            break
        except Exception as e:
            print(f"✗ Attempt {i}: Error - {e}")
    
    # Print statistics
    print(f"\n=== Collection Complete ===")
    print(f"Successful logins: {collected}/{count}")
    db.print_statistics()


def main():
    parser = argparse.ArgumentParser(
        description="Collect captcha samples for training dataset"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of login attempts to collect (default: 5)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run headless (default)",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Show browser window",
    )
    parser.add_argument(
        "--pause-on-each",
        action="store_true",
        help="Pause between each attempt",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing captchas, don't collect new ones",
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Just show database statistics
        db = CaptchaDatabase()
        db.print_statistics()
    else:
        # Collect captchas
        collect_captchas(
            count=args.count,
            headless=args.headless,
            pause_on_each=args.pause_on_each,
        )


if __name__ == "__main__":
    main()
