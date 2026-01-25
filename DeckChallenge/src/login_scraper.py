#!/usr/bin/env python3
"""
Playwright helper that signs into https://deckathon-concordia.com/login.

Credentials are read from (priority order):
1) CLI flags (--email/--password)
2) creds.json (--creds flag, defaults to DeckChallenge/creds.json)
3) Environment variables (DECK_EMAIL or DECK_USERNAME, and DECK_PASSWORD)

Use --dump-candidates/--dump-only to print form field IDs/names/labels when
selectors need manual hints. Captcha detection is supported, but solving
requires a human step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import urlparse

from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from captcha_solver import get_solver, solve_captcha_on_page

# Default login page; can be overridden with --url if needed.
DEFAULT_LOGIN_URL = "https://deckathon-concordia.com/login"

Selector = str

CAPTCHA_SELECTORS: Tuple[str, ...] = (
    "iframe[src*='recaptcha']",
    "iframe[src*='hcaptcha']",
    "div.g-recaptcha",
    "div.h-captcha",
    "img[src*='captcha']",
    "input[name*='captcha']",
    "input[id*='captcha']",
)

def human_pause(page, min_ms: int = 80, max_ms: int = 220) -> None:
    """Pause for a short, human-like delay."""
    delay = random.uniform(min_ms, max_ms)
    page.wait_for_timeout(int(delay))


def human_move_mouse_to(page, element) -> bool:
    """Move mouse near the element center with slight jitter."""
    try:
        element.scroll_into_view_if_needed()
    except Exception:
        pass
    try:
        box = element.bounding_box()
    except Exception:
        box = None
    if not box:
        return False
    jitter_x = random.uniform(-box["width"] * 0.2, box["width"] * 0.2)
    jitter_y = random.uniform(-box["height"] * 0.2, box["height"] * 0.2)
    target_x = box["x"] + box["width"] / 2 + jitter_x
    target_y = box["y"] + box["height"] / 2 + jitter_y
    steps = random.randint(6, 14)
    page.mouse.move(target_x, target_y, steps=steps)
    return True


def human_click(page, element, label: str | None = None) -> None:
    """Click an element with a mouse move and short delay."""
    if label:
        print(f"Clicking {label} (humanized)")
    human_move_mouse_to(page, element)
    human_pause(page)
    element.click()
    human_pause(page)


def human_type(page, element, text: str, label: str | None = None) -> None:
    """Click and type with per-character delay."""
    if label:
        print(f"Typing into {label} (humanized)")
    human_click(page, element)
    delay = random.randint(35, 90)
    page.keyboard.type(text, delay=delay)
    human_pause(page, 120, 280)


def wait_for_first(handle, selectors: Iterable[Selector], timeout: float, state: str = "attached"):
    """Try selectors in order and return the first match within the timeout."""
    deadline = time.time() + timeout
    last_exc: Exception | None = None

    for selector in selectors:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            return handle.wait_for_selector(
                selector,
                timeout=int(remaining * 1000),
                state=state,
            )
        except PlaywrightTimeoutError as exc:
            last_exc = exc

    raise last_exc or PlaywrightTimeoutError("No matching element found for provided selectors.")


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return cleaned or "root"


def get_origin(url: str) -> str | None:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def attach_api_logger(page, state: dict, log_dir: str | None = None) -> None:
    """Capture key API responses and optionally log them to disk."""
    out_dir = Path(log_dir).expanduser() if log_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    def handle_response(response) -> None:
        url = response.url
        path = urlparse(url).path
        if not any(segment in path for segment in ("/login", "/mfa", "/user-info", "/payment", "/dropout", "/captcha")):
            return
        payload = None
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                payload = response.json()
            except Exception:
                payload = None

        if isinstance(payload, dict):
            for key in ("auth_token", "mfa_authenticated_token", "captcha_solved_token"):
                token = payload.get(key)
                if token:
                    state[key] = token
                    print(f"Captured {key} from {path}")

        origin = get_origin(url)
        if origin:
            state["api_origin"] = origin

        if out_dir:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            method = response.request.method
            filename = f"{stamp}_{method}_{safe_filename(path)}_{response.status}.json"
            record = {
                "url": url,
                "status": response.status,
                "method": method,
                "path": path,
            }
            if payload is not None:
                record["json"] = payload
            (out_dir / filename).write_text(json.dumps(record, indent=2), encoding="utf-8")

    page.on("response", handle_response)


def dump_candidates(page) -> None:
    """Print inputs/buttons with their key attributes for debugging selectors."""

    def fmt(el):
        get = el.get_attribute
        text = (el.text_content() or "").strip()
        return (
            f"type={get('type') or 'n/a'} "
            f"id={get('id') or 'n/a'} "
            f"name={get('name') or 'n/a'} "
            f"placeholder={get('placeholder') or 'n/a'} "
            f"aria-label={get('aria-label') or 'n/a'} "
            f"text={text or 'n/a'}"
        )

    print(f"Current URL: {page.url}")

    frames = [page.main_frame] + [f for f in page.frames if f is not page.main_frame]
    for idx, frame in enumerate(frames, 1):
        label = "main document" if frame is page.main_frame else f"frame[{idx}]"
        print(f"\n== Input fields ({label}) ==")
        try:
            inputs = frame.query_selector_all("input")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not read inputs for {label}: {exc}")
            inputs = []
        for i, el in enumerate(inputs, 1):
            print(f"[{i}] {fmt(el)}")

        print(f"\n== Buttons ({label}) ==")
        try:
            buttons = frame.query_selector_all("button")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not read buttons for {label}: {exc}")
            buttons = []
        for i, el in enumerate(buttons, 1):
            print(f"[{i}] {fmt(el)}")


def dump_dom(page, out_path: str, include_frames: bool = False) -> None:
    """Write the current page DOM (and optional frame DOMs) to disk."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(out_path)
    if out_path.endswith(os.sep) or out.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        main_path = out / f"dom_{stamp}.html"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        main_path = out
    main_path.write_text(page.content(), encoding="utf-8")
    print(f"Saved DOM to {main_path}")

    if not include_frames:
        return

    frames_dir = main_path.parent / f"{main_path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames_info = []
    for idx, frame in enumerate(page.frames):
        if frame is page.main_frame:
            continue
        try:
            content = frame.content()
        except Exception as exc:  # noqa: BLE001
            print(f"Could not read frame[{idx}] content: {exc}")
            continue
        frame_path = frames_dir / f"frame_{idx}.html"
        frame_path.write_text(content, encoding="utf-8")
        frames_info.append({"index": idx, "url": frame.url, "path": str(frame_path)})
    if frames_info:
        (frames_dir / "frames.json").write_text(json.dumps(frames_info, indent=2), encoding="utf-8")
        print(f"Saved {len(frames_info)} frame DOM files to {frames_dir}")


def dump_dom_interactive(
    page,
    base_path: str | None,
    suffix: str,
    include_frames: bool = False,
    prompt: bool = True,
) -> None:
    """Optionally prompt for Enter, then dump DOM to a suffixed file."""
    if not base_path:
        return
    base = Path(base_path)
    if base_path.endswith(os.sep) or base.is_dir():
        base.mkdir(parents=True, exist_ok=True)
        target = base / f"dom_{suffix}.html"
    else:
        target = base.with_name(f"{base.stem}_{suffix}{base.suffix}")
    if prompt:
        print(f"Press Enter to dump DOM for {suffix}...")
        input()
    dump_dom(page, str(target), include_frames=include_frames)


def fetch_user_info(request_context, api_origin: str | None, auth_token: str | None, timeout: float) -> dict | None:
    """Fetch user-info from the backend to decide the next step."""
    if not api_origin or not auth_token:
        return None
    try:
        resp = request_context.get(
            f"{api_origin}/user-info",
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=int(timeout * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to fetch user-info: {exc}")
        return None
    if not resp.ok:
        print(f"user-info request failed: {resp.status}")
        return None
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to parse user-info JSON: {exc}")
        return None


def extract_balance(user_info: dict) -> float | None:
    finance = user_info.get("finance") or {}
    if isinstance(finance, dict) and "balance" in finance:
        try:
            return float(finance["balance"])
        except (TypeError, ValueError):
            return None
    try:
        base = float(finance.get("base_balance", 0))
        paid = float(finance.get("amount_paid", 0))
        classes = user_info.get("classes") or []
        class_total = sum(float(c.get("cost", 0)) for c in classes if isinstance(c, dict))
        return base + class_total - paid
    except (TypeError, ValueError):
        return None


def navigate_to_dropout(page, timeout: float) -> bool:
    """Attempt to open the dropout flow to capture the DOM."""
    if click_button_by_text(page, "Student Dropout", timeout=timeout, label="student dropout"):
        return True
    origin = get_origin(page.url)
    if not origin:
        return False
    try:
        page.goto(f"{origin}/dropout", wait_until="domcontentloaded", timeout=int(timeout * 1000))
        print("Navigated to dropout page.")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to navigate to dropout page: {exc}")
        return False


def load_creds(path: str | None) -> Tuple[str | None, str | None]:
    """Load credentials from a JSON file."""
    if not path:
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, None
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in creds file {path}: {exc}", file=sys.stderr)
        return None, None

    user = data.get("email") or data.get("username") or data.get("netname")
    password = data.get("password")
    return user, password


def find_captcha_element(page):
    """Return (frame, element, selector) if a captcha indicator is found."""
    frames = [page.main_frame] + [f for f in page.frames if f is not page.main_frame]
    for frame in frames:
        for selector in CAPTCHA_SELECTORS:
            try:
                el = frame.query_selector(selector)
            except Exception:
                el = None
            if el:
                return frame, el, selector
    return None, None, None


def save_captcha_snapshot(page, out_dir: str | None, frame=None, element=None) -> str | None:
    """Save a captcha screenshot and metadata; returns the screenshot path."""
    if not out_dir:
        return None
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = out_path / f"captcha_{stamp}.png"

    page.screenshot(path=str(image_path), full_page=True)

    sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    meta = {
        "timestamp": stamp,
        "page_url": page.url,
        "frame_url": getattr(frame, "url", None),
        "screenshot": str(image_path),
        "sha256": sha256,
    }
    meta_path = out_path / f"captcha_{stamp}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Captcha snapshot saved: {image_path} (sha256={sha256})")
    return str(image_path)


def extract_otp_code(page) -> str | None:
    """Extract a 6-digit OTP code from the MFA toast/banner if present."""
    try:
        toast = page.query_selector("div:has-text('MFA Code Generated')")
    except Exception:
        toast = None

    if toast:
        try:
            code_el = toast.query_selector("p.text-3xl, p.font-mono")
        except Exception:
            code_el = None
        if code_el:
            code = (code_el.text_content() or "").strip()
            if code.isdigit() and len(code) == 6:
                print(f"Found OTP code in toast: {code}")
                return code

    try:
        code_el = page.query_selector("p.text-3xl")
    except Exception:
        code_el = None
    if code_el:
        code = (code_el.text_content() or "").strip()
        if code.isdigit() and len(code) == 6:
            print(f"Found OTP code in page: {code}")
            return code

    return None


def wait_for_otp_code(page, timeout: float) -> str | None:
    """Poll quickly for the OTP code with a short deadline."""
    deadline = time.monotonic() + min(timeout, 4.0)
    while time.monotonic() < deadline:
        code = extract_otp_code(page)
        if code:
            return code
        page.wait_for_timeout(120)
    return None


def handle_two_factor(page, timeout: float = 20.0) -> bool:
    """Fill OTP from toast and submit the Verify Code form."""
    try:
        otp_input = wait_for_first(
            page,
            selectors=["#otp", "input#otp", "input[name='otp']"],
            timeout=timeout,
            state="visible",
        )
    except Exception:
        print("OTP input not found; skipping MFA handling.")
        return False

    code = wait_for_otp_code(page, timeout=timeout)
    if not code:
        print("OTP code not available; MFA requires manual entry.")
        return False

    try:
        otp_input.fill(code)
        page.wait_for_timeout(60)
        page.keyboard.press("Tab")
        page.wait_for_timeout(60)
    except Exception:
        human_type(page, otp_input, code, label="otp")

    try:
        verify_button = wait_for_first(
            page,
            selectors=[
                "button:has-text('Verify Code')",
                "button:has(span:has-text('Verify Code'))",
                "button:has-text('Verify')",
            ],
            timeout=timeout,
            state="visible",
        )
    except Exception:
        print("Verify Code button not found.")
        return False

    for _ in range(5):
        try:
            if verify_button.is_enabled():
                break
        except Exception:
            break
        page.wait_for_timeout(80)

    if not verify_button.is_enabled():
        print("Verify Code button is disabled; MFA may need manual step.")
        return False

    human_click(page, verify_button, label="verify code")
    page.wait_for_timeout(600)
    return True


def click_button_by_text(page, text: str, timeout: float = 20.0, label: str | None = None) -> bool:
    """Click a button or link by visible text."""
    selectors = [
        f"button:has-text('{text}')",
        f"button:has(span:has-text('{text}'))",
        f"a:has-text('{text}')",
    ]
    try:
        el = wait_for_first(page, selectors=selectors, timeout=timeout, state="visible")
    except Exception:
        print(f"Could not find '{text}' button/link.")
        return False
    try:
        el.scroll_into_view_if_needed()
        page.wait_for_timeout(120)
    except Exception:
        pass
    human_click(page, el, label=label or text)
    return True


def handle_slider_challenge(page, timeout: float = 20.0) -> bool:
    """Attempt to solve a slider challenge if present."""
    def modal_is_visible() -> bool:
        try:
            modal = page.query_selector(".modal-overlay")
        except Exception:
            return False
        if not modal:
            return False
        try:
            return modal.is_visible()
        except Exception:
            return True

    selectors = [
        "div.slider-handle",
        "div.slider-container",
        "input[type='range']",
        "[role='slider']",
        "div[class*='slider']",
        "div:has-text('Slide')",
        "div:has-text('슬라이드')",
    ]
    combined_selector = ", ".join(selectors)
    for attempt in range(1, 4):
        try:
            slider = page.wait_for_selector(
                combined_selector,
                timeout=int(timeout * 1000),
                state="visible",
            )
        except Exception:
            print("Slider challenge not found.")
            return False

        try:
            tag = (slider.evaluate("el => el.tagName") or "").lower()
        except Exception:
            tag = ""

        if tag == "input":
            try:
                page.evaluate(
                    """el => {
                        const max = Number(el.max || 100);
                        el.value = String(max);
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }""",
                    slider,
                )
                print("Slider input set to max value.")
                page.wait_for_timeout(400)
                if not modal_is_visible():
                    return True
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to set slider input: {exc}")

        try:
            handle = page.query_selector(".modal-overlay .slider-handle") or page.query_selector("div.slider-handle")
        except Exception:
            handle = None
        if not handle:
            handle = slider
        try:
            container = (
                page.query_selector(".modal-overlay .slider-container")
                or page.query_selector("div.slider-container")
            )
        except Exception:
            container = None

        try:
            handle.scroll_into_view_if_needed()
        except Exception:
            pass

        try:
            box = handle.bounding_box()
        except Exception:
            box = None
        if not box:
            print("Slider bounding box not available.")
            return False
        if container:
            try:
                container_box = container.bounding_box()
            except Exception:
                container_box = None
        else:
            container_box = None

        start_x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        if container_box:
            end_x = container_box["x"] + container_box["width"] - (box["width"] / 2) - 4
        else:
            end_x = box["x"] + box["width"] * (6 + attempt * 2)
        if end_x <= start_x:
            end_x = start_x + max(box["width"] * 4, 120)

        print(f"Dragging slider attempt {attempt}: start=({start_x:.1f},{y:.1f}) end=({end_x:.1f},{y:.1f})")
        page.mouse.move(start_x, y, steps=12)
        page.mouse.down()
        page.wait_for_timeout(120)
        steps = 28
        for step in range(1, steps + 1):
            x = start_x + (end_x - start_x) * (step / steps)
            page.mouse.move(x, y)
            page.wait_for_timeout(8)
        page.mouse.up()
        page.wait_for_timeout(500)

        if not modal_is_visible():
            print("Slider drag completed; modal dismissed.")
            return True
        print("Slider modal still visible after drag.")

        try:
            modal_content = page.query_selector(".modal-overlay .modal-content") or page.query_selector(".modal-content")
        except Exception:
            modal_content = None
        if modal_content:
            try:
                content_box = modal_content.bounding_box()
            except Exception:
                content_box = None
            if content_box:
                start_x = content_box["x"] + content_box["width"] / 2
                y = content_box["y"] + content_box["height"] / 2
                end_x = start_x + max(content_box["width"] * 0.6, 160)
                print(f"Dragging modal fallback: start=({start_x:.1f},{y:.1f}) end=({end_x:.1f},{y:.1f})")
                page.mouse.move(start_x, y, steps=8)
                page.mouse.down()
                page.wait_for_timeout(120)
                steps = 22
                for step in range(1, steps + 1):
                    x = start_x + (end_x - start_x) * (step / steps)
                    page.mouse.move(x, y)
                    page.wait_for_timeout(8)
                page.mouse.up()
                page.wait_for_timeout(500)
                if not modal_is_visible():
                    print("Modal drag fallback dismissed the slider.")
                    return True
        page.wait_for_timeout(300)

    return False


def handle_payment_flow(
    page,
    timeout: float = 20.0,
    dump_dom_path: str | None = None,
    dump_dom_frames: bool = False,
) -> bool:
    """Navigate payment flow and handle secondary verification."""
    def dump_step(suffix: str) -> None:
        dump_dom_interactive(
            page,
            dump_dom_path,
            f"payment_{suffix}",
            include_frames=dump_dom_frames,
            prompt=False,
        )

    print("Starting payment flow...")
    dump_step("start")
    clicked = click_button_by_text(page, "Make a Payment", timeout=timeout, label="make payment")
    if not clicked:
        clicked = click_button_by_text(page, "Make Payment", timeout=timeout, label="make payment")
    if clicked:
        try:
            page.wait_for_url("**/payment**", timeout=int(timeout * 1000))
        except Exception:
            pass
    dump_step("after_make_payment")
    if "payment" not in page.url:
        try:
            link = page.query_selector("a[href='/payment']")
        except Exception:
            link = None
        if not link:
            try:
                origin = page.evaluate("() => location.origin")
            except Exception:
                origin = None
            if origin:
                page.goto(f"{origin}/payment", wait_until="domcontentloaded")
            else:
                print("Make a Payment link not found.")
                return False
        else:
            try:
                is_visible = link.is_visible()
            except Exception:
                is_visible = False
            if not is_visible:
                try:
                    origin = page.evaluate("() => location.origin")
                except Exception:
                    origin = None
                if origin:
                    page.goto(f"{origin}/payment", wait_until="domcontentloaded")
                else:
                    print("Make a Payment link not visible.")
                    return False
            else:
                human_click(page, link, label="make payment link")
    try:
        page.wait_for_url("**/payment**", timeout=int(timeout * 1000))
    except Exception:
        pass
    page.wait_for_timeout(1000)
    dump_step("after_payment_nav")

    if not click_button_by_text(page, "Continue to payment", timeout=timeout, label="continue to payment"):
        print("Continue to payment button not found.")
        return False
    page.wait_for_timeout(600)
    dump_step("after_continue")

    confirm_selector = ".modal-overlay, .slider-handle, #otp, input#otp, input[name='otp']"
    try:
        page.wait_for_selector(confirm_selector, timeout=int(timeout * 1000), state="attached")
    except Exception:
        print("No payment confirmation step detected after Continue to Payment.")
        return False

    otp_present = False
    try:
        otp_present = page.query_selector("#otp, input#otp, input[name='otp']") is not None
    except Exception:
        otp_present = False

    if otp_present:
        if handle_two_factor(page, timeout=timeout):
            print("Secondary MFA step completed.")
            page.wait_for_timeout(800)
        else:
            print("Secondary MFA step not completed.")
        dump_step("after_mfa")
    else:
        print("No secondary OTP prompt detected for payment step.")
        dump_step("no_mfa_detected")

    slider_ok = handle_slider_challenge(page, timeout=timeout)
    if slider_ok:
        print("Slider challenge completed.")
    else:
        print("Slider challenge failed.")
    dump_step("after_slider")
    if slider_ok:
        page.wait_for_timeout(1200)
        dump_step("after_slider_success")
    else:
        dump_step("after_slider_failure")
    return slider_ok


def handle_captcha(page, captcha_dir: str | None, pause_on_captcha: bool, solver=None) -> bool:
    """Detect, solve, and inject captcha solution."""
    frame, element, selector = find_captcha_element(page)
    if not element:
        print("No captcha detected.")
        return False

    print(f"Captcha detected (selector={selector}, frame_url={getattr(frame, 'url', None)})")
    save_captcha_snapshot(page, captcha_dir, frame=frame, element=element)
    
    if solver is None:
        print("No solver available. Manual solve required.")
        if pause_on_captcha:
            input("Solve the captcha in the browser, then press Enter to continue...")
        return True
    
    # Attempt to solve with configured solver
    return solve_captcha_on_page(page, solver, inject_solution=True)


def perform_login(
    page,
    url: str,
    email: str,
    password: str,
    timeout: float,
    already_loaded: bool = False,
    wait_after_load: float = 0.0,
    captcha_dir: str | None = None,
    pause_on_captcha: bool = False,
    captcha_solver=None,
) -> str:
    """Navigate to login page and submit credentials."""
    if not already_loaded:
        page.goto(url, wait_until="domcontentloaded")
    if wait_after_load > 0:
        page.wait_for_timeout(int(wait_after_load * 1000))

    email_input = wait_for_first(
        page,
        selectors=[
            "#netname",
            "input#netname",
            "input[type='email']",
            "input[name='email']",
            "input[name*='email']",
            "input[name='username']",
            "input[id*='email']",
            "input[name*='netname']",
            "input[id*='netname']",
        ],
        timeout=timeout,
        state="visible",
    )
    human_type(page, email_input, email, label="email")
    try:
        page.keyboard.press("Tab")
        human_pause(page, 80, 180)
    except Exception:
        pass

    password_input = wait_for_first(
        page,
        selectors=[
            "#password",
            "input#password",
            "input[type='password']",
            "input[name*='password']",
            "input[id*='password']",
        ],
        timeout=timeout,
        state="visible",
    )
    human_type(page, password_input, password, label="password")
    try:
        page.keyboard.press("Tab")
        human_pause(page, 80, 180)
    except Exception:
        pass

    submit_button = wait_for_first(
        page,
        selectors=[
            "button[type='submit']",
            "input[type='submit']",
            "button:has-text('Sign In')",
            "button:has-text('Log In')",
        ],
        timeout=timeout,
        state="visible",
    )
    human_click(page, submit_button, label="submit")

    page.wait_for_timeout(1000)
    handle_captcha(page, captcha_dir=captcha_dir, pause_on_captcha=pause_on_captcha, solver=captcha_solver)

    start_url = page.url
    try:
        page.wait_for_function(
            "start => window.location.href !== start",
            arg=start_url,
            timeout=int(timeout * 1000),
        )
    except PlaywrightTimeoutError:
        pass

    return page.url


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Login scraper for deckathon-concordia.com")
    parser.add_argument("--url", default=DEFAULT_LOGIN_URL, help="Login page URL")

    default_creds_path = os.path.join(os.path.dirname(__file__), "..", "creds.json")
    parser.add_argument(
        "--creds",
        default=os.path.abspath(default_creds_path),
        help="Path to creds.json containing email/username/netname and password",
    )

    email_env = os.getenv("DECK_EMAIL") or os.getenv("DECK_USERNAME")
    password_env = os.getenv("DECK_PASSWORD")
    parser.add_argument(
        "--email",
        default=email_env,
        help="Email/username (or set DECK_EMAIL/DECK_USERNAME env var)",
    )
    parser.add_argument(
        "--password",
        default=password_env,
        help="Password (or set DECK_PASSWORD env var)",
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Run Chromium in headless mode",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Run Chromium with a visible window",
    )
    parser.set_defaults(headless=True)
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for elements/navigation",
    )
    parser.add_argument(
        "--wait-after-load",
        type=float,
        default=2.0,
        help="Seconds to pause after page load before dumping/typing",
    )
    parser.add_argument(
        "--screenshot",
        metavar="PATH",
        help="Optional path to save a post-login screenshot for debugging",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Leave the browser open after logging in (useful for manual inspection)",
    )
    parser.add_argument(
        "--pause-after-login",
        dest="pause_after_login",
        action="store_true",
        help="Pause after login so you can navigate before dumping the DOM",
    )
    parser.add_argument(
        "--no-pause-after-login",
        dest="pause_after_login",
        action="store_false",
        help="Do not pause after login before dumping the DOM",
    )
    parser.set_defaults(pause_after_login=None)
    parser.add_argument(
        "--dump-candidates",
        action="store_true",
        help="After loading the page, print available input/button IDs/names/labels",
    )
    parser.add_argument(
        "--dump-only",
        action="store_true",
        help="Load page, dump candidates, and exit without attempting login",
    )
    parser.add_argument(
        "--dump-dom",
        default="post_login_dom.html",
        help="Path to save DOM after login (set to '' to disable)",
    )
    parser.add_argument(
        "--dump-dom-frames",
        action="store_true",
        help="Also dump iframe DOMs after login",
    )
    parser.add_argument(
        "--captcha-dir",
        default="captcha_logs",
        help="Directory to store captcha snapshots (set to '' to disable)",
    )
    parser.add_argument(
        "--pause-on-captcha",
        action="store_true",
        help="Pause and wait for manual captcha solve when detected",
    )
    parser.add_argument(
        "--captcha-solver",
        default="intelligent",
        choices=["manual", "ocr", "2captcha", "browser", "intelligent", "similarity", "sizebased", "openai"],
        help="Captcha solving method (default: manual)",
    )
    parser.add_argument(
        "--captcha-logo-path",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ref", "logo.png")),
        help="Path to the reference logo image for the 'intelligent' solver",
    )
    parser.add_argument(
        "--captcha-api-key",
        help="API key for 2captcha service (or set CAPTCHA_API_KEY env var)",
    )
    parser.add_argument(
        "--openai-api-key",
        help="API key for OpenAI service (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--openai-prompt",
        default="What is the text in the image?",
        help="Prompt to use for the OpenAI solver",
    )
    parser.add_argument(
        "--captcha-config",
        help="Path to JSON file with captcha solver configuration",
    )
    parser.add_argument(
        "--api-log",
        help="Directory to store JSON snapshots of key API responses",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if args.pause_after_login is None:
        args.pause_after_login = not args.headless
    if not args.dump_dom:
        args.dump_dom = str(Path(__file__).resolve().parent.parent / "post_login_dom.html")

    file_email, file_password = load_creds(args.creds)
    if not args.email and file_email:
        args.email = file_email
    if not args.password and file_password:
        args.password = file_password

    missing = []
    if not args.email:
        missing.append("email/DECK_EMAIL/creds file")
    if not args.password:
        missing.append("password/DECK_PASSWORD/creds file")
    if missing:
        print(f"Missing required credential(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    captcha_dir = args.captcha_dir or None
    
    # Load captcha solver configuration
    solver_config = {}
    if hasattr(args, "captcha_config") and args.captcha_config:
        try:
            with open(args.captcha_config, "r", encoding="utf-8") as f:
                solver_config = json.load(f)
        except Exception as e:
            print(f"Failed to load captcha config: {e}", file=sys.stderr)
    
    # Build solver config from CLI args
    if hasattr(args, "captcha_solver"):
        solver_config["method"] = args.captcha_solver
    
    if hasattr(args, "captcha_api_key") and args.captcha_api_key:
        solver_config["api_key"] = args.captcha_api_key

    if hasattr(args, "openai_api_key") and args.openai_api_key:
        solver_config["api_key"] = args.openai_api_key
    elif os.getenv("OPENAI_API_KEY"):
        solver_config["api_key"] = os.getenv("OPENAI_API_KEY")

    if hasattr(args, "openai_prompt") and args.openai_prompt:
        solver_config["openai_prompt"] = args.openai_prompt
    
    if hasattr(args, "captcha_logo_path") and args.captcha_logo_path:
        solver_config["logo_path"] = args.captcha_logo_path
    
    # Initialize captcha solver
    # Default to 'browser' solver which handles image selection captchas
    if not solver_config:
        solver_config["method"] = "browser"
    
    captcha_solver = get_solver(solver_config)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        api_state: dict = {}
        attach_api_logger(page, api_state, log_dir=args.api_log)

        try:
            preloaded = False
            if args.dump_candidates or args.dump_only:
                page.goto(args.url, wait_until="domcontentloaded")
                if args.wait_after_load > 0:
                    page.wait_for_timeout(int(args.wait_after_load * 1000))
                dump_candidates(page)
                preloaded = True
                if args.dump_only:
                    return 0

            final_url = perform_login(
                page,
                url=args.url,
                email=args.email,
                password=args.password,
                timeout=args.timeout,
                already_loaded=preloaded,
                wait_after_load=args.wait_after_load,
                captcha_dir=captcha_dir,
                pause_on_captcha=args.pause_on_captcha if hasattr(args, "pause_on_captcha") else False,
                captcha_solver=captcha_solver,
            )
            print(f"Finished navigation at: {final_url}")

            if handle_two_factor(page, timeout=args.timeout):
                print("MFA step completed.")
                page.wait_for_timeout(1000)

            if args.screenshot:
                page.screenshot(path=args.screenshot, full_page=True)
                print(f"Saved screenshot to {args.screenshot}")

            user_info = fetch_user_info(
                context.request,
                api_state.get("api_origin") or get_origin(page.url),
                api_state.get("auth_token"),
                timeout=args.timeout,
            )
            if user_info:
                api_state["user_info"] = user_info
                balance = extract_balance(user_info)
                api_state["balance"] = balance
                status = user_info.get("status")
                classes = user_info.get("classes") or []
                print(f"User status: {status}, classes: {len(classes)}")
                if balance is not None:
                    print(f"User balance: {balance:.2f}")

            dump_dom_path = args.dump_dom or None
            dump_dom_interactive(
                page,
                dump_dom_path,
                "post_login",
                include_frames=args.dump_dom_frames,
                prompt=False,
            )

            balance = api_state.get("balance")
            if balance is not None and balance <= 0:
                if navigate_to_dropout(page, timeout=args.timeout):
                    dump_dom_interactive(
                        page,
                        dump_dom_path,
                        "dropout",
                        include_frames=args.dump_dom_frames,
                        prompt=False,
                    )
            else:
                if handle_payment_flow(
                    page,
                    timeout=args.timeout,
                    dump_dom_path=dump_dom_path,
                    dump_dom_frames=args.dump_dom_frames,
                ):
                    print("Payment flow completed.")
                    dump_dom_interactive(
                        page,
                        dump_dom_path,
                        "payment",
                        include_frames=args.dump_dom_frames,
                        prompt=False,
                    )

            if args.keep_open:
                print("Browser left open (--keep-open). Close it manually when done.")
                input("Press Enter to close the browser...")
                return 0
        finally:
            browser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
