#!/usr/bin/env python3
"""
Captcha solver module supporting:
- Image-based CAPTCHAs (OCR via Tesseract/pytesseract)
- reCAPTCHA v2/v3 (via 2captcha or manual solving)
- hCaptcha (via 2captcha or manual solving)
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

from captcha_database import CaptchaDatabase

try:
    import pytesseract
    from PIL import Image, ImageChops, ImageStat, ImageFilter
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from captcha_image_analysis import GridAnalyzer, LogoDetector
    HAS_IMAGE_ANALYSIS = True
except ImportError:
    HAS_IMAGE_ANALYSIS = False

try:
    import cv2
    import numpy as np
    HAS_CV = True
except ImportError:
    HAS_CV = False


def human_pause(page, min_ms: int = 80, max_ms: int = 220) -> None:
    """Pause for a short, human-like delay."""
    delay = random.uniform(min_ms, max_ms)
    page.wait_for_timeout(int(delay))


def human_mouse_click_at(page, x: float, y: float, label: str | None = None) -> None:
    """Move mouse to a point with jitter, then click."""
    if label:
        print(f"Clicking {label} (humanized)")
    jitter_x = random.uniform(-3, 3)
    jitter_y = random.uniform(-3, 3)
    target_x = x + jitter_x
    target_y = y + jitter_y
    steps = random.randint(6, 14)
    page.mouse.move(target_x, target_y, steps=steps)
    human_pause(page)
    page.mouse.click(target_x, target_y)
    human_pause(page)


def human_click(page, element, label: str | None = None) -> None:
    """Click an element with a slight mouse move and delay."""
    if label:
        print(f"Clicking {label} (humanized)")
    try:
        element.scroll_into_view_if_needed()
    except Exception:
        pass
    try:
        box = element.bounding_box()
    except Exception:
        box = None
    if box:
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2
        human_mouse_click_at(page, center_x, center_y)
        return
    element.click()
    human_pause(page)


class CaptchaSolver:
    """Base captcha solver interface."""
    
    def solve(self, captcha_data: dict) -> Optional[str]:
        """
        Solve a captcha and return the solution token/answer.
        
        Args:
            captcha_data: Dict containing captcha info (type, image_path, etc)
        
        Returns:
            Solution token/string or None if unable to solve
        """
        raise NotImplementedError


class TemplateMatchingSolver(CaptchaSolver):
    """Solver that uses template matching to find a logo."""

    def __init__(self, logo_path: str, threshold: float = 0.8):
        if not HAS_CV:
            raise ImportError("opencv-python and numpy are required for template matching")
        self.logo_path = logo_path
        self.threshold = threshold
        self.logo_img = cv2.imread(logo_path, 0)
        if self.logo_img is None:
            raise ValueError(f"Could not read logo image at: {logo_path}")
        self.logo_w, self.logo_h = self.logo_img.shape[::-1]

    def solve(self, captcha_data: dict) -> Optional[str]:
        """Marker return only, logic is in solve_on_page."""
        return "REQUIRES_BROWSER"

    def solve_on_page(self, page, frame=None, **kwargs) -> bool:
        """Find and click on images matching the logo."""
        print("🤖 Starting template matching captcha solving...")
        search_frame = frame or page.main_frame

        try:
            # Get captcha screenshot
            captcha_element = search_frame.query_selector("div:has(img[alt^='Captcha image'])")
            if not captcha_element:
                captcha_element = search_frame.query_selector("div[class*='grid']")
            if not captcha_element:
                captcha_element = search_frame.query_selector("div:has(div:has-text('Select all'))")
            if not captcha_element:
                return False

            temp_path = "/tmp/captcha_template_matching.png"
            captcha_element.screenshot(path=temp_path)
            print(f"Saved template matching screenshot to {temp_path}")
            
            captcha_img = cv2.imread(temp_path)
            captcha_gray = cv2.cvtColor(captcha_img, cv2.COLOR_BGR2GRAY)

            # Perform template matching
            print(f"Running template matching with threshold {self.threshold}")
            res = cv2.matchTemplate(captcha_gray, self.logo_img, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= self.threshold)

            points = list(zip(*loc[::-1]))
            if not points:
                print("No matches found.")
                return False
            print(f"Found {len(points)} raw match points.")

            # Cluster points to identify individual logos
            clusters = []
            for pt in points:
                found_cluster = False
                for cluster in clusters:
                    if any(np.linalg.norm(np.array(pt) - np.array(c_pt)) < 20 for c_pt in cluster):
                        cluster.append(pt)
                        found_cluster = True
                        break
                if not found_cluster:
                    clusters.append([pt])
            
            centers = [tuple(np.mean(cluster, axis=0, dtype=int)) for cluster in clusters]
            print(f"Found {len(centers)} potential logos.")

            # Get bounding box of the captcha element to offset click coordinates
            captcha_box = captcha_element.bounding_box()
            if not captcha_box:
                print("Could not get bounding box of captcha element.")
                return False

            for center in centers:
                # Adjust click position relative to the captcha element's top-left corner
                click_x = captcha_box['x'] + center[0] + self.logo_w / 2
                click_y = captcha_box['y'] + center[1] + self.logo_h / 2
                print(f"Clicking at ({click_x}, {click_y})")
                human_mouse_click_at(page, click_x, click_y)

            print(f"✅ Clicked {len(centers)} potential logos.")
            page.wait_for_timeout(500)
            return True

        except Exception as e:
            print(f"Template matching solving failed: {e}")
            return False


class OCRCaptchaSolver(CaptchaSolver):
    """Solve image-based CAPTCHAs using OCR or image analysis."""
    
    def __init__(self, pytesseract_path: Optional[str] = None):
        if not HAS_OCR:
            raise ImportError("pytesseract and Pillow required for OCR solving")
        
        if pytesseract_path:
            pytesseract.pytesseract.pytesseract_path = pytesseract_path
    
    def _analyze_grid_captcha(self, image_path: str) -> Optional[list]:
        """Analyze multi-select grid captcha and return indices of likely matches."""
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # Detect if this is a grid-based captcha (multiple clickable areas)
            # Typical grid captchas are square or near-square with 3x3 or 2x3 layout
            grid_size = self._detect_grid_size(img)
            if grid_size:
                print(f"Detected {grid_size[0]}x{grid_size[1]} grid captcha")
                # Return grid info for UI-based solving
                return {"type": "grid", "rows": grid_size[0], "cols": grid_size[1], "image": image_path}
        except Exception as e:
            print(f"Grid analysis failed: {e}")
        
        return None
    
    def _detect_grid_size(self, img: 'Image.Image') -> Optional[tuple]:
        """Detect grid layout from image using edge detection."""
        try:
            import numpy as np
            
            # Convert to grayscale
            if img.mode != 'L':
                gray = img.convert('L')
            else:
                gray = img
            
            # Convert to numpy array
            arr = np.array(gray)
            
            # Simple edge detection to find grid lines
            # Look for horizontal and vertical lines
            horizontal_edges = np.abs(np.diff(arr, axis=0)).sum(axis=1)
            vertical_edges = np.abs(np.diff(arr, axis=1)).sum(axis=0)
            
            # Find peaks (grid lines)
            h_threshold = np.mean(horizontal_edges) * 2
            v_threshold = np.mean(vertical_edges) * 2
            
            h_lines = np.where(horizontal_edges > h_threshold)[0]
            v_lines = np.where(vertical_edges > v_threshold)[0]
            
            # Count grid cells
            if len(h_lines) > 0 and len(v_lines) > 0:
                rows = len(h_lines) // 2 + 1
                cols = len(v_lines) // 2 + 1
                
                # Sanity check: grid should be reasonable size
                if 1 <= rows <= 5 and 1 <= cols <= 5:
                    return (rows, cols)
        except Exception:
            pass
        
        return None
    
    def solve(self, captcha_data: dict) -> Optional[str]:
        """Solve image captcha using OCR or grid analysis."""
        image_path = captcha_data.get("image_path")
        if not image_path:
            return None
        
        try:
            img = Image.open(image_path)
            
            # Try OCR first for text-based captchas
            text = pytesseract.image_to_string(img)
            text = ''.join(text.split())
            
            if text and len(text) > 2:
                print(f"OCR extracted: {text}")
                return text
            
            # If no text found, might be image selection captcha
            print("No text detected. Analyzing as image selection captcha...")
            grid_info = self._analyze_grid_captcha(image_path)
            
            if grid_info:
                # For grid captchas, we need UI interaction
                # Return a marker indicating this needs browser interaction
                return "GRID_CAPTCHA"
                
        except Exception as e:
            print(f"OCR/Analysis failed: {e}")
        
        return None


class TwoCaptchaSolver(CaptchaSolver):
    """Solve captchas using 2captcha API service."""
    
    BASE_URL = "http://2captcha.com"
    
    def __init__(self, api_key: str):
        if not HAS_REQUESTS:
            raise ImportError("requests library required for 2captcha")
        
        self.api_key = api_key
    
    def _upload_image(self, image_path: str) -> Optional[str]:
        """Upload image to 2captcha and get captcha ID."""
        try:
            with open(image_path, "rb") as f:
                files = {"captchafile": f}
                data = {
                    "key": self.api_key,
                    "method": "post",
                }
                resp = requests.post(f"{self.BASE_URL}/api/upload", files=files, data=data, timeout=10)
                resp.raise_for_status()
                
                # Response format: OK|captcha_id
                if resp.text.startswith("OK"):
                    return resp.text.split("|")[1]
        except Exception as e:
            print(f"Failed to upload captcha to 2captcha: {e}")
        
        return None
    
    def _get_result(self, captcha_id: str, timeout: float = 60) -> Optional[str]:
        """Poll 2captcha for solution."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/api/res",
                    params={"key": self.api_key, "action": "get", "id": captcha_id},
                    timeout=10,
                )
                resp.raise_for_status()
                
                # Response format: OK|solution or CAPCHA_NOT_READY
                if resp.text.startswith("OK"):
                    return resp.text.split("|")[1]
                elif resp.text == "CAPCHA_NOT_READY":
                    time.sleep(2)
                    continue
                else:
                    print(f"2captcha error: {resp.text}")
                    return None
            except Exception as e:
                print(f"Failed to get 2captcha result: {e}")
                return None
        
        print("2captcha solving timeout")
        return None
    
    def solve_image(self, image_path: str, timeout: float = 60) -> Optional[str]:
        """Solve image captcha."""
        captcha_id = self._upload_image(image_path)
        if not captcha_id:
            return None
        
        return self._get_result(captcha_id, timeout)
    
    def solve_recaptcha_v2(self, sitekey: str, pageurl: str, timeout: float = 60) -> Optional[str]:
        """Solve reCAPTCHA v2."""
        try:
            # Submit reCAPTCHA
            resp = requests.post(
                f"{self.BASE_URL}/api/captcha",
                data={
                    "key": self.api_key,
                    "method": "userrecaptcha",
                    "googlekey": sitekey,
                    "pageurl": pageurl,
                },
                timeout=10,
            )
            resp.raise_for_status()
            
            if not resp.text.startswith("OK"):
                print(f"2captcha error: {resp.text}")
                return None
            
            captcha_id = resp.text.split("|")[1]
            return self._get_result(captcha_id, timeout)
        except Exception as e:
            print(f"Failed to solve reCAPTCHA: {e}")
        
        return None
    
    def solve_hcaptcha(self, sitekey: str, pageurl: str, timeout: float = 60) -> Optional[str]:
        """Solve hCaptcha."""
        try:
            # Submit hCaptcha
            resp = requests.post(
                f"{self.BASE_URL}/api/captcha",
                data={
                    "key": self.api_key,
                    "method": "hcaptcha",
                    "sitekey": sitekey,
                    "pageurl": pageurl,
                },
                timeout=10,
            )
            resp.raise_for_status()
            
            if not resp.text.startswith("OK"):
                print(f"2captcha error: {resp.text}")
                return None
            
            captcha_id = resp.text.split("|")[1]
            return self._get_result(captcha_id, timeout)
        except Exception as e:
            print(f"Failed to solve hCaptcha: {e}")
        
        return None
    
    def solve(self, captcha_data: dict) -> Optional[str]:
        """Solve captcha based on type."""
        captcha_type = captcha_data.get("type", "image").lower()
        
        if captcha_type == "recaptcha_v2":
            return self.solve_recaptcha_v2(
                captcha_data.get("sitekey", ""),
                captcha_data.get("pageurl", ""),
            )
        elif captcha_type == "hcaptcha":
            return self.solve_hcaptcha(
                captcha_data.get("sitekey", ""),
                captcha_data.get("pageurl", ""),
            )
        else:  # image-based
            return self.solve_image(captcha_data.get("image_path", ""))


class ImageSelectionSolver(CaptchaSolver):
    """Solve image selection captchas (click matching images)."""
    
    def __init__(self, keywords: Optional[list] = None):
        """
        Initialize with keywords to search for in image selection prompts.
        
        Args:
            keywords: List of keywords that might appear in the captcha prompt
                     e.g., ["logo", "traffic light", "crosswalk"]
        """
        self.keywords = keywords or []
    
    def solve(self, captcha_data: dict) -> Optional[str]:
        """
        Solve image selection captcha.
        
        For Playwright integration, this returns information needed for
        browser-based solving rather than a direct solution string.
        """
        # This solver needs to interact with the page via Playwright
        # Return a marker to indicate browser automation needed
        return "BROWSER_INTERACTION_REQUIRED"
    
    def solve_on_page(self, page, frame=None) -> bool:
        """
        Detect image selection captcha prompt and attempt to solve via browser.
        
        Args:
            page: Playwright page object
            frame: Optional frame if captcha is in iframe
            
        Returns:
            True if solved, False otherwise
        """
        search_frame = frame or page.main_frame
        
        try:
            # Find captcha prompt text
            prompt_text = None
            for selector in ["div:has-text('Select all')", "h2:has-text('Select')", "p:has-text('Select')"]:
                try:
                    el = search_frame.query_selector(selector)
                    if el:
                        prompt_text = el.text_content()
                        break
                except Exception:
                    pass
            
            if not prompt_text:
                return False
            
            print(f"Image selection prompt: {prompt_text}")
            
            # Extract what we're looking for from the prompt
            # e.g., "Select all logos" -> looking for "logo"
            search_term = self._extract_search_term(prompt_text)
            
            if not search_term:
                print("Could not extract search term from prompt")
                return False
            
            print(f"Looking for: {search_term}")
            
            # Find all selectable images/divs
            images = search_frame.query_selector_all("img, div[role='button']")
            
            if not images:
                print("No selectable images found")
                return False
            
            print(f"Found {len(images)} candidate images")
            
            # For now, log the number found
            # Full solving would require image recognition/AI
            print(f"Image selection captcha detected with {len(images)} options")
            print(f"Manual selection required for: {search_term}")
            
            return False  # Requires manual interaction
            
        except Exception as e:
            print(f"Image selection solving failed: {e}")
            return False
    
    def _extract_search_term(self, prompt_text: str) -> Optional[str]:
        """Extract what to search for from prompt text."""
        text_lower = prompt_text.lower()
        
        # Common patterns
        patterns = {
            "logo": ["logo", "company", "brand"],
            "traffic light": ["traffic", "light", "signal"],
            "crosswalk": ["crosswalk", "cross walk", "zebra"],
            "car": ["car", "vehicle", "automobile"],
            "bus": ["bus", "transport"],
            "motorcycle": ["motorcycle", "bike"],
        }
        
        for term, keywords in patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return term
        
        # Fallback: try to extract noun after "Select all"
        if "select all" in text_lower:
            parts = text_lower.split("select all")
            if len(parts) > 1:
                after = parts[1].strip()
                # Remove common endings
                for suffix in [".", ",", " that", " which"]:
                    if suffix in after:
                        after = after.split(suffix)[0]
                return after.split()[0] if after else None
        
        return None


class BrowserAutomatedSolver(CaptchaSolver):
    """Solver that uses Playwright for browser automation and automatic image grid solving."""
    
    def __init__(self, logo_path: Optional[str] = None):
        self.image_selector = ImageSelectionSolver()
        self.db = CaptchaDatabase()
        self.analyzer = GridAnalyzer() if HAS_IMAGE_ANALYSIS else None
        self.detector = LogoDetector() if HAS_IMAGE_ANALYSIS else None
        self.reference_logo = None
        if logo_path and HAS_OCR:
            try:
                self.reference_logo = Image.open(logo_path).convert("RGB")
                print(f"Loaded reference logo for grid matching: {logo_path}")
            except Exception as e:
                print(f"Failed to load reference logo ({logo_path}): {e}")
    
    def solve(self, captcha_data: dict) -> Optional[str]:
        """Marker return only."""
        return "REQUIRES_BROWSER"

    def _edge_rms(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute RMS difference between edge maps of two images."""
        img1 = img1.convert("L").resize((64, 64)).filter(ImageFilter.FIND_EDGES)
        img2 = img2.convert("L").resize((64, 64)).filter(ImageFilter.FIND_EDGES)
        diff = ImageChops.difference(img1, img2)
        stat = ImageStat.Stat(diff)
        return stat.rms[0]

    def _calculate_dhash(self, img: Image.Image, hash_size: int = 8) -> int:
        """Calculate the difference hash (dHash) of an image."""
        img = img.convert("L").resize((hash_size + 1, hash_size))
        pixels = list(img.getdata())
        difference = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = img.getpixel((col, row))
                pixel_right = img.getpixel((col + 1, row))
                difference.append(pixel_left > pixel_right)
        decimal_value = 0
        for i, bit in enumerate(difference):
            if bit:
                decimal_value += 2**i
        return decimal_value

    def _hamming_distance(self, a: int, b: int) -> int:
        """Return Hamming distance between two integer hashes."""
        return (a ^ b).bit_count()

    def _log_dom_tiles(self, elements: list) -> list[dict]:
        """Log DOM attributes for tile elements and return metadata."""
        meta_list = []
        for idx, el in enumerate(elements):
            try:
                meta = el.evaluate(
                    """el => {
                        const attrs = {};
                        for (const attr of el.attributes || []) {
                            attrs[attr.name] = attr.value;
                        }
                        return {
                            tag: el.tagName ? el.tagName.toLowerCase() : 'unknown',
                            text: (el.innerText || '').trim(),
                            attrs,
                        };
                    }"""
                )
            except Exception:
                meta = {"tag": "unknown", "text": "", "attrs": {}}
            attrs = meta.get("attrs", {}) or {}
            preview = {
                k: v
                for k, v in attrs.items()
                if k.startswith("data-") or k.startswith("aria-") or k in ("alt", "title", "src")
            }
            print(f"Tile {idx}: tag={meta.get('tag')} text='{meta.get('text')}' attrs={preview}")
            meta_list.append(meta)
        return meta_list

    def _match_cells_by_dom_metadata(self, elements: list) -> list[int]:
        """Find duplicate tiles by matching DOM attribute values."""
        meta_list = self._log_dom_tiles(elements)
        attr_map: dict[tuple[str, str], list[int]] = {}
        for idx, meta in enumerate(meta_list):
            attrs = meta.get("attrs", {}) or {}
            for key, value in attrs.items():
                key_lower = key.lower()
                if not value:
                    continue
                if key_lower in {"class", "style", "role", "id", "name"}:
                    continue
                if not (
                    key_lower.startswith("data-")
                    or key_lower.startswith("aria-")
                    or key_lower in {"alt", "title", "src"}
                ):
                    continue
                val = value.strip()
                if len(val) < 2:
                    continue
                attr_map.setdefault((key_lower, val), []).append(idx)

        groups = [indices for indices in attr_map.values() if len(indices) >= 2]
        if not groups:
            print("No duplicate DOM attributes found across tiles.")
            return []
        matched = sorted({i for group in groups for i in group})
        for (key, val), indices in attr_map.items():
            if len(indices) >= 2:
                print(f"DOM duplicate: {key}='{val[:80]}' -> {indices}")
        print(f"DOM-matched tiles: {matched}")
        return matched

    def _select_tile_elements(self, search_frame, captcha_element=None) -> list:
        """Return tile elements scoped to the captcha grid."""
        elements = []
        role_buttons = []
        if captcha_element:
            try:
                elements = captcha_element.query_selector_all("img, div[role='button'], input[type='radio']")
                role_buttons = captcha_element.query_selector_all("div[role='button']")
            except Exception:
                elements = []
                role_buttons = []
        if not elements:
            elements = search_frame.query_selector_all("img, div[role='button'], input[type='radio']")

        if len(role_buttons) >= 9:
            return role_buttons

        alt_filtered = []
        for el in elements:
            try:
                alt = el.get_attribute("alt")
            except Exception:
                alt = None
            if alt and "captcha image" in alt.lower():
                alt_filtered.append(el)

        if alt_filtered:
            if len(alt_filtered) >= 9:
                return alt_filtered
            merged = alt_filtered + [el for el in elements if el not in alt_filtered]
            return merged
        return elements

    def _hash_tile_element(self, element) -> Optional[dict]:
        """Capture tile screenshot and return hash info."""
        try:
            img_bytes = element.screenshot()
        except Exception as e:
            print(f"Tile screenshot failed: {e}")
            return None
        try:
            img = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            print(f"Tile image decode failed: {e}")
            return None
        stddev, edge_mean, edge_img = self._cell_features(img)
        dhash = self._calculate_dhash(edge_img)
        return {"stddev": stddev, "edge_mean": edge_mean, "dhash": dhash}

    def _match_tiles_by_element_hashes(self, elements: list, hamming_threshold: int = 6) -> list[int]:
        """Compare each tile element by hash and return duplicated indices."""
        entries = []
        entry_by_idx = {}
        for idx, el in enumerate(elements):
            info = self._hash_tile_element(el)
            if not info:
                continue
            entries.append({"idx": idx, **info})
            entry_by_idx[idx] = info
            print(
                f"Tile {idx} (screenshot): stddev={info['stddev']:.2f}, "
                f"edge_mean={info['edge_mean']:.2f}, dhash={info['dhash']}"
            )

        if not entries:
            print("No tile screenshots captured for matching.")
            return []

        groups: list[list[int]] = []
        used = set()
        for i, entry in enumerate(entries):
            if entry["idx"] in used:
                continue
            group = [entry["idx"]]
            used.add(entry["idx"])
            for other in entries[i + 1 :]:
                if other["idx"] in used:
                    continue
                dist = self._hamming_distance(entry["dhash"], other["dhash"])
                if dist <= hamming_threshold:
                    group.append(other["idx"])
                    used.add(other["idx"])
            groups.append(group)

        duplicate_groups = [g for g in groups if len(g) >= 2]
        if not duplicate_groups:
            print(f"No duplicate groups found (groups={groups}).")
            return []
        ranked = []
        for group in duplicate_groups:
            mean_edge = sum(entry_by_idx[i]["edge_mean"] for i in group) / len(group)
            mean_stddev = sum(entry_by_idx[i]["stddev"] for i in group) / len(group)
            ranked.append((mean_edge, mean_stddev, len(group), sorted(group)))
        ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        for mean_edge, mean_stddev, size, group in ranked:
            print(
                f"Group {group}: mean_edge={mean_edge:.2f}, "
                f"mean_stddev={mean_stddev:.2f}, size={size}"
            )
        best_group = ranked[0][3]
        print(f"Selected duplicate group via tile screenshots: {best_group}")
        return best_group

    def _log_dom_tiles(self, elements: list) -> list[dict]:
        """Log DOM attributes for tile elements and return metadata."""
        meta_list = []
        for idx, el in enumerate(elements):
            try:
                meta = el.evaluate(
                    """el => {
                        const attrs = {};
                        for (const attr of el.attributes || []) {
                            attrs[attr.name] = attr.value;
                        }
                        return {
                            tag: el.tagName ? el.tagName.toLowerCase() : 'unknown',
                            text: (el.innerText || '').trim(),
                            attrs,
                        };
                    }"""
                )
            except Exception:
                meta = {"tag": "unknown", "text": "", "attrs": {}}
            attrs = meta.get("attrs", {}) or {}
            preview = {
                k: v
                for k, v in attrs.items()
                if k.startswith("data-") or k.startswith("aria-") or k in ("alt", "title", "src")
            }
            print(f"Tile {idx}: tag={meta.get('tag')} text='{meta.get('text')}' attrs={preview}")
            meta_list.append(meta)
        return meta_list

    def _match_cells_by_dom_metadata(self, elements: list) -> list[int]:
        """Find duplicate tiles by matching DOM attribute values."""
        meta_list = self._log_dom_tiles(elements)
        attr_map: dict[tuple[str, str], list[int]] = {}
        for idx, meta in enumerate(meta_list):
            attrs = meta.get("attrs", {}) or {}
            for key, value in attrs.items():
                key_lower = key.lower()
                if not value:
                    continue
                if key_lower in {"class", "style", "role", "id", "name"}:
                    continue
                if not (
                    key_lower.startswith("data-")
                    or key_lower.startswith("aria-")
                    or key_lower in {"alt", "title", "src"}
                ):
                    continue
                val = value.strip()
                if len(val) < 2:
                    continue
                attr_map.setdefault((key_lower, val), []).append(idx)

        groups = [indices for indices in attr_map.values() if len(indices) >= 2]
        if not groups:
            print("No duplicate DOM attributes found across tiles.")
            return []
        matched = sorted({i for group in groups for i in group})
        for (key, val), indices in attr_map.items():
            if len(indices) >= 2:
                print(f"DOM duplicate: {key}='{val[:80]}' -> {indices}")
        print(f"DOM-matched tiles: {matched}")
        return matched

    def _cell_features(self, cell_img: Image.Image) -> tuple[float, float, Image.Image]:
        """Return grayscale stddev, edge mean, and edge image."""
        gray = cell_img.convert("L")
        stat = ImageStat.Stat(gray)
        stddev = stat.stddev[0]
        edge_img = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edge_img)
        edge_mean = edge_stat.mean[0]
        return stddev, edge_mean, edge_img

    def _is_uniform_cell(self, cell_img: Image.Image, stddev_threshold: float = 6.0, edge_threshold: float = 2.0) -> bool:
        """Heuristic to treat solid-color tiles as uniform."""
        stddev, edge_mean, _ = self._cell_features(cell_img)
        return stddev < stddev_threshold and edge_mean < edge_threshold

    def _match_cells_to_reference(self, image_path: str, cells: list) -> list:
        """Compare each cell to the reference logo and return matching indices."""
        if not self.reference_logo:
            return []
        scores = []
        for idx, bbox in enumerate(cells):
            cell_img = self.analyzer.extract_cell_image(image_path, bbox)
            if not cell_img:
                continue
            rms = self._edge_rms(cell_img, self.reference_logo)
            scores.append((rms, idx))
        if not scores:
            return []
        scores.sort(key=lambda x: x[0])
        min_rms = scores[0][0]
        threshold = min_rms * 1.25
        matches = [idx for rms, idx in scores if rms <= threshold]
        print(f"Reference match RMS min={min_rms:.2f}, threshold={threshold:.2f}, matches={matches}")
        return matches

    def _match_cells_by_similarity(self, image_path: str, cells: list, hamming_threshold: int = 6) -> list:
        """Compare each cell to every other and return all duplicated indices."""
        entries = []
        for idx, bbox in enumerate(cells):
            cell_img = self.analyzer.extract_cell_image(image_path, bbox)
            if not cell_img:
                continue
            stddev, edge_mean, edge_img = self._cell_features(cell_img)
            dhash = self._calculate_dhash(edge_img)
            entries.append(
                {
                    "idx": idx,
                    "stddev": stddev,
                    "edge_mean": edge_mean,
                    "dhash": dhash,
                }
            )
            print(f"Cell {idx}: stddev={stddev:.2f}, edge_mean={edge_mean:.2f}, dhash={dhash}")

        if not entries:
            print("No cells extracted for duplicate matching.")
            return []

        edge_values = sorted(entry["edge_mean"] for entry in entries)
        edge_threshold = 2.0
        best_gap = 0.0
        for left, right in zip(edge_values, edge_values[1:]):
            gap = right - left
            if gap > best_gap:
                best_gap = gap
                edge_threshold = (left + right) / 2
        print(f"Edge threshold for candidates: {edge_threshold:.2f} (gap={best_gap:.2f})")

        candidates = [entry for entry in entries if entry["edge_mean"] >= edge_threshold]
        print(f"Candidate tiles for matching: {[c['idx'] for c in candidates]}")
        if len(candidates) < 2:
            print("Not enough candidate tiles for duplicate matching.")
            return []

        # Build similarity groups using Hamming distance
        groups: list[list[int]] = []
        used = set()
        for i, entry in enumerate(candidates):
            if entry["idx"] in used:
                continue
            group = [entry["idx"]]
            used.add(entry["idx"])
            for other in candidates[i + 1 :]:
                if other["idx"] in used:
                    continue
                dist = self._hamming_distance(entry["dhash"], other["dhash"])
                if dist <= hamming_threshold:
                    group.append(other["idx"])
                    used.add(other["idx"])
            groups.append(group)

        duplicate_groups = [g for g in groups if len(g) >= 2]
        if not duplicate_groups:
            print(f"No duplicate groups found (groups={groups}).")
            return []
        matched = sorted({idx for group in duplicate_groups for idx in group})
        print(f"Duplicate groups found: {duplicate_groups} -> matches={matched}")
        return matched
    
    def _auto_solve_grid(self, page, frame=None) -> bool:
        """
        Automatically solve image grid captcha by clicking on non-color cells.
        
        Returns True if successfully solved.
        """
        if not self.analyzer:
            print("Image analyzer not available, cannot auto-solve grid.")
            return False
        
        print("🤖 Starting automatic grid solving...")
        try:
            # Find captcha image on page
            search_frame = frame or page.main_frame
            
            # Get captcha screenshot
            captcha_element = search_frame.query_selector("div:has(img[alt^='Captcha image'])")
            if not captcha_element:
                captcha_element = search_frame.query_selector("div[class*='grid']")
            if not captcha_element:
                captcha_element = search_frame.query_selector("div:has(div:has-text('Select all'))")
            if not captcha_element:
                captcha_element = search_frame.query_selector("img[alt*='captcha'], div[role='img']")
            if not captcha_element:
                print("Could not find captcha element on page.")
                return False
            print("Captcha element located for grid analysis.")
            
            # Save temp screenshot
            temp_path = "/tmp/captcha_grid.png"
            captcha_element.screenshot(path=temp_path)
            print(f"Saved temporary captcha image to {temp_path}")
            
            # Analyze grid
            print("Analyzing captcha grid...")
            analysis = self.analyzer.analyze_grid(temp_path, save_analysis=False)
            if not analysis:
                print("Grid analysis failed.")
                return False
            
            image_cells = analysis.get('image_cells', [])
            color_cells = analysis.get('color_cells', [])
            grid_rows, grid_cols = analysis.get("grid_size", (0, 0))
            print(
                f"Grid analysis complete: {grid_rows}x{grid_cols}, "
                f"{len(image_cells)} image cells, {len(color_cells)} color cells."
            )

            tile_elements = self._select_tile_elements(search_frame, captcha_element)
            print(f"Found {len(tile_elements)} tile elements in captcha scope.")

            if not image_cells:
                print("No image cells detected. Comparing tiles for duplicates...")
                detection_result = self.analyzer.detect_grid_cells(temp_path)
                if not detection_result:
                    print("Could not detect grid cells for matching.")
                    return False
                cells, _, _ = detection_result
                dom_matches = self._match_cells_by_dom_metadata(tile_elements)
                image_cells = dom_matches or self._match_tiles_by_element_hashes(tile_elements)
                if not image_cells:
                    image_cells = self._match_cells_by_similarity(temp_path, cells)
                if not image_cells and self.reference_logo:
                    image_cells = self._match_cells_to_reference(temp_path, cells)
                if not image_cells:
                    print("No matching cells found via duplicate/reference comparison.")
                    return False

            clickables = tile_elements or search_frame.query_selector_all(
                "div[role='button'], img, input[type='radio']"
            )
            print(f"Found {len(clickables)} clickable elements in the grid.")

            # Click on image cells (not color cells)
            for idx in image_cells:
                if idx < len(clickables):
                    element = clickables[idx]
                    print(f"Clicking image cell {idx}...")
                    human_click(page, element, label=f"grid cell {idx}")
                else:
                    print(f"Warning: Image cell index {idx} is out of bounds for clickable elements.")

            # Click verify button if present
            if self._click_verify_button(page, search_frame, captcha_element):
                print("Clicked captcha verify button.")
            else:
                print("Captcha verify button not found.")

            # Wait a bit for submission
            page.wait_for_timeout(800)
            print("✅ Automatic grid solving process completed.")
            return True
            
        except Exception as e:
            print(f"Auto-solve grid failed: {e}")
            return False

    def _click_verify_button(self, page, search_frame, captcha_element=None) -> bool:
        """Click the captcha verify button if it exists."""
        scopes = [captcha_element, search_frame]
        selectors = [
            "button:has-text('Verify')",
            "button:has(span:has-text('Verify'))",
            "button[aria-label*='Verify']",
        ]
        for scope in scopes:
            if not scope:
                continue
            for selector in selectors:
                try:
                    el = scope.query_selector(selector)
                except Exception:
                    el = None
                if el:
                    human_click(page, el, label="captcha verify")
                    return True
        return False
    
    def solve_on_page(self, page, frame=None, timeout: float = 300) -> bool:
        """
        Attempt to solve captcha on the page using browser interaction.
        Tries automatic solving first, falls back to manual prompting.
        
        Args:
            page: Playwright page
            frame: Optional iframe
            timeout: Timeout for user manual solving (seconds)
            
        Returns:
            True if captcha likely solved, False otherwise
        """
        search_frame = frame or page.main_frame
        
        # Try image selection solver
        if self.image_selector.solve_on_page(page, search_frame):
            return True
        
        # Try automatic grid solving
        print("Attempting automatic grid solving...")
        if self._auto_solve_grid(page, search_frame):
            print("✓ Auto-solved grid captcha")
            return True
        
        # Check if it's a known image selection captcha
        try:
            prompt = search_frame.query_selector("div:has-text('Select all'), h2:has-text('Verify')")
            if prompt:
                print("\nImage selection captcha detected.")
                print("Instructions shown on page - manually select matching images.")
                print("This requires visual recognition.")
                print(f"Waiting {timeout} seconds for manual completion...")
                
                # Wait for user to complete and page to change
                try:
                    page.wait_for_function(
                        "() => document.querySelectorAll('button:disabled, input[disabled]').length === 0",
                        timeout=int(timeout * 1000),
                    )
                    return True
                except Exception:
                    print("Timeout waiting for captcha completion")
                    return False
        except Exception:
            pass
        
        return False


class IntelligentCaptchaSolver(CaptchaSolver):
    """Solver that uses a reference logo to identify matching images."""
    
    def __init__(self, reference_logo_path: str):
        if not HAS_IMAGE_ANALYSIS:
            raise ImportError("Pillow is required for intelligent solving")
        
        self.analyzer = GridAnalyzer()
        try:
            self.reference_logo = Image.open(reference_logo_path).convert("RGB")
        except FileNotFoundError:
            raise ValueError(f"Reference logo not found at: {reference_logo_path}")

    def _are_images_similar(self, img1: Image.Image, img2: Image.Image, threshold: float = 20.0) -> bool:
        """Compare two images for similarity using RMS difference."""
        try:
            # Resize images to a standard size for consistent comparison
            img1 = img1.resize((64, 64)).convert("L")
            img2 = img2.resize((64, 64)).convert("L")
            
            # Calculate the root-mean-square difference
            diff = ImageChops.difference(img1, img2)
            stat = ImageStat.Stat(diff)
            rms = stat.rms[0]
            
            print(f"Image comparison RMS: {rms}")
            return rms < threshold
        except Exception as e:
            print(f"Image comparison failed: {e}")
            return False

    def _calculate_dhash(self, img: Image.Image, hash_size: int = 8) -> int:
        """Calculate the difference hash (dHash) of an image."""
        img = img.convert('L').resize((hash_size + 1, hash_size))
        pixels = list(img.getdata())
        
        difference = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = img.getpixel((col, row))
                pixel_right = img.getpixel((col + 1, row))
                difference.append(pixel_left > pixel_right)
        
        # Convert binary array to integer
        decimal_value = 0
        for i, bit in enumerate(difference):
            if bit:
                decimal_value += 2**(i)
        
        return decimal_value

    def solve(self, captcha_data: dict) -> Optional[str]:
        """Marker return only, logic is in solve_on_page."""
        return "REQUIRES_BROWSER"

    def solve_on_page(self, page, frame=None, **kwargs) -> bool:
        """Find and click on images similar to the reference logo."""
        print("🤖 Starting intelligent captcha solving...")
        search_frame = frame or page.main_frame
        
        try:
            # Get captcha screenshot
            captcha_element = search_frame.query_selector("div:has(img[alt^='Captcha image'])")
            if not captcha_element:
                captcha_element = search_frame.query_selector("div[class*='grid']")
            if not captcha_element:
                 captcha_element = search_frame.query_selector("div:has(div:has-text('Select all'))")
            if not captcha_element:
                return False

            temp_path = "/tmp/captcha_intelligent.png"
            captcha_element.screenshot(path=temp_path)
            
            # Analyze grid to get cells and filter by type
            analysis = self.analyzer.analyze_grid(temp_path, save_analysis=False)
            if not analysis:
                print("Grid analysis failed.")
                return False

            image_cells_indices = analysis.get('image_cells', [])
            all_cells = analysis.get('cells', [])
            
            if not image_cells_indices:
                print("No image cells detected to click.")
                return False

            clickables = search_frame.query_selector_all("div[role='button'], img")

            if len(all_cells) != len(clickables):
                print(f"Warning: Found {len(all_cells)} cells but {len(clickables)} clickable elements.")

            # Group images by similarity
            hashes = {}
            for idx in image_cells_indices:
                if idx >= len(all_cells):
                    continue
                
                bbox = all_cells[idx]['bbox']
                cell_img = self.analyzer.extract_cell_image(temp_path, bbox)
                if not cell_img:
                    continue
                
                dhash = self._calculate_dhash(cell_img)
                if dhash not in hashes:
                    hashes[dhash] = []
                hashes[dhash].append(idx)

            # Find the group most similar to the reference logo
            best_match_group = None
            min_rms = float('inf')

            for dhash, indices in hashes.items():
                if not indices:
                    continue
                
                # Get a representative image from the group
                rep_idx = indices[0]
                rep_bbox = all_cells[rep_idx]['bbox']
                rep_img = self.analyzer.extract_cell_image(temp_path, rep_bbox)

                if not rep_img:
                    continue

                # Calculate similarity to reference logo
                try:
                    img1 = rep_img.resize((64, 64)).convert("L")
                    img2 = self.reference_logo.resize((64, 64)).convert("L")
                    diff = ImageChops.difference(img1, img2)
                    stat = ImageStat.Stat(diff)
                    rms = stat.rms[0]

                    if rms < min_rms:
                        min_rms = rms
                        best_match_group = indices
                except Exception as e:
                    print(f"Image comparison failed for a group: {e}")

            # Click on all images in the best matching group
            if best_match_group:
                print(
                    f"Found best match group with {len(best_match_group)} images "
                    f"(RMS: {min_rms}). Clicking them..."
                )
                for idx in best_match_group:
                    if idx < len(clickables):
                        human_click(page, clickables[idx], label=f"logo cell {idx}")
                
                print(f"✅ Clicked {len(best_match_group)} matching images.")
                page.wait_for_timeout(500)
                return True
            else:
                print("No matching image groups found.")
                return False

        except Exception as e:
            print(f"Intelligent solving failed: {e}")
            return False


class SimilaritySolver(CaptchaSolver):
    """Solver that finds and clicks on similar images in a grid."""

    def __init__(self):
        if not HAS_IMAGE_ANALYSIS:
            raise ImportError("Pillow is required for similarity solving")
        self.analyzer = GridAnalyzer()

    def _calculate_dhash(self, img: Image.Image, hash_size: int = 8) -> int:
        """Calculate the difference hash (dHash) of an image."""
        img = img.convert('L').resize((hash_size + 1, hash_size))
        pixels = list(img.getdata())
        
        difference = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = img.getpixel((col, row))
                pixel_right = img.getpixel((col + 1, row))
                difference.append(pixel_left > pixel_right)
        
        # Convert binary array to integer
        decimal_value = 0
        for i, bit in enumerate(difference):
            if bit:
                decimal_value += 2**(i)
        
        return decimal_value

    def solve(self, captcha_data: dict) -> Optional[str]:
        """Marker return only, logic is in solve_on_page."""
        return "REQUIRES_BROWSER"

    def solve_on_page(self, page, frame=None, **kwargs) -> bool:
        """Find and click on groups of similar images."""
        print("🤖 Starting similarity-based captcha solving...")
        search_frame = frame or page.main_frame
        
        try:
            captcha_element = search_frame.query_selector("div:has(img[alt^='Captcha image'])")
            if not captcha_element:
                captcha_element = search_frame.query_selector("div[class*='grid']")
            if not captcha_element:
                captcha_element = search_frame.query_selector("div:has(div:has-text('Select all'))")
            if not captcha_element:
                return False

            temp_path = "/tmp/captcha_similarity.png"
            captcha_element.screenshot(path=temp_path)
            
            detection_result = self.analyzer.detect_grid_cells(temp_path)
            if not detection_result:
                print("Could not detect grid cells.")
                return False
            
            cells, _, _ = detection_result
            clickables = search_frame.query_selector_all("div[role='button'], img")

            if len(cells) != len(clickables):
                print(f"Warning: Found {len(cells)} cells but {len(clickables)} clickable elements.")

            hashes = {}
            for idx, bbox in enumerate(cells):
                cell_img = self.analyzer.extract_cell_image(temp_path, bbox)
                if not cell_img:
                    continue
                
                dhash = self._calculate_dhash(cell_img)
                if dhash not in hashes:
                    hashes[dhash] = []
                hashes[dhash].append(idx)

            clicked_something = False
            for dhash, indices in hashes.items():
                if len(indices) >= 3:
                    print(f"Found a group of {len(indices)} similar images. Clicking them...")
                    for idx in indices:
                        if idx < len(clickables):
                            print(f"  Clicking element at index {idx}")
                            human_click(page, clickables[idx], label=f"similarity cell {idx}")
                    clicked_something = True
            
            if clicked_something:
                print("✅ Similarity-based solving process completed.")
                page.wait_for_timeout(500)
                return True
            else:
                print("No groups of 3 or more similar images found.")
                return False

        except Exception as e:
            print(f"Similarity solving failed: {e}")
            return False


class ManualCaptchaSolver(CaptchaSolver):
    """Fallback solver that prompts user to solve manually."""
    
    def solve(self, captcha_data: dict) -> Optional[str]:
        """Prompt user to manually solve captcha."""
        image_path = captcha_data.get("image_path")
        if image_path and Path(image_path).exists():
            print(f"\nPlease view the captcha image: {image_path}")
        
        print("Manual captcha solving required.")
        answer = input("Enter the captcha solution: ").strip()
        return answer if answer else None


def extract_captcha_info(page, frame=None) -> Optional[dict]:
    """
    Extract captcha information from page.
    
    Returns dict with keys:
    - type: 'recaptcha_v2', 'hcaptcha', 'image', etc.
    - sitekey: For reCAPTCHA/hCaptcha
    - image_path: For image captchas
    - pageurl: Current page URL
    """
    captcha_info = {"pageurl": page.url}
    
    search_frame = frame or page.main_frame
    print(f"Scanning for captcha indicators (frame_url={getattr(search_frame, 'url', None)})")
    
    # Check for reCAPTCHA v2
    try:
        recaptcha = search_frame.query_selector("div.g-recaptcha")
        if recaptcha:
            sitekey = recaptcha.get_attribute("data-sitekey")
            if sitekey:
                print(f"Detected reCAPTCHA v2 (sitekey={sitekey})")
                captcha_info["type"] = "recaptcha_v2"
                captcha_info["sitekey"] = sitekey
                return captcha_info
    except Exception:
        pass
    
    # Check for hCaptcha
    try:
        hcaptcha = search_frame.query_selector("div.h-captcha")
        if hcaptcha:
            sitekey = hcaptcha.get_attribute("data-sitekey")
            if sitekey:
                print(f"Detected hCaptcha (sitekey={sitekey})")
                captcha_info["type"] = "hcaptcha"
                captcha_info["sitekey"] = sitekey
                return captcha_info
    except Exception:
        pass
    
    # Check for image captcha
    try:
        captcha_img = search_frame.query_selector("img[src*='captcha']")
        if captcha_img:
            captcha_info["type"] = "image"
            src = captcha_img.get_attribute("src")
            print(f"Detected image captcha (src={'data:' if src and src.startswith('data:') else src})")
            # Handle data URIs and URLs
            if src and src.startswith("data:"):
                # Extract base64 data
                # Format: data:image/png;base64,<data>
                parts = src.split(",", 1)
                if len(parts) == 2:
                    image_data = base64.b64decode(parts[1])
                    temp_path = "/tmp/captcha_temp.png"
                    Path(temp_path).write_bytes(image_data)
                    captcha_info["image_path"] = temp_path
            else:
                captcha_info["image_path"] = src
            print(f"Image captcha path resolved: {captcha_info.get('image_path')}")
            return captcha_info
    except Exception:
        pass
    
    print("No captcha indicators found on page.")
    return None


def solve_captcha_on_page(
    page,
    solver: CaptchaSolver,
    timeout: float = 60,
    inject_solution: bool = True,
) -> bool:
    """
    Detect and solve captcha on page.
    
    Args:
        page: Playwright page object
        solver: CaptchaSolver instance
        timeout: Timeout for solving
        inject_solution: If True, attempt to inject solution into form
    
    Returns:
        True if captcha was solved, False otherwise
    """
    captcha_info = extract_captcha_info(page)
    if not captcha_info:
        print("No captcha info extracted; skipping solve.")
        return False
    
    print(f"Detected captcha type: {captcha_info.get('type')}")
    print(f"Using solver: {solver.__class__.__name__}")
    
    # Handle image selection captchas with browser automation
    if isinstance(solver, (BrowserAutomatedSolver, IntelligentCaptchaSolver, SimilaritySolver, TemplateMatchingSolver)):
        return solver.solve_on_page(page, timeout=timeout)
    
    # Handle regular solvers
    solution = solver.solve(captcha_info)
    if not solution:
        print("Failed to solve captcha")
        return False
    
    # Check for special markers
    if solution == "GRID_CAPTCHA":
        print("Grid/image selection captcha detected")
        # Try browser automation if available
        if isinstance(solver, (OCRCaptchaSolver, ManualCaptchaSolver)):
            print("Falling back to manual solving...")
            # Prompt user
            input("Please solve the captcha in the browser and press Enter...")
            return True
        return False
    
    print(f"Captcha solved: {solution}")
    
    if not inject_solution:
        return True
    
    # Attempt to inject solution
    captcha_type = captcha_info.get("type")
    
    if captcha_type in ("recaptcha_v2", "hcaptcha"):
        # For reCAPTCHA/hCaptcha, inject via response field
        try:
            # Try common response field IDs
            for field_id in ["g-recaptcha-response", "h-captcha-response", "recaptcha", "captcha"]:
                try:
                    page.evaluate(
                        f"""
                        const el = document.getElementById('{field_id}');
                        if (el) {{
                            el.value = '{solution}';
                            el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}
                        """
                    )
                except Exception:
                    pass
        except Exception as e:
            print(f"Failed to inject reCAPTCHA response: {e}")
    
    elif captcha_type == "image":
        # For image captchas, try to find and fill captcha input
        try:
            captcha_input = page.query_selector("input[name*='captcha'], input[id*='captcha']")
            if captcha_input:
                captcha_input.fill(solution)
            else:
                print("Could not find captcha input field")
        except Exception as e:
            print(f"Failed to inject image captcha solution: {e}")
    
    return True


def get_solver(config: Optional[dict] = None) -> CaptchaSolver:
    """
    Factory function to create appropriate solver based on configuration.
    
    Config dict can contain:
    - method: 'ocr', '2captcha', 'browser', 'manual', 'intelligent', 'similarity' (default: 'manual')
    - api_key: For 2captcha
    - pytesseract_path: For OCR
    - logo_path: For intelligent solver
    """
    if not config:
        config = {}
    
    method = config.get("method", "manual").lower()
    
    if method == "2captcha":
        api_key = config.get("api_key") or os.getenv("CAPTCHA_API_KEY")
        if not api_key:
            print("2captcha API key not provided (set CAPTCHA_API_KEY or pass api_key in config)")
            return ManualCaptchaSolver()
        return TwoCaptchaSolver(api_key)
    
    elif method == "ocr":
        if not HAS_OCR:
            print("pytesseract/Pillow not installed. Install with: pip install pytesseract pillow")
            return ManualCaptchaSolver()
        return OCRCaptchaSolver(config.get("pytesseract_path"))
    
    elif method == "browser":
        return BrowserAutomatedSolver(config.get("logo_path"))
        
    elif method == "intelligent":
        logo_path = config.get("logo_path")
        if not logo_path:
            print("Intelligent solver requires a logo_path in config")
            return ManualCaptchaSolver()
        return IntelligentCaptchaSolver(logo_path)
    
    elif method == "similarity":
        return SimilaritySolver()

    else:  # manual or default
        return ManualCaptchaSolver()


if __name__ == "__main__":
    print("Captcha Solver Module")
    print("Supports: Image OCR, reCAPTCHA v2, hCaptcha, 2captcha service")
