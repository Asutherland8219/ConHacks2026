#!/usr/bin/env python3
"""
Image analysis and grid detection for image selection captchas.
Analyzes grid of images and colors to identify matching items.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Tuple
import hashlib

try:
    import numpy as np
    from PIL import Image
    from sklearn.cluster import KMeans
    HAS_IMAGE_ANALYSIS = True
except ImportError:
    HAS_IMAGE_ANALYSIS = False


class GridAnalyzer:
    """Analyze image grids in captchas."""
    
    def __init__(self):
        if not HAS_IMAGE_ANALYSIS:
            raise ImportError("numpy, Pillow, and scikit-learn required")
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load image from path."""
        return Image.open(image_path)
    
    def detect_grid_cells(self, image_path: str) -> Optional[Tuple[List[Tuple[int, int, int, int]], int, int]]:
        """
        Detect individual grid cells in captcha image.
        Returns a tuple containing:
        - List of (x, y, width, height) bounding boxes for each cell.
        - Number of rows in the grid.
        - Number of columns in the grid.
        """
        try:
            img = self.load_image(image_path)
            width, height = img.size
            rows = 3
            cols = 3
            cell_width = width // cols
            cell_height = height // rows
            cells = []
            for i in range(rows):
                for j in range(cols):
                    x1 = j * cell_width
                    y1 = i * cell_height
                    cells.append((x1, y1, cell_width, cell_height))
            return cells, rows, cols
        except Exception as e:
            print(f"Grid cell detection failed: {e}")
            return None

    def _group_lines(self, lines: List[int], gap: int = 50) -> List[int]:
        """Group nearby lines into grid divisions."""
        if not lines:
            return []

        groups = [lines[0]]
        for line in lines[1:]:
            if line - groups[-1] > gap:
                groups.append(line)
        return groups

    def extract_cell_image(
        self, image_path: str, cell_bbox: Tuple[int, int, int, int]
    ) -> Optional[Image.Image]:
        """Extract a single cell image."""
        try:
            img = self.load_image(image_path)
            x, y, w, h = cell_bbox
            return img.crop((x, y, x + w, y + h))
        except Exception:
            return None

    def is_plain_color(self, cell_image: Image.Image, threshold: float = 0.50) -> bool:
        """Check if cell is mostly a solid color (no content)."""
        try:
            arr = np.array(cell_image)

            # Calculate color variance
            if len(arr.shape) == 3:
                # RGB/RGBA - check color uniformity
                unique_colors = len(np.unique(arr.reshape(-1, arr.shape[2]), axis=0))
                total_pixels = arr.shape[0] * arr.shape[1]
                variance_ratio = unique_colors / total_pixels
                return variance_ratio < (1 - threshold)
            else:
                # Grayscale
                variance = np.var(arr)
                return variance < 100
        except Exception:
            return False

    def get_dominant_color(self, cell_image: Image.Image) -> Tuple[int, int, int]:
        """Get the dominant color of a cell."""
        try:
            arr = np.array(cell_image)

            if len(arr.shape) == 3:
                if arr.shape[2] == 4:  # RGBA
                    arr = arr[:, :, :3]
                # Reshape and cluster
                pixels = arr.reshape(-1, 3)
                kmeans = KMeans(n_clusters=1, random_state=0, n_init=10)
                kmeans.fit(pixels)
                color = kmeans.cluster_centers_[0]
                return tuple(int(c) for c in color)
            else:
                # Grayscale
                gray_val = int(np.mean(arr))
                return (gray_val, gray_val, gray_val)
        except Exception:
            return (128, 128, 128)

    def analyze_grid(
        self, image_path: str, save_analysis: bool = True
    ) -> Optional[dict]:
        """
        Analyze entire grid: detect cells, identify colors vs images.
        
        Returns:
        {
            "cells": [
                {"index": 0, "bbox": (x, y, w, h), "type": "image", "content": "logo"},
                {"index": 1, "bbox": (...), "type": "color", "color": (255, 0, 0)},
                ...
            ],
            "grid_size": (rows, cols),
            "image_cells": [0, 2, 5],
            "color_cells": [1, 3, 4],
        }
        """
        try:
            detection_result = self.detect_grid_cells(image_path)
            if not detection_result:
                return None
            
            cells, grid_rows, grid_cols = detection_result
            
            analysis = {
                "image_path": image_path,
                "grid_size": (grid_rows, grid_cols),
                "cells": [],
                "image_cells": [],
                "color_cells": [],
            }
            
            for idx, bbox in enumerate(cells):
                cell_img = self.extract_cell_image(image_path, bbox)
                if not cell_img:
                    continue
                
                is_plain = self.is_plain_color(cell_img)
                
                cell_info = {
                    "index": idx,
                    "bbox": bbox,
                    "type": "color" if is_plain else "image",
                }
                
                if is_plain:
                    color = self.get_dominant_color(cell_img)
                    cell_info["color"] = color
                    analysis["color_cells"].append(idx)
                else:
                    analysis["image_cells"].append(idx)
                
                analysis["cells"].append(cell_info)
            
            if save_analysis:
                # Save analysis
                analysis_path = Path(image_path).parent / f"{Path(image_path).stem}_analysis.json"
                with open(analysis_path, "w") as f:
                    # Convert types for JSON serialization
                    data = analysis.copy()
                    data["grid_size"] = [int(v) for v in data["grid_size"]]
                    for cell in data["cells"]:
                        cell["bbox"] = [int(v) for v in cell["bbox"]]
                        if "color" in cell:
                            cell["color"] = [int(v) for v in cell["color"]]
                    json.dump(data, f, indent=2)
            
            return analysis
        except Exception as e:
            print(f"Grid analysis failed: {e}")
            return None


class LogoDetector:
    """Detect logo/image objects in captcha cells."""
    
    def __init__(self):
        if not HAS_IMAGE_ANALYSIS:
            raise ImportError("Image analysis dependencies required")
    
    def detect_object_type(self, cell_image: Image.Image) -> Optional[str]:
        """
        Attempt to detect what object is in the image.
        
        Returns one of: 'logo', 'traffic_light', 'crosswalk', 'car', 'bus', etc.
        """
        try:
            arr = np.array(cell_image)
            
            # Calculate basic features
            edges = self._count_edges(arr)
            complexity = self._calculate_complexity(arr)
            
            # Simple heuristics for common captcha objects
            if complexity > 0.7:
                # Complex image likely a logo
                return "logo"
            elif edges > 0.4:
                # High edge count might be traffic light or crosswalk
                # Check aspect ratio
                if arr.shape[0] > arr.shape[1] * 1.2:
                    return "traffic_light"
                else:
                    return "crosswalk"
            else:
                return "object"
        except Exception:
            return None
    
    def _count_edges(self, arr: np.ndarray) -> float:
        """Count edge pixels as ratio of total."""
        try:
            if len(arr.shape) == 3:
                gray = np.mean(arr, axis=2)
            else:
                gray = arr
            
            edges = np.abs(np.diff(gray, axis=0)).sum() + np.abs(np.diff(gray, axis=1)).sum()
            total = gray.size * 255
            return edges / total
        except Exception:
            return 0
    
    def _calculate_complexity(self, arr: np.ndarray) -> float:
        """Calculate color complexity (0-1)."""
        try:
            if len(arr.shape) == 3:
                pixels = arr.reshape(-1, arr.shape[2])
            else:
                pixels = arr.reshape(-1, 1)
            
            unique_colors = len(np.unique(pixels, axis=0))
            total_pixels = pixels.shape[0]
            return unique_colors / total_pixels
        except Exception:
            return 0


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 captcha_image_analysis.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)
    
    analyzer = GridAnalyzer()
    analysis = analyzer.analyze_grid(image_path)
    
    if analysis:
        print(f"\nGrid analysis for {image_path}")
        print(f"Grid size: {analysis['grid_size'][0]}x{analysis['grid_size'][1]}")
        print(f"Image cells (items to click): {analysis['image_cells']}")
        print(f"Color cells (background): {analysis['color_cells']}")
        print(f"\nDetailed cells:")
        for cell in analysis['cells']:
            if cell['type'] == 'image':
                print(f"  [{cell['index']}] Image cell")
            else:
                print(f"  [{cell['index']}] Color: {cell.get('color', 'unknown')}")
    else:
        print("Analysis failed")
