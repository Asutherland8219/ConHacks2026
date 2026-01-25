#!/usr/bin/env python3
"""
Captcha database and reference manager.
Stores and retrieves captcha solutions for reuse.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CaptchaRecord:
    """Record of a captcha with its solution."""
    
    sha256: str  # Image hash
    timestamp: str  # When it was solved
    image_path: str  # Path to captcha image
    prompt: str  # What to select
    solution_indices: List[int]  # Which images were selected
    solver_method: str  # How it was solved
    success: bool  # Was it successful
    notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class CaptchaDatabase:
    """Manage captcha records for reference and reuse."""
    
    def __init__(self, db_path: str = "captcha_logs/captcha_db.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[CaptchaRecord] = []
        self.load()
    
    def load(self) -> None:
        """Load database from file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.records = [
                        CaptchaRecord(
                            sha256=r["sha256"],
                            timestamp=r["timestamp"],
                            image_path=r["image_path"],
                            prompt=r["prompt"],
                            solution_indices=r["solution_indices"],
                            solver_method=r["solver_method"],
                            success=r["success"],
                            notes=r.get("notes"),
                        )
                        for r in data
                    ]
                    print(f"Loaded {len(self.records)} captcha records from database")
            except Exception as e:
                print(f"Failed to load database: {e}")
                self.records = []
    
    def save(self) -> None:
        """Save database to file."""
        try:
            data = [r.to_dict() for r in self.records]
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save database: {e}")
    
    def add_record(
        self,
        sha256: str,
        image_path: str,
        prompt: str,
        solution_indices: List[int],
        solver_method: str,
        success: bool,
        notes: Optional[str] = None,
    ) -> None:
        """Add a captcha record to the database."""
        record = CaptchaRecord(
            sha256=sha256,
            timestamp=datetime.now().isoformat(),
            image_path=image_path,
            prompt=prompt,
            solution_indices=solution_indices,
            solver_method=solver_method,
            success=success,
            notes=notes,
        )
        self.records.append(record)
        self.save()
        print(f"Recorded captcha solution: {prompt} → {solution_indices}")
    
    def find_by_hash(self, sha256: str) -> Optional[CaptchaRecord]:
        """Find a captcha record by image hash."""
        for record in self.records:
            if record.sha256 == sha256:
                return record
        return None
    
    def find_by_prompt(self, prompt: str) -> List[CaptchaRecord]:
        """Find all captcha records matching a prompt."""
        return [r for r in self.records if r.prompt.lower() == prompt.lower()]
    
    def get_statistics(self) -> dict:
        """Get statistics about the database."""
        if not self.records:
            return {
                "total_records": 0,
                "success_rate": 0,
                "unique_prompts": 0,
                "solver_methods": {},
                "prompts": [],
            }
        
        success_count = sum(1 for r in self.records if r.success)
        prompts = set(r.prompt for r in self.records)
        methods = {}
        for r in self.records:
            methods[r.solver_method] = methods.get(r.solver_method, 0) + 1
        
        return {
            "total_records": len(self.records),
            "success_count": success_count,
            "success_rate": f"{100 * success_count / len(self.records):.1f}%",
            "unique_prompts": len(prompts),
            "solver_methods": methods,
            "prompts": sorted(list(prompts)),
        }
    
    def print_statistics(self) -> None:
        """Print database statistics."""
        stats = self.get_statistics()
        print("\n=== Captcha Database Statistics ===")
        print(f"Total records: {stats['total_records']}")
        print(f"Success rate: {stats['success_rate']}")
        print(f"Unique prompts: {stats['unique_prompts']}")
        print(f"Solver methods: {stats['solver_methods']}")
        if stats['prompts']:
            print(f"Prompts seen:")
            for prompt in stats['prompts']:
                count = len(self.find_by_prompt(prompt))
                print(f"  - {prompt} ({count} times)")


if __name__ == "__main__":
    db = CaptchaDatabase()
    db.print_statistics()
