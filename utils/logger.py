from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable


class TryOnLogger:
    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return

        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event", "selected_items", "scale_factor", "extra"])

    def log(self, event: str, selected_items: Iterable[str], scale_factor: float, extra: str = "") -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    event,
                    ",".join(sorted(set(selected_items))),
                    f"{scale_factor:.2f}",
                    extra,
                ]
            )
