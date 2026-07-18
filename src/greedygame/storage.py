from __future__ import annotations

import csv
from pathlib import Path
from . import OUTCOMES


def normalise(value: str) -> str:
    cleaned = value.strip().title()
    if cleaned not in OUTCOMES:
        raise ValueError(f"Unknown outcome {value!r}. Use one of: {', '.join(OUTCOMES)}")
    return cleaned


class HistoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("spin,outcome\n", encoding="utf-8")

    def load(self) -> list[str]:
        with self.path.open(newline="", encoding="utf-8") as handle:
            return [row["outcome"] for row in csv.DictReader(handle) if row.get("outcome")]

    def append(self, values: list[str]) -> list[str]:
        clean = [normalise(value) for value in values if value.strip()]
        current = len(self.load())
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for offset, value in enumerate(clean, 1):
                writer.writerow([current + offset, value])
        return clean

    def import_lines(self, source: str | Path) -> list[str]:
        values = Path(source).read_text(encoding="utf-8").replace(",", "\n").splitlines()
        return self.append(values)
