"""CSV encoding and delimiter sniffing helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DELIMITER = ";"
DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1251")


@dataclass(frozen=True)
class CsvSniffResult:
    """Detected CSV read options."""

    encoding: str
    delimiter: str


def detect_text_encoding(path: str | Path, encodings: tuple[str, ...] = DEFAULT_ENCODINGS) -> str:
    """Detect a compatible encoding by trial-decoding a file prefix."""
    file_path = Path(path)
    raw_bytes = file_path.read_bytes()[:65536]

    for encoding in encodings:
        try:
            raw_bytes.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    return encodings[0]


def detect_csv_delimiter(path: str | Path, encoding: str, fallback: str = DEFAULT_DELIMITER) -> str:
    """Detect a likely delimiter for a CSV file."""
    file_path = Path(path)
    sample = file_path.read_text(encoding=encoding, errors="replace")[:8192]
    if not sample.strip():
        return fallback

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
    except csv.Error:
        return fallback

    return dialect.delimiter or fallback


def sniff_csv_options(path: str | Path) -> CsvSniffResult:
    """Detect CSV encoding and delimiter for raw ingestion."""
    encoding = detect_text_encoding(path)
    delimiter = detect_csv_delimiter(path, encoding=encoding)
    return CsvSniffResult(encoding=encoding, delimiter=delimiter)
