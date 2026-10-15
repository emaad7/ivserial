"""Reading data files: CSV, TXT (one column or delimited) and Excel (needs pandas + openpyxl)."""
from __future__ import annotations

import csv
import os
from typing import Dict, List

import numpy as np


def _to_float(s: str):
    s = s.strip().replace(",", ".") if s.count(",") == 1 and "." not in s else s.strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_table(path: str) -> Dict[str, np.ndarray]:
    """Return {column name: numeric array} for every column that is mostly numeric."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("reading Excel files needs pandas and openpyxl: pip install pandas openpyxl") from e
        df = pd.read_excel(path)
        cols = {}
        for c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            if np.isfinite(v).mean() > 0.5:
                cols[str(c)] = v
        return cols

    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("the file is empty")
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:50]), delimiters=",;\t |")
        delim = dialect.delimiter
    except csv.Error:
        delim = None
    rows: List[List[str]] = [l.split() if delim in (None, " ") else next(csv.reader([l], delimiter=delim))
                             for l in lines]
    first = rows[0]
    has_header = any(np.isnan(_to_float(c)) for c in first if c.strip())
    header = [c.strip() or f"column {i + 1}" for i, c in enumerate(first)] if has_header else \
             [f"column {i + 1}" for i in range(len(first))]
    body = rows[1:] if has_header else rows
    ncol = len(header)
    cols = {}
    for j in range(ncol):
        v = np.array([_to_float(r[j]) if j < len(r) else np.nan for r in body], float)
        if v.size and np.isfinite(v).mean() > 0.5:
            cols[header[j]] = v
    if not cols:
        raise ValueError("no numeric column found")
    return cols
