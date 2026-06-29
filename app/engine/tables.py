from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HAS_CAMELOT = False
try:
    import camelot

    HAS_CAMELOT = True
except ImportError:
    pass


@dataclass
class ExtractedTable:
    page: int
    markdown: str
    csv: str
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    accuracy: float = 0.0
    method: str = ""


def extract_tables_camelot(
    path: str | Path,
    pages: str = "all",
    flavor: str = "lattice",
) -> list[ExtractedTable]:
    if not HAS_CAMELOT:
        return []

    tables: list[ExtractedTable] = []
    try:
        parsed = camelot.read_pdf(
            str(path),
            pages=pages,
            flavor=flavor,
            suppress_stdout=True,
        )
    except Exception:
        return []

    for t in parsed:
        try:
            md = _camelot_to_markdown(t)
            csv_data = t.df.to_csv(index=False)
            x0 = float(t._bbox[0]) if t._bbox else 0
            y0 = float(t._bbox[1]) if t._bbox else 0
            x1 = float(t._bbox[2]) if t._bbox else 0
            y1 = float(t._bbox[3]) if t._bbox else 0
            tables.append(ExtractedTable(
                page=t.page,
                markdown=md,
                csv=csv_data,
                bbox=(x0, y0, x1, y1),
                accuracy=float(t.parsing_report.get("accuracy", 0)),
                method=flavor,
            ))
        except Exception:
            continue

    return tables


def extract_tables_hybrid(
    path: str | Path,
    pages: str = "all",
) -> list[ExtractedTable]:
    if not HAS_CAMELOT:
        return []

    tables = extract_tables_camelot(path, pages, flavor="lattice")

    if not tables:
        tables = extract_tables_camelot(path, pages, flavor="stream")

    return tables


def _camelot_to_markdown(table) -> str:
    rows: list[str] = []
    for _, row in table.df.iterrows():
        cells = [str(c or "").strip() for c in row]
        rows.append("| " + " | ".join(cells) + " |")

    if len(rows) >= 2:
        ncols = len(rows[0].split("|")) - 2
        sep = "| " + " | ".join(["---"] * ncols) + " |"
        rows.insert(1, sep)

    return "\n".join(rows)


def is_camelot_available() -> bool:
    return HAS_CAMELOT
