from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import fitz
import pdfplumber

from app.engine.quality import score_extraction

_LIST_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^[-•–]\s"), "bullet"),
    (re.compile(r"^\*\s"), "bullet"),
    (re.compile(r"^\d+[.)]\s"), "ordered"),
    (re.compile(r"^[a-z][.)]\s"), "ordered"),
]


def extract_text(
    path: str | Path,
    use_ocr: bool = False,
    password: str | None = None,
    extract_images: bool = False,
    parallel: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {path}")

    doc = fitz.open(path)

    if doc.is_encrypted:
        if not password:
            raise ValueError("PDF criptografado. Forneça uma senha.")
        if not doc.authenticate(password):
            raise ValueError("Senha inválida")

    metadata = {
        "title": doc.metadata.get("title") or "",
        "author": doc.metadata.get("author") or "",
        "subject": doc.metadata.get("subject") or "",
        "keywords": doc.metadata.get("keywords") or "",
        "encrypted": doc.is_encrypted,
    }

    num_pages = len(doc)
    if progress:
        progress(0, num_pages, "scanning")

    ocr_engine = None
    if use_ocr:
        from .ocr import OCREngine
        ocr_engine = OCREngine()

    # ── detect body font size ─────────────────────────────────────
    size_mass: dict[float, int] = {}
    for page_num in range(num_pages):
        page = doc[page_num]
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if not _is_text_block(block):
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if len(text) > 1:
                        sz = round(span["size"], 1)
                        if 7 <= sz <= 30:
                            size_mass[sz] = size_mass.get(sz, 0) + len(text)

    body_size = max(size_mass, key=size_mass.get) if size_mass else 12

    # ── page extraction ───────────────────────────────────────────
    import math
    _prog_interval = max(1, num_pages // 50)
    _safe_progress = (lambda p, t, s: None) if progress is None else (
        lambda p, t, s: progress(p, t, s) if p % _prog_interval == 0 or p == t else None
    )
    use_parallel = parallel and num_pages > 4
    if use_parallel:
        page_results = _parallel_extract(doc, path, num_pages, body_size, use_ocr, ocr_engine, _safe_progress)
    else:
        page_results = _sequential_extract(doc, num_pages, body_size, use_ocr, ocr_engine, _safe_progress)

    # ── column reordering ─────────────────────────────────────────
    all_blocks: list[dict] = []
    all_headings: list[dict] = []
    pages_text: list[str] = []
    scanned_pages: list[int] = []

    for pr in page_results:
        pr["blocks"] = _reorder_columns(pr["blocks"])
        all_blocks.extend(pr["blocks"])
        all_headings.extend(pr["headings"])
        pages_text.append(pr["text"])
        if pr.get("scanned"):
            scanned_pages.append(pr["page"])

    all_text = "\n".join(pages_text)

    # ── tables ────────────────────────────────────────────────────
    if progress:
        progress(num_pages, num_pages, "tables")

    tables: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                if progress:
                    progress(i + 1, num_pages, "tables")
                extracted = page.extract_tables()
                for table in extracted:
                    rows: list[str] = []
                    for row in table:
                        cells = [str(c or "") for c in row]
                        rows.append("| " + " | ".join(cells) + " |")
                    if rows:
                        md = "\n".join(rows)
                        tables.append(md)
                        all_blocks.append({
                            "type": "table",
                            "page": i,
                            "markdown": md,
                            "bbox": (0, 0, 0, 0),
                        })
    except Exception:
        pass

    # ── images ────────────────────────────────────────────────────
    images: list[dict] = []
    if extract_images:
        if progress:
            progress(0, 1, "images")
        img_count = 0
        for page_num in range(num_pages):
            page = doc[page_num]
            for img in page.get_images(full=True):
                if img_count >= 20:
                    break
                xref = img[0]
                try:
                    base = doc.extract_image(xref)
                    if base["width"] * base["height"] < 2_000_000:
                        import base64 as b64
                        b64_str = b64.b64encode(base["image"]).decode()
                        images.append({
                            "data": b64_str,
                            "format": base.get("ext", "png"),
                            "width": base["width"],
                            "height": base["height"],
                            "page": page_num,
                        })
                        img_count += 1
                except Exception:
                    pass
            if img_count >= 20:
                break

    doc.close()

    # ── build result ──────────────────────────────────────────────
    result = {
        "text": all_text.strip(),
        "blocks": all_blocks,
        "headings": all_headings,
        "pages_text": pages_text,
        "tables": tables,
        "images": images,
        "metadata": metadata,
        "num_pages": num_pages,
        "scanned_pages": scanned_pages,
    }

    result["quality"] = score_extraction(result)

    if progress:
        progress(num_pages, num_pages, "done")

    return result


# ── page processing ──────────────────────────────────────────────


def _process_single_page(
    doc: fitz.Document,
    page_num: int,
    body_size: float,
    use_ocr: bool,
    ocr_engine: object | None,
) -> dict:
    page = doc[page_num]
    page_dict = page.get_text("dict")

    text_blocks_raw = [b for b in page_dict["blocks"] if _is_text_block(b)]
    total_chars = sum(_block_text_len(b) for b in text_blocks_raw)

    page_width = page.rect.width
    blocks: list[dict] = []
    headings: list[dict] = []
    scanned = False

    if total_chars < 50 and ocr_engine:
        scanned = True
        ocr_text = ocr_engine.extract_page(page)
        if ocr_text.strip():
            blocks.append({
                "type": "text",
                "text": ocr_text,
                "page": page_num,
                "bbox": (0, 0, 0, 0),
                "font_size": body_size,
                "font": "",
                "is_heading": False,
                "heading_level": None,
                "is_list_item": False,
                "list_type": None,
            })
        return {
            "page": page_num,
            "blocks": blocks,
            "headings": headings,
            "text": ocr_text,
            "page_width": page_width,
            "scanned": scanned,
        }

    for block in page_dict["blocks"]:
        if _is_text_block(block):
            full_text = _block_text(block).strip()
            if not full_text:
                continue

            first_span = block["lines"][0]["spans"][0]
            font_size = round(first_span["size"], 1)
            font_name = first_span["font"]

            ratio = font_size / body_size if body_size > 0 else 1
            is_bold = "Bold" in font_name or "Black" in font_name
            is_heading = ratio >= 1.2 or (ratio >= 1.0 and is_bold)

            heading_level = None
            if is_heading:
                if ratio >= 1.8:
                    heading_level = 1
                elif ratio >= 1.5:
                    heading_level = 2
                elif ratio >= 1.3:
                    heading_level = 3
                else:
                    heading_level = 4

            is_list_item = False
            list_type: str | None = None
            if not is_heading:
                for pattern, lt in _LIST_PATTERNS:
                    if pattern.match(full_text):
                        is_list_item = True
                        list_type = lt
                        break

            blocks.append({
                "type": "text",
                "text": full_text,
                "page": page_num,
                "bbox": block["bbox"],
                "font_size": font_size,
                "font": font_name,
                "is_heading": is_heading,
                "heading_level": heading_level,
                "is_list_item": is_list_item,
                "list_type": list_type,
            })

            if is_heading:
                headings.append({
                    "level": heading_level,
                    "text": full_text,
                    "page": page_num,
                    "bbox": block["bbox"],
                })

        elif "width" in block or "height" in block:
            blocks.append({
                "type": "image",
                "page": page_num,
                "bbox": block.get("bbox", (0, 0, 0, 0)),
                "width": block.get("width", 0),
                "height": block.get("height", 0),
                "image_ref": None,
            })

    page_text = _blocks_to_plain_text(blocks, page_num)

    return {
        "page": page_num,
        "blocks": blocks,
        "headings": headings,
        "text": page_text,
        "page_width": page_width,
        "scanned": scanned,
    }


def _sequential_extract(
    doc: fitz.Document,
    num_pages: int,
    body_size: float,
    use_ocr: bool,
    ocr_engine: object | None,
    progress: Callable | None,
) -> list[dict]:
    results = []
    for pn in range(num_pages):
        if progress:
            progress(pn + 1, num_pages, "extracting")
        results.append(_process_single_page(doc, pn, body_size, use_ocr, ocr_engine))
    return results


def _parallel_extract(
    doc: fitz.Document,
    path: Path,
    num_pages: int,
    body_size: float,
    use_ocr: bool,
    ocr_engine: object | None,
    progress: Callable | None,
) -> list[dict]:
    results: dict[int, dict] = {}
    workers = max(2, min(4, num_pages // 20))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_single_page, doc, pn, body_size, use_ocr, ocr_engine): pn
            for pn in range(num_pages)
        }
        for future in as_completed(futures):
            pn = futures[future]
            results[pn] = future.result()
            if progress:
                progress(pn + 1, num_pages, "extracting")

    return [results[pn] for pn in range(num_pages)]


# ── column detection ─────────────────────────────────────────────


def _reorder_columns(blocks: list[dict]) -> list[dict]:
    text_blocks = [(i, b) for i, b in enumerate(blocks) if b.get("type") == "text"]
    if len(text_blocks) < 3:
        return blocks

    sorted_tb = sorted(text_blocks, key=lambda x: (x[1]["bbox"][1], x[1]["bbox"][0]))

    rows: list[list[tuple[int, dict]]] = []
    used: set[int] = set()

    for i in range(len(sorted_tb)):
        if i in used:
            continue
        row = [sorted_tb[i]]
        used.add(i)
        b_y0, b_y1 = sorted_tb[i][1]["bbox"][1], sorted_tb[i][1]["bbox"][3]
        b_h = max(b_y1 - b_y0, 1)

        for j in range(i + 1, len(sorted_tb)):
            if j in used:
                continue
            c_y0, c_y1 = sorted_tb[j][1]["bbox"][1], sorted_tb[j][1]["bbox"][3]
            overlap = min(b_y1, c_y1) - max(b_y0, c_y0)
            if overlap > 0:
                row.append(sorted_tb[j])
                used.add(j)

        row.sort(key=lambda x: x[1]["bbox"][0])
        rows.append(row)

    gaps: list[float] = []
    for row in rows:
        if len(row) >= 2:
            for k in range(len(row) - 1):
                gap = row[k + 1][1]["bbox"][0] - row[k][1]["bbox"][2]
                if gap > 15:
                    gaps.append(row[k][1]["bbox"][2])

    if not gaps:
        return blocks

    gap_x = sorted(gaps)[len(gaps) // 2]

    left: list[dict] = []
    right: list[dict] = []

    for _, b in text_blocks:
        cx = (b["bbox"][0] + b["bbox"][2]) / 2
        (left if cx < gap_x else right).append(b)

    left.sort(key=lambda b: b["bbox"][1])
    right.sort(key=lambda b: b["bbox"][1])

    reordered_text = left + right
    non_text = [b for b in blocks if b.get("type") != "text"]
    return reordered_text + non_text


# ── helpers ──────────────────────────────────────────────────────


def _is_text_block(block: dict) -> bool:
    return "lines" in block


def _block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        line_text = "".join(span["text"] for span in line.get("spans", []))
        lines.append(line_text)
    return "\n".join(lines)


def _block_text_len(block: dict) -> int:
    total = 0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            total += len(span["text"])
    return total


def _blocks_to_plain_text(blocks: list[dict], page_num: int) -> str:
    lines: list[str] = []
    for b in blocks:
        if b.get("type") != "text" or b.get("page") != page_num:
            continue
        text = b["text"]
        if b.get("is_heading") and b.get("heading_level"):
            prefix = "#" * b["heading_level"] + " "
            lines.append(prefix + text)
        else:
            lines.append(text)
    return "\n".join(lines)
