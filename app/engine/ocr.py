from __future__ import annotations

import io

import fitz
from PIL import Image


class OCREngine:
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self._reader = None

    def _lazy_init(self):
        if self._reader is None:
            from paddleocr import PaddleOCR
            self._reader = PaddleOCR(
                lang=self.lang, show_log=False, use_angle_cls=True
            )

    def extract_text(self, image: Image.Image) -> str:
        self._lazy_init()
        result = self._reader.ocr(image, cls=True)
        if not result or not result[0]:
            return ""
        lines = []
        for line in result[0]:
            text = line[1][0]
            if text.strip():
                lines.append(text)
        return "\n".join(lines)

    def extract_page(self, page: fitz.Page) -> str:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return self.extract_text(img)
