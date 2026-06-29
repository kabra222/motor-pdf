from __future__ import annotations

import logging
from pathlib import Path

import fitz
import numpy as np

logger = logging.getLogger(__name__)

_HAS_EASYOCR = False
try:
    import easyocr
    _HAS_EASYOCR = True
except ImportError:
    pass

_HAS_DOCTR = False
try:
    import doctr
    _HAS_DOCTR = True
except ImportError:
    pass

_HAS_TESSERACT = False
try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    pass


class HybridOCR:
    """Hybrid OCR engine that chains backends: EasyOCR -> docTR -> Tesseract."""

    def __init__(
        self,
        chain: list[str] | None = None,
        min_confidence: float = 0.5,
        lang: str = "pt",
    ):
        self.chain = chain or ["easyocr", "doctr", "tesseract"]
        self.min_confidence = min_confidence
        self.lang = lang
        self._easyocr_reader = None
        self._doctr_predictor = None

    @property
    def is_available(self) -> bool:
        return len(self.available_backends()) > 0

    def available_backends(self) -> list[str]:
        available: list[str] = []
        if _HAS_EASYOCR:
            available.append("easyocr")
        if _HAS_DOCTR:
            available.append("doctr")
        if _HAS_TESSERACT:
            available.append("tesseract")
        return available

    def extract_page(self, page: fitz.Page) -> str:
        text, _ = self._run_chain(self._page_to_array(page))
        return text

    def extract_image(self, image_path: str | Path) -> str:
        from PIL import Image
        pil_img = Image.open(image_path).convert("RGB")
        arr = np.array(pil_img)
        text, _ = self._run_chain(arr)
        return text

    def _page_to_array(self, page: fitz.Page) -> np.ndarray:
        pix = page.get_pixmap()
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            arr = arr[:, :, :3]
        elif pix.n == 1:
            arr = np.stack([arr.squeeze()] * 3, axis=-1)
        return arr

    def _run_chain(self, image: np.ndarray) -> tuple[str, float]:
        available = self.available_backends()
        for backend in self.chain:
            if backend not in available:
                logger.debug("Backend '%s' not available, skipping", backend)
                continue
            method = getattr(self, f"_ocr_{backend}", None)
            if method is None:
                continue
            try:
                text, conf = method(image)
            except Exception:
                logger.debug("Backend '%s' failed", backend, exc_info=True)
                continue
            if conf >= self.min_confidence:
                return text, conf
        return "", 0.0

    def _ocr_easyocr(self, image: np.ndarray) -> tuple[str, float]:
        if not _HAS_EASYOCR:
            return "", 0.0
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader([self.lang], gpu=False, verbose=False)
        results = self._easyocr_reader.readtext(image)
        if not results:
            return "", 0.0
        lines: list[str] = []
        total_conf = 0.0
        for _, text, conf in results:
            stripped = text.strip()
            if stripped:
                lines.append(stripped)
                total_conf += conf
        if not lines:
            return "", 0.0
        return "\n".join(lines), total_conf / len(lines)

    def _ocr_doctr(self, image: np.ndarray) -> tuple[str, float]:
        if not _HAS_DOCTR:
            return "", 0.0
        from doctr.models import ocr_predictor
        if self._doctr_predictor is None:
            self._doctr_predictor = ocr_predictor(pretrained=True)
        result = self._doctr_predictor([image])
        lines: list[str] = []
        total_conf = 0.0
        word_count = 0
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_words: list[str] = []
                    for word in line.words:
                        line_words.append(word.value)
                        total_conf += word.confidence
                        word_count += 1
                    if line_words:
                        lines.append(" ".join(line_words))
        if not lines or word_count == 0:
            return "", 0.0
        return "\n".join(lines), total_conf / word_count

    def _ocr_tesseract(self, image: np.ndarray) -> tuple[str, float]:
        if not _HAS_TESSERACT:
            return "", 0.0
        text = pytesseract.image_to_string(image, lang=self.lang)
        text = text.strip()
        if not text:
            return "", 0.0
        data = pytesseract.image_to_data(
            image, lang=self.lang, output_type=pytesseract.Output.DICT
        )
        confs = [c for c in data["conf"] if c != -1]
        avg_conf = sum(confs) / len(confs) / 100.0 if confs else 0.0
        return text, avg_conf
