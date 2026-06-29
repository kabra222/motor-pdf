from __future__ import annotations

HAS_EASYOCR = False
try:
    import easyocr

    HAS_EASYOCR = True
except ImportError:
    pass


class EasyOCREngine:
    _reader: object = None
    _langs: list[str] = ["pt", "en"]

    def __init__(self, langs: list[str] | None = None):
        self._langs = langs or ["pt", "en"]
        self._reader = None

    def _ensure_reader(self):
        if self._reader is None and HAS_EASYOCR:
            try:
                import easyocr
                self._reader = easyocr.Reader(
                    self._langs,
                    gpu=False,
                    verbose=False,
                )
            except Exception:
                pass

    def extract_page(
        self, page_or_image: object, paragraph: bool = True, dpi: int = 200
    ) -> str:
        self._ensure_reader()
        if self._reader is None:
            return ""

        try:
            try:
                import io

                from PIL import Image
                if hasattr(page_or_image, "get_pixmap"):
                    pix = page_or_image.get_pixmap(dpi=dpi)
                    img_data = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_data))
                else:
                    pil_image = page_or_image
            except Exception:
                pil_image = page_or_image

            results = self._reader.readtext(
                pil_image,
                paragraph=paragraph,
                width_ths=0.7,
                height_ths=0.7,
            )
            lines: list[str] = []
            for _bbox, text, conf in results:
                if conf > 0.3:
                    lines.append(text.strip())
            return "\n".join(lines)
        except Exception:
            return ""

    def extract_pages_from_pdf(
        self, doc: object, pages: list[int] | None = None
    ) -> dict[int, str]:
        if not HAS_EASYOCR:
            return {}

        try:
            import io

            from PIL import Image
        except ImportError:
            return {}

        results: dict[int, str] = {}
        try:
            num_pages = len(doc)
            page_range = pages if pages is not None else range(num_pages)
            for pn in page_range:
                page = doc[pn]
                pix = page.get_pixmap(dpi=200)
                img_data = pix.tobytes("png")
                pil_img = Image.open(io.BytesIO(img_data))
                text = self.extract_page(pil_img)
                if text.strip():
                    results[pn] = text
        except Exception:
            pass

        return results


def is_easyocr_available() -> bool:
    return HAS_EASYOCR
