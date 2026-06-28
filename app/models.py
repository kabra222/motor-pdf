from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field


class FormatType(str, Enum):
    text = "text"
    markdown = "markdown"
    semantic_html = "semantic_html"


class JobStatus(str, Enum):
    processing = "processing"
    done = "done"
    error = "error"


class Chunk(BaseModel):
    index: int
    text: str
    page: int
    section: Optional[str] = None
    heading: Optional[str] = None
    tokens: int = 0
    start_char: int = 0
    end_char: int = 0


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    page: int
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    font_size: float = 12
    font: str = ""
    is_heading: bool = False
    heading_level: Optional[int] = None
    is_list_item: bool = False
    list_type: Optional[str] = None


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    page: int
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    width: float = 0
    height: float = 0
    image_ref: Optional[int] = None


class TableBlock(BaseModel):
    type: Literal["table"] = "table"
    page: int
    markdown: str = ""
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)


Block = Annotated[TextBlock | ImageBlock | TableBlock, Field(discriminator="type")]


class HeadingInfo(BaseModel):
    level: int
    text: str
    page: int
    bbox: tuple[float, float, float, float]


class QualityDimension(BaseModel):
    score: float
    detail: str


class ExtractionResult(BaseModel):
    filename: str
    text: str
    chunks: list[Chunk]
    blocks: list[Block]
    headings: list[HeadingInfo]
    tables: list[str]
    images: list[dict]
    metadata: dict
    num_pages: int
    num_chunks: int
    scanned_pages: list[int]
    quality: Optional[dict] = None


class DocumentJob(BaseModel):
    id: str
    status: JobStatus = JobStatus.processing
    result: Optional[ExtractionResult] = None
    error: Optional[str] = None
    progress: int = 0
    total: int = 0


class JobResult(BaseModel):
    id: str
    status: JobStatus
    result: Optional[ExtractionResult] = None
    error: Optional[str] = None
    progress: int = 0
    total: int = 0


class ExtractRequest(BaseModel):
    chunk_size: int = Field(default=2000, ge=100, le=32000)
    chunk_overlap: int = Field(default=200, ge=0, le=4000)
    use_ocr: bool = False
    password: Optional[str] = None
    format: FormatType = FormatType.text
    model: str = "gpt-4"
    extract_images: bool = False

    model_config = {"use_enum_values": True}


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
