from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class FormatType(StrEnum):
    text = "text"
    markdown = "markdown"
    semantic_html = "semantic_html"


class JobStatus(StrEnum):
    processing = "processing"
    done = "done"
    error = "error"


class Chunk(BaseModel):
    """Um segmento do texto extraído, com metadados de localização."""

    index: int = Field(description="Índice sequencial do chunk no documento")
    text: str = Field(description="Conteúdo textual do chunk")
    page: int = Field(description="Número da página de onde o chunk foi extraído")
    section: str | None = Field(None, description="Seção do documento (ex: 'Capítulo 1')")
    heading: str | None = Field(None, description="Título da seção imediatamente anterior")
    tokens: int = Field(0, description="Contagem de tokens (tiktoken)")
    depth: int = Field(0, description="Profundidade hierárquica da seção")
    start_char: int = Field(0, description="Posição inicial no texto completo")
    end_char: int = Field(0, description="Posição final no texto completo")


class TextBlock(BaseModel):
    """Bloco de texto com metadados de formatação e classificação."""

    type: Literal["text"] = "text"
    text: str = Field(description="Conteúdo do bloco")
    page: int = Field(description="Número da página")
    bbox: tuple[float, float, float, float] = Field(
        (0, 0, 0, 0), description="Bounding box (x0, y0, x1, y1)"
    )
    font_size: float = Field(12, description="Tamanho da fonte em pt")
    font: str = Field("", description="Nome da fonte")
    is_heading: bool = Field(False, description="Se é um heading detectado por fonte")
    heading_level: int | None = Field(None, description="Nível do heading (1-6)")
    is_list_item: bool = Field(False, description="Se é item de lista")
    list_type: str | None = Field(None, description="Tipo da lista (ordered/bullet)")
    layout_type: str | None = Field(
        None, description="Classificação semântica (heading/paragraph/short_text/header/footer/page_number/list_item/table_cell)"
    )
    is_noise: bool = Field(False, description="Se é ruído (header/footer/page_number repetido)")


class ImageBlock(BaseModel):
    """Bloco de imagem com dimensões e referência."""

    type: Literal["image"] = "image"
    page: int = Field(description="Número da página")
    bbox: tuple[float, float, float, float] = Field(
        (0, 0, 0, 0), description="Bounding box (x0, y0, x1, y1)"
    )
    width: float = Field(0, description="Largura em px")
    height: float = Field(0, description="Altura em px")
    image_ref: int | None = Field(None, description="Índice de referência interno")


class TableBlock(BaseModel):
    """Bloco de tabela com representação markdown."""

    type: Literal["table"] = "table"
    page: int = Field(description="Número da página")
    markdown: str = Field("", description="Tabela em formato markdown")
    bbox: tuple[float, float, float, float] = Field(
        (0, 0, 0, 0), description="Bounding box (x0, y0, x1, y1)"
    )


Block = Annotated[TextBlock | ImageBlock | TableBlock, Field(discriminator="type")]


class HeadingInfo(BaseModel):
    """Informação sobre um heading detectado no documento."""

    level: int = Field(description="Nível hierárquico (1 = capítulo, 2 = seção, etc)")
    text: str = Field(description="Texto do heading")
    page: int = Field(description="Página onde aparece")
    bbox: tuple[float, float, float, float] = Field(
        description="Bounding box (x0, y0, x1, y1)"
    )


class QualityDimension(BaseModel):
    """Métrica de qualidade para uma dimensão específica."""

    score: float = Field(description="Pontuação normalizada (0-1)")
    detail: str = Field(description="Descrição textual da métrica")


class ExtractionResult(BaseModel):
    """Resultado completo da extração de um PDF."""

    filename: str = Field(description="Nome do arquivo original")
    text: str = Field(description="Texto extraído completo, pós-processado")
    chunks: list[Chunk] = Field(description="Chunks semânticos do texto")
    blocks: list[Block] = Field(description="Blocos extraídos com metadados")
    headings: list[HeadingInfo] = Field(description="Headings detectados")
    tables: list[str] = Field(description="Tabelas em formato markdown")
    images: list[dict] = Field(description="Imagens extraídas (base64 + descrição opcional)")
    metadata: dict = Field(description="Metadados do PDF (título, autor, layout, etc)")
    num_pages: int = Field(description="Número total de páginas")
    num_chunks: int = Field(description="Número de chunks gerados")
    scanned_pages: list[int] = Field(
        default_factory=list, description="Páginas processadas por OCR"
    )
    quality: dict | None = Field(None, description="Métricas de qualidade da extração")
    classified_count: int = Field(0, description="Quantidade de blocos classificados semanticamente")
    annotations: list[dict] = Field(default_factory=list, description="Anotações do PDF (comentários, destaques, etc)")
    links: list[dict] = Field(default_factory=list, description="Hyperlinks e referências cruzadas do PDF")


class DocumentJob(BaseModel):
    """Job assíncrono de processamento de documento."""

    id: str = Field(description="Identificador único do job")
    status: JobStatus = Field(JobStatus.processing, description="Status atual")
    result: ExtractionResult | None = Field(None, description="Resultado (disponível quando done)")
    error: str | None = Field(None, description="Mensagem de erro se houver")
    progress: int = Field(0, description="Progresso atual (páginas processadas)")
    total: int = Field(0, description="Total de páginas a processar")


class JobResult(BaseModel):
    """Resposta resumida de um job assíncrono."""

    id: str = Field(description="Identificador único do job")
    status: JobStatus = Field(..., description="Status atual")
    result: ExtractionResult | None = Field(None, description="Resultado (disponível quando done)")
    error: str | None = Field(None, description="Mensagem de erro se houver")
    progress: int = Field(0, description="Progresso atual")
    total: int = Field(0, description="Total de páginas")


class ExtractRequest(BaseModel):
    """Parâmetros para extração de PDF."""

    chunk_size: int = Field(default=2000, ge=100, le=32000, description="Tamanho alvo de cada chunk em caracteres")
    chunk_overlap: int = Field(default=200, ge=0, le=4000, description="Sobreposição entre chunks")
    use_ocr: bool = Field(False, description="Habilitar OCR para páginas escaneadas")
    password: str | None = Field(None, description="Senha para PDF protegido")
    format: FormatType = Field(FormatType.text, description="Formato de saída (text/markdown/semantic_html)")
    model: str = Field("gpt-4", description="Modelo usado para chunking (tiktoken)")
    extract_images: bool = Field(False, description="Extrair imagens do PDF")
    use_bcpd: bool = Field(False, description="Usar BCPD para segmentação")

    model_config = {"use_enum_values": True}


class ErrorResponse(BaseModel):
    """Resposta de erro padronizada."""

    error: str = Field(description="Mensagem de erro resumida")
    detail: str | None = Field(None, description="Detalhes adicionais do erro")
