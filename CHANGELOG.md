# Changelog

## [0.6.0] - 2026-06-29

### Added (ULTP Architecture)
- WassersteinTopologicalCost: custom BaseCost for ruptures BCPD using optimal transport (POT)
- Constituency guardrails: stanza-based constituency parse trees to prevent cutting inside syntactic constituents
- Coreference tracker: anaphora resolution to prevent "orphans of context" in Portuguese legal text
- SimCSE InfoNCE loss: contrastive learning regularization to mitigate embedding anisotropy
- Conformal prediction: crepes-based confidence intervals for segmentation decisions
- Spacecutter ordinal regression: hierarchical depth classification (chapter/section/paragraph)
- EBNF grammar support: outlines integration for structured LLM output via CFG-constrained generation
- Captum XAI: integrated gradients attribution for model explainability
- Segeval metrics: WindowDiff and P_k for segmentation quality evaluation
- ULTP extras group: `pip install -e ".[ultp]"` installs all ULTP dependencies

### Changed
- segmenter.py: integrated all ULTP modules — Wasserstein cost, SimCSE anisotropy correction,
  constituency guardrails, coreference resolution, confidence scoring, anisotropy reporting
- layout.py: enhanced multi-column detection with `detect_multicolumn_projection()`
- formatter.py: richer semantic HTML/markdown with footnotes, blockquotes, code blocks, references
- Extractor: annotations and links extraction from PDF via PyMuPDF

### Added (Infrastructure)
- Ruff + mypy in CI pipeline
- CHANGELOG versioning (0.6.0)
- Sentry error tracking (optional sentry-sdk)
- railway.staging.json for isolated staging environment
- Enriched API docs on all models and endpoints
- 10 integration tests with realistic legal PDF (8 pages, multi-column, lists, footnotes, references)
- ONNX-compatible classifier (app/engine/onnx_classifier.py)
- Hybrid OCR engine (app/engine/ocr_hybrid.py): EasyOCR → docTR → Tesseract chain
- 4 performance benchmarks with pytest-benchmark
- New optional dep groups: nlp, ml, conformal, ordinal, segeval, captum, outlines
- ULTP test suite: 18 tests covering Wasserstein cost, SimCSE, constituency, coreference, BCPD

## [0.5.0] - 2026-06-29

### Added
- PersistentVectorStore: SQLite-backed agent storage, survives deploys
- Tool calling loop: prompt-based tool selection (search, summarize, classify)
- Session memory: create/list/get sessions with persistent history
- Reranking: cross-encoder for search result reordering
- Query expansion: LLM generates 3 query variants, merges results
- Local embeddings fallback: sentence-transformers when API fails
- CI/CD pipeline: GitHub Actions runs 16 tests on every push
- Auto-cleanup: POST /agent/sessions/cleanup removes old sessions
- Vision description: LLM describes extracted images (OpenRouter vision)
- Ruff + mypy: lint and type checking in CI
- Sentry error tracking: optional exception monitoring
- Staging environment: railway.staging.json for isolated deploy
- API docs: enriched descriptions on all endpoints and models

### Changed
- Agent vector store: in-memory → SQLite (persistent)
- Agent chat: prompt-based tool calling replaces separate endpoints
- build-backend: setuptools.build_meta (was _legacy)
- Default OCR model: support for "easyocr" string parameter
- BCPD segmenter: optional SimCSE embeddings concatenation
- Layout metadata: stored in API response

### Fixed
- agent/load: `name 'text' not defined` error
- use_ocr: string parameters from Form not accepted (was bool-only)
- ocr_val: used before assignment in cache_params
- ExtractionResult: missing classified_count, layout_type, is_noise in response
- Session history: duplicate messages on chat (saved in both agent + routes)

## [0.4.0] - 2026-06-27

### Added
- Initial release with extract, chunk, agent, web UI
- EasyOCR engine support
- Layout analysis (centroid-based column detection)
- Builtin classifier (rule-based)
- BCPD segmenter (ruptures PELT + Wasserstein)
- Quality scoring (hyphenation, orphan_lines, line_quality)
- OpenRouter provider for agent
- Dockerfile with Railway deploy
