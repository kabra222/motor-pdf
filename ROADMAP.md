# ROADMAP - Motor PDF

## Visão Geral
Motor de backend para extração de texto de PDFs para consumo por LLMs.

**Versão**: 0.4.0 | **Repo**: https://github.com/kabra222/motor-pdf | **Produção**: https://motor-pdf-production.up.railway.app

---

## Funcionalidades

| Funcionalidade | Status | Planejado | Real | Obs |
|---|---|---|---|---|
| Extração básica (PyMuPDF) | ✅ Produção | OK | OK | 53 págs, 379 blocks classificados |
| Quality scoring real | ✅ Produção | overall ~0.91 | 0.90-0.91 | hyp, orphan, line_quality |
| Hyphenation repair | ✅ Produção | OK | OK | Distingue enclíticos de quebra silábica |
| Orphan merge (justificativa) | ✅ Produção | OK | OK | Artigos/prep/conj curtas |
| Header/footer stripping | ✅ Produção | Repetição ≥60% | OK | Estatístico por página |
| Layout analysis (colunas) | ✅ Produção | centroid-based | OK | metadata.layout no response |
| Classificador builtin | ✅ Produção | Rule-based (0 deps) | OK | heading/paragraph/short_text |
| Tabelas (Camelot+pdfplumber) | ✅ Produção | lattice→stream→fallback | OK | Até 9 tabelas por PDF |
| EasyOCR | ✅ Produção | `use_ocr=easyocr` | OK | 4 págs escaneadas detectadas |
| Agente IA (OpenRouter) | ✅ Produção | Nemotron 30B free | OK | RAG com chunks, streaming |
| BCPD Segmenter | ✅ Produção | PELT + Wasserstein | OK | Fallback heading-based |
| SimCSE embeddings (BCPD) | ✅ Produção | Opcional | OK | Se sentence-transformers instalado |
| Chunking semântico | ✅ Produção | token-aware + headings | OK | 98 chunks p/ capítulo |
| Cache SHA256 (SQLite) | ✅ Produção | OK | OK | Persistente |
| Jobs assíncronos (SSE) | ✅ Produção | OK | OK | Background processing |
| Rate limiter + API Key | ✅ Produção | 60 req/min | OK | Opcional |
| Web UI | ✅ Produção | Vanilla HTML+JS | OK | Aba Leitura + Quality bar |
| Descrição de imagens (LLM Vision) | ✅ Produção | OpenRouter vision | OK | Qwen VL 72B free, 20 img max |
| CI/CD Pipeline (GitHub Actions) | ✅ Configurado | Testes automáticos no push | OK | 16 testes, 5s |
| Sessões com memória | ✅ Produção | SQLite + `/agent/sessions` | OK | Persistente entre deploys |
| Tool calling | ✅ Produção | Prompt-based | OK | search/summarize/classify |
| Query expansion | ✅ Produção | 3 variações por pergunta | OK | Fusão + rerank |
| Embeddings locais | ✅ Produção | sentence-transformers fallback | OK | all-MiniLM-L6-v2 |
| Auto-cleanup sessões | ✅ Produção | `POST /agent/sessions/cleanup` | OK | Remove sessões com mais de 7 dias |
| Docker + Railway | ✅ Produção | Deploy automático | OK | `railway up` ou push GitHub |
| OCR PaddleOCR | 🟡 Instável | PaddleOCR + EasyOCR fallback | PaddleOCR pesado demais | Railway OOM |
| Unstructured classifier | ❌ Bloqueado | torch/transformers 1GB+ | Substituído por builtin | Pesado demais |

---

## Pipeline de Dados (Estado Atual)

```
PDF → [PyMuPDF] → Blocks → [Layout Analysis] → [Classifier Builtin]
    → [Header/Footer Strip] → [Hyphen Repair] → [Orphan Merge]
    → [Camelot/PDFPlumber Tables] → [Images (opcional)] → [LLM Vision desc]
    → [Quality Scoring] → [BCPD Segmenter] → [Semantic Chunker]
    → [PersistentVectorStore (SQLite)]
    → [Query Expansion] → [Search + Rerank] → [Context w/ metadata]
    → [Tool Calling Loop] → [LLM Agent (OpenRouter)]
    → [Session Memory (SQLite)]
```

---

## Melhorias no Agente (implementadas Jun/2026)

| Melhoria | Status | Detalhes |
|---|---|---|
| VectorStore Persistente | ✅ | SQLite, namespace-isolated, sobrevive a deploys |
| Tool Calling Real | ✅ | Prompt-based tool selection (funciona com qualquer LLM) |
| Reranking (Cross-Encoder) | ✅ | `cross-encoder/ms-marco-MiniLM-L6-v2` se disponível |
| Metadados na Resposta | ✅ | `[p.X, §heading]` no contexto fornecido ao LLM |
| Sessões com Memória | ✅ | SQLite, `/agent/sessions` CRUD, histórico persistente |
| Query Expansion | ✅ | LLM gera 3 variações da pergunta, busca fundida |
| Embeddings Locais | ✅ | Fallback `all-MiniLM-L6-v2` quando API de embedding falha |

---

## Próximos Passos Prioritários

1. Melhorar detecção de multi-coluna com PDFs reais (ex: artigos científicos)
2. Suporte a formatação richer (listas aninhadas, notas de rodapé)

---

## Comportamento Esperado vs Real

### Extração
- **Esperado**: Texto limpo, sem ruído, hífens reparados, órfãos mesclados
- **Real**: ✅ Coincide. `_repair_hyphenation` + `_merge_justified_orphans` + header/footer strip

### Qualidade
- **Esperado**: Score realista baseado em métricas observáveis
- **Real**: ✅ hyp=1.0, orphan=0.97, line_quality=0.66, overall=0.91

### Classificação
- **Esperado**: Cada bloco textual classificado semanticamente (heading/paragraph/etc)
- **Real**: ✅ layout_type em ~90% dos blocks textuais. Noise blocks filtrados.

### Layout
- **Esperado**: Detecção de colunas, headers, footers
- **Real**: ✅ centroid-based column detection. PDFs atuais são single-column.

### OCR
- **Esperado**: PaddleOCR com fallback EasyOCR
- **Real**: 🔶 EasyOCR funciona (`use_ocr=easyocr`). PaddleOCR pula por peso (Railway 512MB).

### Agente
- **Esperado**: Chat RAG com contexto do PDF via OpenRouter
- **Real**: ✅ Nemotron 30B free, chunks indexados, citações por score. Streaming funcional.

### Imagens
- **Esperado**: Extrair imagens + descrição via LLM vision
- **Real**: ✅ 20 imagens extraídas por PDF, descrições via Qwen VL 72B free.

---

## Variáveis de Ambiente (Produção)

| Variável | Valor | Função |
|---|---|---|
| `MOTOR_PDF_LLM_PROVIDER` | `openrouter` | Provider do agente |
| `OPENROUTER_API_KEY` | configurado | API key OpenRouter |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-nano-30b-a3b:free` | Modelo chat |
| `OPENROUTER_VISION_MODEL` | `qwen/qwen-2-vl-72b-instruct:free` | Modelo visão |
| `PORT` | (automático) | Porta Railway |
