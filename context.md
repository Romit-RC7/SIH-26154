# SIH-26154 — Semantic Document Processing System: Technical Context

> **Project**: AI-Powered Content Transformation Platform (SIH-26154)
> **Phase**: Milestone 1 Complete + Knowledge & Retrieval Layer (Phase 3) Complete
> **Updated**: September 6, 2026 (Session 8)
> **Repository**: [https://github.com/Romit-RC7/SIH-26154.git](https://github.com/Romit-RC7/SIH-26154.git)
> **Workspace**: `Ai-Services`

---

## 1. System Overview & Problem Statement

The **SIH-26154 AI-Powered Content Transformation Platform** transforms unstructured, multi-page, heterogeneous documents (**PDF**, **DOCX**, **PPTX**, and standalone **Images** such as PNG, JPG, JPEG, WEBP, BMP, TIFF) into a canonical, structured **Semantic Document JSON**.

Rather than relying on naive raw text extraction, the platform treats documents as multi-modal composites containing:

- Hierarchical text (titles, headings, paragraphs, headers, footers)
- Structured tables (cell matrices, header rows, markdown & HTML tables)
- Visual elements (figures, charts, diagrams, images)
- Relational metadata (bounding boxes, reading order, captions, cross-element references)

### The Canonical Contract

Every downstream AI module (BGE Embedding Engine, pgvector semantic search, Qwen2.5-VL visual enricher, UniChart reasoning, Knowledge Graph constructor, and Multi-Format Output Generators) interacts exclusively with the **Unified Semantic Document JSON** schema, rather than dealing directly with raw document bytes.

```
PDF / DOCX / PPTX / Image Upload
       ↓
Document Parsers (PyMuPDF / python-docx / python-pptx / Pillow)
       ↓
Offline Layout + OCR Analysis (Primary: PP-StructureV3 | Backup: PyMuPDF Fallback | Direct Structural Parsers)
       ↓
Intermediate Region Document (stable element IDs & disk-persisted crops in uploads/extracted/)
       ↓
Formula + Chart Recognition, then Qwen2.5-VL Image/Figure Recognition
       ↓
Qwen3-4B Structured Fusion
       ↓
Semantic Fusion Engine (reading order sort, caption-to-visual linking)
       ↓
Semantic Document Builder
       ↓
Canonical Semantic Document JSON (Pydantic Schema)
       ↓
PostgreSQL 16 + pgvector Storage (JSONB document + relational elements)
```

### Multi-Format Ingestion Strategy & Image Persistence

1. **PDF Ingestion (`pdf_parser.py` + `pp_structure.py` / `fallback_analyzer.py`)**:
   - PyMuPDF rasterizes pages into numpy image arrays for PP-StructureV3 OCR/layout detection or runs heuristic layout analysis in fallback mode.
2. **Word Ingestion (`docx_parser.py`)**:
   - `python-docx` parses hierarchical headings, paragraphs, native tables, and extracts embedded document images.
3. **PowerPoint Ingestion (`ppt_parser.py`)**:
   - `python-pptx` parses presentation slides as logical pages (`ParsedPage`).
   - Extracts text shapes with calculated slide bounding boxes and reading order.
   - Formats native PPT tables into dual representations (GitHub Flavored Markdown + raw HTML + row matrix data).
   - Extracts embedded image shapes (`shape_type == 13`) as converted RGB PIL Images.
4. **Standalone Image Ingestion (`image_parser.py`)**:
   - Handles standalone image formats (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`).
   - Maps the file to a single-page document with an image element spanning full dimensions.
5. **Visual Asset Persistence (`storage_service.py` -> `save_image_crop`)**:
   - Both embedded PPTX images and standalone uploaded images are saved to disk under `uploads/extracted/{document_id}/{element_id}.png`.
   - The relative file path is attached to `elem.attributes["saved_image_path"]`, ready for downstream visual intelligence models (Qwen2.5-VL and UniChart).

---

## 2. Staged Offline Model Weights (`models/`)

All required AI models have been downloaded, verified, and staged locally in dedicated subdirectories under `models/` for completely offline execution:

| Subfolder                       | Model Identifier                      | Purpose / Task                                                         | Format / Key Files                                                              |
| :------------------------------ | :------------------------------------ | :--------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| **`models/pp_structure_v3/`**   | `PaddleOCR / PP-Structure`            | Layout parsing, table structure (SLANet), Text Detection & Recognition | `layout/` (`model.pdiparams`), `table/` (`inference.pdiparams`), `det/`, `rec/` |
| **`models/unichart_base_960/`** | `ahmed-masry/unichart-base-960`       | Specialized Chart Comprehension, Data Extraction & Visual Reasoning    | `pytorch_model.bin` (809 MB), `tokenizer.json`, `config.json`                   |
| **`models/qwen2.5_vl_3b_q4/`**  | `unsloth/Qwen2.5-VL-3B-Instruct-GGUF` | Multimodal Vision-Language for figures, diagrams & flowcharts          | `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` (1.92 GB) + `mmproj-F16.gguf` (1.33 GB)    |
| **`models/qwen3_4b_q4/`**       | `unsloth/Qwen3-4B-GGUF`               | Lightweight High-Efficiency Reasoning & Fast Synthesis LLM             | `Qwen3-4B-Q4_K_M.gguf` (2.49 GB single quant)                                   |
| **`models/qwen3_8b_q4/`**       | `unsloth/Qwen3-8B-GGUF`               | Core Deep-Reasoning & Multi-Format Content Generation LLM              | `Qwen3-8B-Q4_K_M.gguf` (5.02 GB single quant)                                   |
| **`models/bge_small_en_v1.5/`** | `BAAI/bge-small-en-v1.5`              | Dense Semantic Vector Embeddings for pgvector search                   | `model.safetensors` (133 MB), ONNX runtime models, tokenizer configs            |

### Downloader Utility (`scripts/download_models.py`)

- Python script powered by `huggingface_hub` native API (with `hf` CLI fallback).
- Multi-mirror download with exponential backoff retry for PaddleOCR models.
- Selective downloads supported via `--select` flag.

PP-Structure model groups are stored under `models/pp_structure_v3/` as local
packages for layout, OCR, table recognition, formulas, charts, and table-cell
submodels. The current PP-Structure stage loads layout, OCR, and table models
together. Formula and chart services then share a later model stage, followed by
the Qwen2.5-VL image/figure stage. Runtime services must never download weights.

### Staged Recognition Lifecycle

1. PP-Structure identifies regions, performs OCR, recognizes tables, and produces crops.
2. Formula and chart services process only their matching regions in one shared stage.
3. Qwen2.5-VL processes figures and images in a separate stage.
4. Qwen3-8B is reserved for final structured fusion and summarization.

Each stage updates stable element IDs in an intermediate recognition document;
the Semantic Fusion Engine and Semantic Document Builder remain the final
normalization and validation boundary.

---

## 3. PP-StructureV3 Offline Integration

### Architecture & Local Weight Injection

The analyzer in `backend/app/processors/pp_structure.py` automatically inspects `models/pp_structure_v3/` and injects local weight directories directly into the unified engine, requiring zero internet connectivity at runtime:

```python
models_root = settings.PP_STRUCTURE_MODEL_DIR
layout_dir = models_root / "layout"
table_dir = models_root / "table"
det_dir = models_root / "det"
rec_dir = models_root / "rec"

kwargs = {
    "table": True,
    "ocr": True,
    "layout": True,
    "lang": "en",
    "show_log": False,
    "recovery": True,
}

if layout_dir.exists(): kwargs["layout_model_dir"] = str(layout_dir)
if table_dir.exists():  kwargs["table_model_dir"] = str(table_dir)
if det_dir.exists():    kwargs["det_model_dir"] = str(det_dir)
if rec_dir.exists():    kwargs["rec_model_dir"] = str(rec_dir)

self.engine = PPStructure(**kwargs)
```

### Memory Efficiency & Unified Inference

- Rather than instantiating separate OCR and Layout classes, all components run in a **single unified `PPStructure` call** (`self.engine(img_np)`).
- PaddlePaddle shares intermediate memory buffers across Layout $\to$ Table $\to$ OCR stages, keeping RAM usage to $\approx 600\text{--}800\text{ MB}$.

### System Diagnostics & Model Health Inspection (`system_diagnostics.py` & `health.py`)

A dedicated `SystemDiagnostics` service (`backend/app/services/system_diagnostics.py`) inspects filesystem weights and engine initializations without loading heavy models into RAM:

- **Model Weight Presence**: Confirms the physical presence of staged weights in `models/` by validating directories and non-empty file contents:
  - `pp_structure`: Engine initialized and callable.
  - `layout_model`, `table_model`, `ocr_det_model`, `ocr_rec_model`: Component weights in `models/pp_structure_v3/`.
  - `bge_small_en_v1.5`, `qwen2.5_vl_3b_q4`, `qwen3_4b_q4`, `qwen3_8b_q4`, `unichart_base_960`: Offline weights in `models/`.
- **Engine Readiness**: Evaluates `pp_structure_initialized` and `fallback_available`.
- **Storage Paths**: Confirms readiness of `raw_uploads` and `extracted_uploads` directories.
- **Enriched Health Endpoint (`GET /api/v1/health`)**:
  - Validates PostgreSQL connectivity asynchronously (`SELECT 1`).
  - Reports overall health (`online` or `degraded` if DB is unavailable).
  - Exposes granular boolean flags (`true` / `false`) for every model, engine, and storage directory to facilitate deployment verification and CI smoke tests.

---

## 4. Architecture & Data Contracts

### 1. `SemanticDocument` Schema (Pydantic v2)

File: `backend/app/schemas/semantic_document.py`

```typescript
SemanticDocument {
  version: "1.0.0",
  document_id: UUID,
  metadata: {
    file_name: string,
    file_size: int,
    mime_type: string,
    page_count: int,
    title?: string,
    sha256?: string,
    created_at: ISO8601,
    extra: {}
  },
  elements: [
    {
      id: string,                 // e.g. "elem_ff7e2f88_1_3"
      type: "text" | "table" | "image" | "chart" | "figure",
      page: int,
      bbox?: [x1, y1, x2, y2],    // normalized or points
      content: {
        text?: string,
        markdown?: string,        // clean GFM table
        html?: string,            // raw <table>...</table>
        image_path?: string,      // relative storage path
        confidence?: float,
        reading_order?: int,
        caption?: string,
        table_structure?: {},
        raw_attributes?: {}       // role (header, footer, title), pp_type, layout_index
      }
    }
  ],
  entities: [],                   // Phase 3: Knowledge Graph entities
  claims: [],                     // Phase 3: Factual propositions
  relationships: [],              // Phase 3: Entity cross-element triples
  sources: []
}
```

### 2. Database Models (SQLAlchemy 2.0 Async + PostgreSQL + pgvector)

Files: `backend/app/models/`

- **`Document` (`documents` table)**: Stores document metadata, SHA-256 hash, raw file path, page count, status, and full `semantic_json` JSONB payload.
- **`DocumentElement` (`document_elements` table)**: Relational element rows with foreign keys to `Document`, containing bounding box coordinates, element type, text, markdown, image path, reading order, and `Vector(768)` column for pgvector embeddings.
- **`ProcessingJob` (`processing_jobs` table)**: Asynchronous processing job tracker (`status`, `stage`, `error_message`, duration telemetry).

---

## 5. Codebase Inventory

```
Ai-Services/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── deps.py                       # DB & Service dependencies
│   │   │   └── v1/
│   │   │       ├── api.py                    # Router aggregation
│   │   │       └── endpoints/
│   │   │           ├── documents.py          # Upload, status, semantic retrieval, list
│   │   │           └── health.py             # Health check & multi-model diagnostic status
│   │   ├── core/
│   │   │   ├── config.py                     # Pydantic BaseSettings, allowed extensions, model paths
│   │   │   └── logging.py                    # Structured logging configuration
│   │   ├── database/
│   │   │   ├── base.py                       # Declarative Base
│   │   │   └── session.py                    # Async engine & session factory
│   │   ├── models/
│   │   │   ├── document.py                   # Document SQLAlchemy model
│   │   │   ├── document_element.py           # DocumentElement model (with pgvector)
│   │   │   └── processing_job.py             # ProcessingJob model
│   │   ├── processors/
│   │   │   ├── base.py                       # RawDocumentElement dataclasses & ABC
│   │   │   ├── pdf_parser.py                 # PyMuPDF document rasterizer
│   │   │   ├── docx_parser.py                # python-docx document parser
│   │   │   ├── ppt_parser.py                 # python-pptx presentation parser (slides, tables, images)
│   │   │   ├── image_parser.py               # Standalone image parser (PNG, JPG, WEBP, BMP, TIFF)
│   │   │   ├── pp_structure.py               # PP-Structure integration with local model paths
│   │   │   ├── fallback_analyzer.py          # Rule-based PyMuPDF layout fallback
│   │   │   └── extractor.py                  # Multi-modal extraction coordinator
│   │   ├── schemas/
│   │   │   ├── document.py                   # API request/response schemas
│   │   │   ├── processing_job.py             # Job status schemas
│   │   │   └── semantic_document.py          # Unified System Contract schema
│   │   ├── services/
│   │   │   ├── system_diagnostics.py         # Offline model presence & system health evaluator
│   │   │   ├── pipeline_service.py           # Full async processing pipeline coordinator
│   │   │   ├── storage_service.py            # Disk storage & image crop persistence (save_image_crop)
│   │   │   ├── semantic_fusion.py            # Reading order sorting & caption linking
│   │   │   └── semantic_builder.py           # SemanticDocument JSON builder
│   │   ├── utils/
│   │   │   └── file_utils.py                 # SHA-256, mime detection, validation
│   │   └── main.py                           # FastAPI application factory & lifespan
│   ├── scripts/
│   │   └── run_benchmarks_and_validation.py  # Benchmark suite (latency, memory, throughput)
│   └── tests/
│       ├── conftest.py                       # Test fixtures (async test engine, client)
│       ├── test_api.py                       # REST API endpoint tests
│       ├── test_api_pagination.py            # Document list pagination tests
│       ├── test_database.py                  # SQLAlchemy async CRUD tests
│       ├── test_failure_scenarios.py         # Corrupt files, unsupported formats
│       ├── test_file_utils.py                # Hashing and validation tests
│       ├── test_parsers.py                   # PDF and DOCX parsing unit tests
│       ├── test_parsers_extended.py          # Multi-column & table parsing tests
│       ├── test_pipeline_integration.py      # End-to-end pipeline execution tests
│       ├── test_semantic_document.py         # System contract schema validation
│       ├── test_storage_service.py           # Image cropping & disk persistence tests
│       └── test_structure_analyzers.py       # PPStructure, fallback, & failover tests
├── docker/
│   ├── Dockerfile                            # Production multi-stage Docker build
│   ├── docker-compose.yml                    # sih_backend + sih_postgres (pgvector)
│   └── .dockerignore                         # Docker build exclusions (models/ excluded)
├── models/                                   # ⭐ Offline Model Weights Directory (.gitignore)
│   ├── pp_structure_v3/                      # Layout, Table, Det, Rec
│   ├── unichart_base_960/                    # Chart Comprehension
│   ├── qwen2.5_vl_3b_q4/                     # Multimodal Vision (GGUF + mmproj)
│   ├── qwen3_4b_q4/                          # Fast Reasoning LLM (GGUF)
│   ├── qwen3_8b_q4/                          # Deep Reasoning LLM (GGUF)
│   └── bge_small_en_v1.5/                    # Vector Embeddings
├── scripts/
│   └── download_models.py                    # Resilient HF + Paddle model downloader
├── uploads/
│   ├── raw/                                  # Uploaded original files (.gitkeep)
│   └── extracted/                            # Cropped visual regions (.gitkeep)
├── .env.example                              # Environment configuration template
├── .gitignore                                # Comprehensive Python, runtime & models/ ignore
├── requirements.txt                          # Python dependencies (includes python-pptx)
├── PHASE1_IMPLEMENTATION_SUMMARY.md          # Architectural summary
└── PHASE1_TEST_REPORT.md                     # QA report (41 tests passing, 84% coverage)
```

---

## 6. Verification & Test Status

- **Unit & Integration Tests**: 41/41 passing (`backend/tests/`).
- **Code Coverage**: 84% statement coverage across core pipeline modules.
- **Git Hygiene**: `models/` directory, `*.gguf`, `*.safetensors`, and `*.pdiparams` are strictly ignored by `.gitignore` and `docker/.dockerignore`.

---

## 6. Knowledge & Retrieval Layer (Phase 3) — Complete

> See `knowledge_retrieval.md` for the full technical reference.

### Architecture

```
Semantic Document JSON
    → Text Cleaner (NFKC, whitespace, dehyphenation)
    → Semantic Chunker (text/table/visual-aware, context prefix)
    → BGE Small EN v1.5 (384-dim dense vectors, transformers backend)
    → PostgreSQL document_chunks + pgvector
    ← Cosine Similarity Retrieval (Python, top_k ranked)
    → Knowledge Engine (Qwen3-4B GGUF or deterministic fallback)
    → KnowledgePackage (self-contained orchestrator input)
```

### Key Files Added

| File | Purpose |
|------|---------|
| `backend/app/services/embedding/text_cleaner.py` | Unicode normalization, whitespace collapsing |
| `backend/app/services/embedding/chunker.py` | Structure-aware chunker with context prefix |
| `backend/app/services/embedding/embedding_service.py` | Clean → chunk → embed → persist orchestrator |
| `backend/app/services/model_initializer/bge_initializer.py` | BGE lazy loader (transformers backend) |
| `backend/app/services/retrieval_service.py` | Cosine similarity search over pgvector |
| `backend/app/services/knowledge_engine.py` | Knowledge assembly + Qwen3-4B / deterministic |
| `backend/app/models/document_chunk.py` | `document_chunks` table + custom `Vector(384)` type |
| `backend/app/schemas/intent.py` | `IntentAndPersonalization` schema + enums |
| `backend/app/schemas/knowledge_package.py` | `KnowledgePackage` output contract schema |
| `backend/app/api/v1/endpoints/knowledge.py` | 5 REST endpoints (embed / search / assemble) |

### KnowledgePackage — What It Contains

The `KnowledgePackage` is the **sole input** to the Content Orchestrator (Phase 4). It contains:
- `intent` — user's `output_type`, `audience`, `tone`, `objective`, `focus_keywords`
- `retrieved_evidence[]` — top-k semantically retrieved chunks with cosine scores
- `entities[]` — named entities (Qwen3-4B or heuristic NER)
- `claims[]` — factual propositions with source element citations
- `key_metrics[]` — numeric values extracted with surrounding context
- `tables[]` — GFM markdown tables from the document
- `visual_insights[]` — figure/chart descriptions and image paths
- `strategy` — `headline_hook`, `key_themes`, `suggested_structure` (per output format), `recommended_cta`, `tone_guidelines`
- `orchestrator_prompt_context` — pre-compiled dense markdown block ready for prompt injection

### Architecture Decision

The Content Orchestrator will have **separate prompt templates per output format** (LinkedIn, Twitter, Executive Summary, Infographic, etc.). All templates consume the same `KnowledgePackage`. Format-specific constraints (word limits, character limits, section counts) are the orchestrator's responsibility — the knowledge layer provides **what to say**, the orchestrator determines **how to format it**.

---

## 7. Codebase Inventory (Updated Session 8)

```
Ai-Services/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── deps.py
│   │   │   └── v1/
│   │   │       ├── api.py
│   │   │       └── endpoints/
│   │   │           ├── documents.py        # Upload, status, semantic retrieval
│   │   │           ├── health.py           # Health check & model diagnostics
│   │   │           └── knowledge.py        # ★ NEW: embed / search / assemble endpoints
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── logging.py
│   │   ├── database/
│   │   │   ├── base.py
│   │   │   └── session.py                  # init_db() creates vector extension + document_chunks
│   │   ├── models/
│   │   │   ├── document.py                 # + chunks relationship
│   │   │   ├── document_chunk.py           # ★ NEW: DocumentChunk + custom Vector(384)
│   │   │   ├── document_element.py
│   │   │   └── processing_job.py
│   │   ├── processors/
│   │   │   ├── base.py
│   │   │   ├── pdf_parser.py
│   │   │   ├── docx_parser.py
│   │   │   ├── ppt_parser.py
│   │   │   ├── image_parser.py
│   │   │   ├── pp_structure.py
│   │   │   ├── fallback_analyzer.py
│   │   │   └── extractor.py
│   │   ├── schemas/
│   │   │   ├── document.py
│   │   │   ├── processing_job.py
│   │   │   ├── semantic_document.py        # Canonical contract
│   │   │   ├── intent.py                   # ★ NEW: IntentAndPersonalization + enums
│   │   │   └── knowledge_package.py        # ★ NEW: KnowledgePackage output contract
│   │   ├── services/
│   │   │   ├── embedding/                  # ★ NEW package
│   │   │   │   ├── __init__.py
│   │   │   │   ├── text_cleaner.py
│   │   │   │   ├── chunker.py
│   │   │   │   └── embedding_service.py
│   │   │   ├── model_initializer/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── bge_initializer.py      # ★ NEW
│   │   │   │   ├── pp_structure_initializer.py
│   │   │   │   ├── qwen_initializers.py
│   │   │   │   └── unichart_initializer.py
│   │   │   ├── knowledge_engine.py         # ★ NEW: Qwen3-4B + deterministic assembly
│   │   │   ├── retrieval_service.py        # ★ NEW: pgvector cosine search
│   │   │   ├── system_diagnostics.py
│   │   │   ├── pipeline_service.py
│   │   │   ├── storage_service.py
│   │   │   ├── semantic_fusion.py
│   │   │   └── semantic_builder.py
│   │   └── main.py
│   └── tests/                              # 66+ tests passing
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── models/
│   ├── pp_structure_v3/
│   ├── unichart_base_960/
│   ├── qwen2.5_vl_3b_q4/
│   ├── qwen3_4b_q4/
│   ├── qwen3_8b_q4/
│   └── bge_small_en_v1.5/
├── scripts/
│   └── download_models.py
├── uploads/
│   ├── raw/
│   └── extracted/
├── requirements.txt                        # includes pgvector>=0.3.0
├── context.md                              # THIS FILE
├── history.md                              # Session changelog
└── knowledge_retrieval.md                  # ★ NEW: Full Phase 3 reference doc
```

---

## 8. Next: Phase 4 — Content Orchestrator

**Input**: `KnowledgePackage` from `/api/v1/knowledge/assemble/{document_id}`

**Architecture**:
- One format-specific prompt template per `output_type`
- Main model: Qwen3-8B Q4 GGUF (`models/qwen3_8b_q4/`) for deep reasoning + generation
- Each template injects `orchestrator_prompt_context` + format constraints into Qwen3-8B prompt
- Output: structured generated content in the target format

**Supported output formats** (each needs its own prompt template):
- `linkedin_post` — 150–300 words, hook + insight + CTA + hashtags
- `twitter_thread` — 5–8 tweets, each ≤ 280 chars, numbered thread
- `executive_summary` — formal multi-section report with headers
- `presentation_deck` — slide-by-slide structured brief
- `infographic_brief` — visual-first, stats + minimal prose
- `video_script` — scene-structured narrative with visual cues

## 7. Next Milestone: Phase 2 Roadmap

With Phase 1 document intelligence, offline model staging, and PP-Structure integration complete, the platform is ready for Phase 2:

1. **UniChart + Qwen2.5-VL Visual Intelligence Pipeline**:
   - Ingest cropped figures and charts from `uploads/extracted/`.
   - Run `unichart_base_960` for tabular data extraction from plots/charts.
   - Run `Qwen2.5-VL-3B Q4` for diagram explanation, trend analysis, and descriptive takeaways.
2. **Dense Vector Embeddings & pgvector**:
   - Run `bge_small_en_v1.5` to generate 384/768-dim embeddings for all text and table chunks.
   - Populate pgvector tables for sub-second semantic retrieval.
3. **Intent & Content Orchestration**:
   - Synthesize multi-format outputs (Executive Summary, LinkedIn, PPT Slides, Infographics) using `Qwen3-4B` and `Qwen3-8B`.

---

## 8. Current Video Processing Extension

The current implementation accepts MP4 (preferred), WebM, and MOV videos up to 100 MB and two minutes. `video_parser.py` uses local FFmpeg/FFprobe to extract a 16 kHz mono WAV artifact and sample one visual frame every 10 seconds (six per minute), capped at 1280 pixels wide. `FasterWhisperInitializer` loads `models/faster_whisper_small/` only for extracted audio, and `speech_service.py` persists the transcript onto a semantic text element before unloading the model. Sampled frames flow to the existing Qwen2.5-VL stage. Runtime is offline after Docker build and one-time model staging:

```bash
python scripts/download_models.py --select faster_whisper
```
