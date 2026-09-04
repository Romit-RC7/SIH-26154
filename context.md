# SIH-26154 — Semantic Document Processing System: Technical Context

> **Project**: AI-Powered Content Transformation Platform (SIH-26154)  
> **Phase**: Milestone 1 — Foundational Semantic Document Processing System & Model Weights Staging  
> **Updated**: September 5, 2026  
> **Repository**: [https://github.com/Romit-RC7/SIH-26154.git](https://github.com/Romit-RC7/SIH-26154.git)  
> **Workspace**: `Ai-Services`

---

## 1. System Overview & Problem Statement

The **SIH-26154 AI-Powered Content Transformation Platform** transforms unstructured, multi-page, heterogeneous documents (**PDF** and **DOCX**) into a canonical, structured **Semantic Document JSON**.

Rather than relying on naive raw text extraction, the platform treats documents as multi-modal composites containing:

- Hierarchical text (titles, headings, paragraphs, headers, footers)
- Structured tables (cell matrices, header rows, markdown & HTML tables)
- Visual elements (figures, charts, diagrams, images)
- Relational metadata (bounding boxes, reading order, captions, cross-element references)

### The Canonical Contract

Every downstream AI module (BGE Embedding Engine, pgvector semantic search, Qwen2.5-VL visual enricher, UniChart reasoning, Knowledge Graph constructor, and Multi-Format Output Generators) interacts exclusively with the **Unified Semantic Document JSON** schema, rather than dealing directly with raw document bytes.

```
PDF / DOCX Upload
       ↓
Document Parsers (PyMuPDF / python-docx)
       ↓
Layout & Structure Analysis (Primary: Offline PP-StructureV3 | Backup: PyMuPDF Fallback)
       ↓
Multi-Modal Visual Extraction (disk persistence of cropped tables/images/charts)
       ↓
Semantic Fusion Engine (reading order sort, caption-to-visual linking)
       ↓
Semantic Document Builder
       ↓
Canonical Semantic Document JSON (Pydantic Schema)
       ↓
PostgreSQL 16 + pgvector Storage (JSONB document + relational elements)
```

---

## 2. Staged Offline Model Weights (`models/`)

All required AI models have been downloaded, verified, and staged locally in dedicated subdirectories under `models/` for completely offline execution:

| Subfolder | Model Identifier | Purpose / Task | Format / Key Files |
| :--- | :--- | :--- | :--- |
| **`models/pp_structure_v3/`** | `PaddleOCR / PP-Structure` | Layout parsing, table structure (SLANet), Text Detection & Recognition | `layout/` (`model.pdiparams`), `table/` (`inference.pdiparams`), `det/`, `rec/` |
| **`models/unichart_base_960/`** | `ahmed-masry/unichart-base-960` | Specialized Chart Comprehension, Data Extraction & Visual Reasoning | `pytorch_model.bin` (809 MB), `tokenizer.json`, `config.json` |
| **`models/qwen2.5_vl_3b_q4/`** | `unsloth/Qwen2.5-VL-3B-Instruct-GGUF` | Multimodal Vision-Language for figures, diagrams & flowcharts | `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` (1.92 GB) + `mmproj-F16.gguf` (1.33 GB) |
| **`models/qwen3_4b_q4/`** | `unsloth/Qwen3-4B-GGUF` | Lightweight High-Efficiency Reasoning & Fast Synthesis LLM | `Qwen3-4B-Q4_K_M.gguf` (2.49 GB single quant) |
| **`models/qwen3_8b_q4/`** | `unsloth/Qwen3-8B-GGUF` | Core Deep-Reasoning & Multi-Format Content Generation LLM | `Qwen3-8B-Q4_K_M.gguf` (5.02 GB single quant) |
| **`models/bge_small_en_v1.5/`** | `BAAI/bge-small-en-v1.5` | Dense Semantic Vector Embeddings for pgvector search | `model.safetensors` (133 MB), ONNX runtime models, tokenizer configs |

### Downloader Utility (`scripts/download_models.py`)
- Python script powered by `huggingface_hub` native API (with `hf` CLI fallback).
- Multi-mirror download with exponential backoff retry for PaddleOCR models.
- Selective downloads supported via `--select` flag.

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
│   │   │           └── health.py             # Health check & engine status
│   │   ├── core/
│   │   │   ├── config.py                     # Pydantic BaseSettings, model paths
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
│   │   │   ├── pp_structure.py               # PP-Structure integration with local model paths
│   │   │   ├── fallback_analyzer.py          # Rule-based PyMuPDF layout fallback
│   │   │   └── extractor.py                  # Multi-modal extraction coordinator
│   │   ├── schemas/
│   │   │   ├── document.py                   # API request/response schemas
│   │   │   ├── processing_job.py             # Job status schemas
│   │   │   └── semantic_document.py          # Unified System Contract schema
│   │   ├── services/
│   │   │   ├── pipeline_service.py           # Full async processing pipeline coordinator
│   │   │   ├── storage_service.py            # Disk storage for raw & cropped images
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
├── requirements.txt                          # Python dependencies definition
├── PHASE1_IMPLEMENTATION_SUMMARY.md          # Architectural summary
└── PHASE1_TEST_REPORT.md                     # QA report (41 tests passing, 84% coverage)
```

---

## 6. Verification & Test Status

- **Unit & Integration Tests**: 41/41 passing (`backend/tests/`).
- **Code Coverage**: 84% statement coverage across core pipeline modules.
- **Git Hygiene**: `models/` directory, `*.gguf`, `*.safetensors`, and `*.pdiparams` are strictly ignored by `.gitignore` and `docker/.dockerignore`.

---

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
