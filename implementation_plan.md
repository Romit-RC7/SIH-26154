# Implementation Plan - Phase 1: Foundational Semantic Document Processing System

Phase 1 of the AI-Powered Content Transformation Platform for Smart India Hackathon (SIH-26154) builds the core foundational backend service. This service ingests PDF documents, processes them via a structured document analysis pipeline (integrating PP-StructureV3), extracts text, tables, figures, charts, and layout metadata, and compiles them into a unified, extensible **Semantic Document JSON** stored in PostgreSQL.

Future AI modules (Qwen2.5-VL, pgvector embeddings, Knowledge Engine, Content Orchestrator, LLM Generation) are decoupled from raw documents and will consume this standardized semantic schema.

---

## User Review Required

> [!IMPORTANT]
> **PP-StructureV3 Engine & Local Fallback Strategy**:
> PaddlePaddle / PP-Structure requires heavy system dependencies (OpenCV, libgomp, CUDA/C++ runtime). We will architect an extensible `BaseStructureAnalyzer` interface with:
> 1. `PPStructureAnalyzer` (Production implementation calling `paddleocr.PPStructure` / layout parser).
> 2. `RuleBasedStructureAnalyzer` (PyMuPDF / pdfplumber fallback engine for lightweight local testing and fallback execution if Paddle weights are downloading or running in environments without full Paddle binaries).
> A configuration flag `DOC_ANALYZER_ENGINE` (`"pp_structure"` or `"fallback"`) enables switching seamlessly.

> [!NOTE]
> **Asynchronous vs. Background Processing**:
> Document OCR and table recognition can take anywhere from a few seconds to a minute per multi-page PDF. The `/documents/upload` endpoint will immediately accept the file, persist the record with `status="PROCESSING"`, and dispatch the pipeline via FastAPI `BackgroundTasks` (or async worker pipeline), updating a `ProcessingJob` tracking record throughout each stage.

---

## Proposed System Architecture

```mermaid
flowchart TD
    Client[Client / Frontend] -->|POST /documents/upload| API[FastAPI Upload Endpoint]
    API -->|Save Raw PDF| LocalStorage[(Local Filesystem Storage)]
    API -->|Create Record| DB[(PostgreSQL Document & Job)]
    API -->|Enqueue Task| Pipeline[Document Processing Pipeline]

    subgraph "Document Processing Pipeline"
        Pipeline -->|1. Render Pages| PDFParser[PDF Parser - PyMuPDF]
        PDFParser -->|Page Images & Metadata| StructureEngine{PP-StructureV3 / Analyzer}
        StructureEngine -->|Layout Detection + OCR + Tables| ExtractionLayer[Extraction & Normalization Layer]
        ExtractionLayer -->|Crop & Save Visuals| FigureStorage[(Extracted Figures/Tables Storage)]
        ExtractionLayer -->|Formula + Chart + Image Recognition| Recognition[Staged Recognition Services]
        Recognition -->|Enriched Raw Elements| Builder[Semantic Document Builder]
        Builder -->|Schema Validation| PydanticSchema[Semantic Document Schema]
    end

    Builder -->|Update Semantic JSON & Status| DB
    DB -->|Populate Elements| DocElement[(DocumentElement Table)]

    Client -->|GET /documents/{id}/semantic| API
    API -->|Retrieve JSON| DB
```

---

## Proposed Changes

### 1. Project Directory Structure
We will establish a modular, production-ready project layout within `backend/`:

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application factory and lifespan events
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py                 # Dependency injection (DB session, services)
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── api.py              # Router aggregation
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── documents.py    # POST /upload, GET /{id}, GET /{id}/semantic, GET /
│   │           └── health.py       # GET /health
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Pydantic BaseSettings (DB, paths, timeouts)
│   │   └── logging.py              # Structured logging configuration
│   ├── database/
│   │   ├── __init__.py
│   │   ├── base.py                 # SQLAlchemy DeclarativeBase
│   │   └── session.py              # Async / Sync SQLAlchemy session engine
│   ├── models/
│   │   ├── __init__.py
│   │   ├── document.py             # Document model (metadata, status, semantic JSON)
│   │   ├── document_element.py     # DocumentElement model (relational elements)
│   │   └── processing_job.py       # ProcessingJob model (step tracking, errors)
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── document.py             # Request/Response schemas for Document
│   │   ├── processing_job.py       # Processing job schemas
│   │   └── semantic_document.py    # Standardized Semantic Document Pydantic Schema
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract BaseStructureAnalyzer
│   │   ├── pdf_parser.py           # Page rasterization & text layout parser (PyMuPDF)
│   │   ├── pp_structure.py         # PP-StructureV3 wrapper & parser
│   │   ├── fallback_analyzer.py    # PyMuPDF / rule-based fallback analyzer
│   │   └── extractor.py            # Normalization of layout boxes, text, tables, figures
│   ├── services/
│   │   ├── __init__.py
│   │   ├── storage_service.py      # Local file management (raw uploads, image crops)
│   │   ├── semantic_builder.py     # SemanticDocumentBuilder service
│   │   └── pipeline_service.py     # End-to-end orchestration service
│   └── utils/
│       ├── __init__.py
│       └── file_utils.py           # File validation, hashing, MIME checks
├── uploads/                        # Local file storage
│   ├── raw/                        # Uploaded PDFs
│   └── extracted/                  # Cropped figures, chart snapshots, table images
├── docker/
│   ├── Dockerfile                  # Python 3.12 Dockerfile with system dependencies
│   ├── docker-compose.yml          # FastAPI service + PostgreSQL 16
│   └── .dockerignore
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Test fixtures (SQLite/Postgres in-memory, TestClient)
│   ├── test_api.py                 # API endpoint tests
│   └── test_semantic_builder.py    # Unit tests for builder and schemas
├── requirements.txt
├── .env.example
├── README.md
└── ROADMAP.md                      # Architecture roadmap for Phase 2-5
```

---

### 2. Database Design & SQLAlchemy Models
[NEW] `backend/app/models/document.py`
- `Document`:
  - `id`: `UUID` (primary key)
  - `filename`: Original upload filename
  - `stored_path`: Absolute or relative path in `uploads/raw/`
  - `file_size`: Bytes
  - `mime_type`: Application/pdf
  - `page_count`: Integer
  - `status`: Enum (`PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`)
  - `semantic_json`: `JSONB` / `JSON` storing the unified semantic schema
  - `created_at`, `updated_at`: UTC timestamps
  - Relationships: `elements`, `jobs`

[NEW] `backend/app/models/document_element.py`
- `DocumentElement`:
  - `id`: `UUID` (primary key)
  - `document_id`: Foreign key to `documents.id`
  - `element_index`: Order of appearance
  - `type`: Enum (`text`, `table`, `image`, `chart`, `figure`)
  - `page`: Page number (1-indexed)
  - `bbox`: JSON `[x1, y1, x2, y2]`
  - `content`: JSON dictionary with element payload (text, html, markdown, image_path, confidence)
  - `created_at`: UTC timestamp

[NEW] `backend/app/models/processing_job.py`
- `ProcessingJob`:
  - `id`: `UUID` (primary key)
  - `document_id`: Foreign key to `documents.id`
  - `status`: Enum (`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`)
  - `step`: Current pipeline phase (`INIT`, `PDF_RENDERING`, `STRUCTURE_ANALYSIS`, `SEMANTIC_COMPILATION`, `PERSISTENCE`, `DONE`)
  - `error_message`: Nullable string
  - `started_at`, `completed_at`: UTC timestamps
  - `processing_metadata`: JSON with runtime stats (execution time, OCR config)

---

### 3. Unified Semantic Document Schema (Pydantic v2)
[NEW] `backend/app/schemas/semantic_document.py`
Matching the target specification:
```json
{
  "document_id": "string",
  "metadata": {
    "title": "string",
    "page_count": 0,
    "file_name": "string",
    "file_size": 0,
    "mime_type": "application/pdf",
    "created_at": "ISO-8601",
    "extra": {}
  },
  "elements": [
    {
      "id": "elem_uuid",
      "type": "text | table | image | chart | figure",
      "page": 1,
      "content": {
        "text": "...",
        "markdown": "...",
        "html": "...",
        "image_path": "uploads/extracted/...",
        "confidence": 0.95,
        "bbox": [10.0, 20.0, 300.0, 400.0],
        "reading_order": 1
      }
    }
  ],
  "entities": [],
  "claims": [],
  "relationships": [],
  "sources": []
}
```
Extensible Pydantic models with strict typing and validation:
- `ElementContent`: Polymorphic / flexible content model supporting text, tables, and visual crops.
- `SemanticElement`: Individual element with `type` constrained to `text | table | image | chart | figure`.
- `SemanticDocument`: Top-level container. Notice `entities`, `claims`, `relationships`, and `sources` are fully declared with schema stubs for future modules to populate seamlessly.

---

### 4. PP-StructureV3 Integration & Semantic Document Builder
[NEW] `backend/app/processors/base.py`
- `BaseStructureAnalyzer` defines the unified interface:
  `analyze(pdf_path: str, page_images: List[str]) -> List[RawExtractedElement]`

[NEW] `backend/app/processors/pp_structure.py`
- Implements `PPStructureAnalyzer` using PaddleOCR's `PPStructure` (`table=True`, `ocr=True`, `layout=True`).
- Extracts layout regions, tables (HTML/Markdown converted via TableEngine), figures/charts, and OCR bounding boxes.

[NEW] `backend/app/processors/fallback_analyzer.py`
- Fallback analyzer using PyMuPDF / pdfplumber for instant zero-GPU test environments, extracting text blocks, tables, and images with full bounding boxes.

[NEW] `backend/app/services/semantic_builder.py`
- `SemanticDocumentBuilder`:
  1. Ingests normalized extracted elements.
  2. Sorts elements into natural reading order based on coordinates and column layout.
  3. Sanitizes markdown tables and text blocks.
  4. Generates unique deterministic element IDs.
  5. Assembles metadata (title, page count, file stats).
  6. Validates against `SemanticDocument` Pydantic schema.
  7. Returns validated semantic model.

[NEW] `backend/app/services/pipeline_service.py`
- Orchestrates:
  1. Update `ProcessingJob` step.
  2. Parse PDF and extract page views.
  3. Run structure analyzer (`PP-StructureV3` or configured engine).
  4. Crop and save figures/charts to `uploads/extracted/`.
  5. Build Semantic Document JSON via `SemanticDocumentBuilder`.
  6. Persist results in PostgreSQL (`Document.semantic_json` and relational `DocumentElement` rows).
  7. Mark `Document` and `ProcessingJob` as `COMPLETED`.

---

### 5. FastAPI Endpoints
[NEW] `backend/app/api/v1/endpoints/documents.py`
- `POST /documents/upload`:
  - Validates MIME type and file extension (`.pdf`).
  - Saves file securely via `StorageService`.
  - Creates `Document` and `ProcessingJob` in PostgreSQL.
  - Spawns processing task via `BackgroundTasks`.
  - Returns `201 Created` with document summary and initial status.
- `GET /documents/{id}`:
  - Returns document details, current status, page count, and element breakdown count.
- `GET /documents/{id}/semantic`:
  - Returns the complete `SemanticDocument` JSON structure. Returns 404 if not found, 422 if still processing.
- `GET /documents`:
  - Paginated list of documents (`skip`, `limit`, `status` filter).

---

### 6. Docker & Local Deployment Setup
[NEW] `docker/Dockerfile`
- Multi-stage Python 3.12 image with required system libraries (`poppler-utils`, `libgl1`, `libgomp1`).
[NEW] `docker/docker-compose.yml`
- Services: `backend` (FastAPI) and `db` (PostgreSQL 16 with healthchecks and persistent volumes).

---

### 7. Extensibility & Future Phase Roadmap
[NEW] `ROADMAP.md`
- Clear documentation showing how future modules plug into the Semantic Document foundation:
  - **Phase 2 (Visual Intelligence)**: Qwen2.5-VL consuming `figure` and `chart` elements to generate rich semantic summaries.
  - **Phase 3 (Knowledge Engine)**: Populating `entities`, `claims`, and `relationships` from semantic elements.
  - **Phase 4 (Vector Search & Embeddings)**: Chunking semantic elements and indexing into `pgvector`.
  - **Phase 5 (Orchestration & LLM Generation)**: Content generation pipelines consuming the unified semantic schema.

---

## Verification Plan

### Automated Tests
- Unit test for `SemanticDocumentBuilder` verifying correct construction and schema validation.
- Integration tests for `/documents/upload`, `/documents/{id}`, `/documents/{id}/semantic`, and `/documents` using FastAPI `TestClient`.
- Test PDF generation and pipeline execution verification.
```powershell
python -m pytest backend/tests/ -v
```

### Manual Verification
- Run FastAPI service locally with uvicorn.
- Upload a test PDF file to `POST /api/v1/documents/upload`.
- Query `GET /api/v1/documents/{id}/semantic` to verify the generated JSON structure matches the SIH specification.
- Verify files in `uploads/raw` and `uploads/extracted`.
