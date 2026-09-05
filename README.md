# AI-Powered Content Transformation Platform (SIH-26154)

## Phase 1: Foundational Semantic Document Processing System

Phase 1 provides the foundational document intelligence and semantic extraction engine. It ingests document formats (**PDF** and **DOCX**), runs structural layout analysis using **PP-StructureV3** (with an intelligent fallback layer), extracts text, tables, figures, and charts, and normalizes them into a unified **Semantic Document JSON**.

> [!IMPORTANT]
> **The System Contract**:
> The `Semantic Document JSON` acts as the canonical system contract across the entire platform. Every downstream AI module (Qwen2.5-VL visual enricher, BGE Embedding Engine, pgvector semantic search, Knowledge Graph builder, Content Orchestrator, and Multi-Format Output Generators) consumes this structured contract rather than raw unstructured files.

---

## Architecture Flow

```
PDF / DOCX Upload
  ↓
Document Parser (PyMuPDF / docx)
  ↓
Offline Layout + OCR Stage
  ↓
Region Crops and Intermediate Recognition Document
  ↓
Table → Formula → Chart → Visual Recognition (staged)
  ↓
Qwen3-4B Fusion Stage
  ↓
Semantic Fusion Engine (reading order, captions, structure)
  ↓
Semantic Document Builder
       ↓
Unified Semantic Document JSON (Pydantic Schema)
       ↓
PostgreSQL Storage (JSONB document + relational elements)
```

    Recognition stages are resource-aware. Layout and OCR may share a loaded stage;
    table, formula, chart, Qwen2.5-VL, and Qwen3 models are loaded sequentially,
    with at most one or two model groups resident according to available RAM/VRAM.
    All model paths are local and missing optional weights must disable only that
    stage, never trigger a runtime download.

---

## The Semantic Document Schema (System Contract)

```json
{
  "version": "1.0.0",
  "document_id": "7b8f9e1a-4c2d-4e9f-8a1b-0c2d3e4f5a6b",
  "metadata": {
    "file_name": "quarterly_brief.pdf",
    "file_size": 248102,
    "mime_type": "application/pdf",
    "page_count": 4,
    "title": "Quarterly Financial Analysis",
    "created_at": "2026-09-04T10:15:30Z",
    "sha256": "3a7b9c1d2e...",
    "extra": {}
  },
  "elements": [
    {
      "id": "elem_7b8f9e1a_1_1",
      "type": "text",
      "page": 1,
      "bbox": [50.0, 72.0, 545.0, 110.0],
      "content": {
        "text": "Executive Summary and Strategic Goals...",
        "reading_order": 1,
        "confidence": 0.98
      }
    },
    {
      "id": "elem_7b8f9e1a_1_2",
      "type": "table",
      "page": 1,
      "bbox": [50.0, 130.0, 545.0, 280.0],
      "content": {
        "markdown": "| Metric | Target | Actual |\n| --- | --- | --- |\n| Revenue | $5M | $5.4M |",
        "html": "<table>...</table>",
        "reading_order": 2,
        "caption": "Table 1: Financial Performance Overview",
        "confidence": 0.97
      }
    },
    {
      "id": "elem_7b8f9e1a_2_1",
      "type": "figure",
      "page": 2,
      "bbox": [60.0, 90.0, 520.0, 360.0],
      "content": {
        "image_path": "uploads/extracted/7b8f9e1a/elem_7b8f9e1a_2_1.png",
        "caption": "Figure 1: Platform User Growth",
        "reading_order": 3
      }
    }
  ],
  "entities": [],
  "claims": [],
  "relationships": [],
  "sources": [
    {
      "id": "src_7b8f9e1a_1",
      "title": "quarterly_brief.pdf",
      "citation": "Original Document: quarterly_brief.pdf"
    }
  ]
}
```

---

## Tech Stack

- **Language & Runtime**: Python 3.12
- **API Framework**: FastAPI with modern asynchronous endpoints
- **Database**: PostgreSQL 16 (`pgvector/pgvector:pg16` ready for Phase 3)
- **ORM**: SQLAlchemy 2.0 (AsyncSession with `asyncpg`, sync fallback)
- **Data Contract & Validation**: Pydantic v2
- **Document & PDF Parsing**: PyMuPDF (`fitz`), `python-docx`
- **Structure & Layout OCR**: PP-StructureV3 (`paddleocr`), with automatic fallback
- **Containerization**: Docker & Docker Compose

---

## Folder Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── deps.py                 # Dependency injection (DB session)
│   │   └── v1/
│   │       ├── api.py              # Router aggregation
│   │       └── endpoints/
│   │           ├── documents.py    # POST /upload, GET /{id}, GET /{id}/semantic, GET /
│   │           └── health.py       # GET /health
│   ├── core/
│   │   ├── config.py               # Settings (Pydantic BaseSettings)
│   │   └── logging.py              # Structured logging
│   ├── database/
│   │   ├── base.py                 # SQLAlchemy DeclarativeBase
│   │   └── session.py              # Async / Sync engine & session management
│   ├── models/
│   │   ├── document.py             # Document model
│   │   ├── document_element.py     # DocumentElement model
│   │   └── processing_job.py       # ProcessingJob model
│   ├── schemas/
│   │   ├── semantic_document.py    # System Contract (Pydantic v2)
│   │   ├── document.py             # Request & Response schemas
│   │   └── processing_job.py       # Job tracking schema
│   ├── processors/
│   │   ├── base.py                 # Base analyzer abstractions
│   │   ├── pdf_parser.py           # PyMuPDF rasterizer & text block parser
│   │   ├── docx_parser.py          # DOCX parser (paragraphs, tables, media)
│   │   ├── pp_structure.py         # PP-StructureV3 integration
│   │   ├── fallback_analyzer.py    # Geometry & PyMuPDF fallback analyzer
│   │   └── extractor.py            # Normalization & visual crop persistence
│   ├── services/
│   │   ├── storage_service.py      # Local file & artifact management
│   │   ├── semantic_fusion.py      # Reading order & caption linkage engine
│   │   ├── semantic_builder.py     # Schema validation & assembly
│   │   └── pipeline_service.py     # End-to-end asynchronous pipeline
│   ├── utils/
│   │   └── file_utils.py           # Hashing, MIME resolution, validation
│   └── main.py                     # FastAPI application factory
├── uploads/
│   ├── raw/                        # Uploaded PDFs and DOCX files
│   └── extracted/                  # Cropped figures, charts, and tables
├── docker/
│   ├── Dockerfile                  # Python 3.12 Dockerfile with OpenCV & Poppler
│   └── docker-compose.yml          # FastAPI backend + PostgreSQL 16
├── tests/
│   ├── conftest.py                 # In-memory SQLite async fixtures
│   ├── test_semantic_document.py   # Schema & builder unit tests
│   ├── test_parsers.py             # Parser tests
│   └── test_api.py                 # API endpoint tests
├── requirements.txt
├── .env.example
└── ROADMAP.md                      # Phase 2-8 Architecture Roadmap
```

---

## API Reference

### 1. Upload Document
```bash
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: <document.pdf | document.docx>
```
**Response (201 Created)**:
```json
{
  "message": "Document accepted and enqueued for semantic processing",
  "document_id": "7b8f9e1a-4c2d-4e9f-8a1b-0c2d3e4f5a6b",
  "job_id": "0c1b2a3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d",
  "status": "PENDING",
  "filename": "quarterly_brief.pdf"
}
```

### 2. Get Document Status & Metadata
```bash
GET /api/v1/documents/{document_id}
```
**Response (200 OK)**:
```json
{
  "id": "7b8f9e1a-4c2d-4e9f-8a1b-0c2d3e4f5a6b",
  "filename": "quarterly_brief.pdf",
  "file_size": 248102,
  "mime_type": "application/pdf",
  "page_count": 4,
  "status": "COMPLETED",
  "element_count": 18,
  "created_at": "2026-09-04T10:15:30Z",
  "updated_at": "2026-09-04T10:15:38Z"
}
```

### 3. Get Unified Semantic Document JSON (System Contract)
```bash
GET /api/v1/documents/{document_id}/semantic
```
Returns the complete validated `SemanticDocument` JSON.

### 4. List Documents
```bash
GET /api/v1/documents?skip=0&limit=20&status=COMPLETED
```

### 5. Health Check
```bash
GET /api/v1/health
```

---

## Quickstart Guide

### Option A: Running with Docker Compose (Recommended)

1. Make sure Docker is running.
2. Build and start services:
   ```bash
   cd docker
   docker-compose up --build
   ```
3. Open API docs at `http://localhost:8000/docs`.

### Option B: Local Python Development

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Copy environment settings:
   ```bash
   cp .env.example .env
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```

---

## Running the Automated Test Suite

```bash
pytest backend/tests -v
```
All unit tests and API integration tests run against an in-memory SQLite database, requiring zero external infrastructure.
