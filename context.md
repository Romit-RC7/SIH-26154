# SIH-26154 — Semantic Document Processing System: Technical Context

> **Project**: AI-Powered Content Transformation Platform (SIH-26154)  
> **Phase**: Milestone 1 — Foundational Semantic Document Processing System & Production PP-StructureV3 Migration  
> **Updated**: September 4, 2026  
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

Every downstream AI module (BGE Embedding Engine, pgvector semantic search, Qwen2.5-VL visual enricher, Knowledge Graph constructor, and Multi-Format Output Generators) interacts exclusively with the **Unified Semantic Document JSON** schema, rather than dealing directly with raw document bytes.

```
PDF / DOCX Upload
       ↓
Document Parsers (PyMuPDF / python-docx)
       ↓
Layout & Structure Analysis (Primary: PP-StructureV3 | Backup: PyMuPDF Fallback)
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

## 2. PaddleOCR PP-StructureV3 Production Migration

### Context & Incident

During Phase 1 operations, the runtime log reported:

```
ImportError: cannot import name 'PPStructure' from 'paddleocr'
Did you mean: 'PPStructureV3'?
```

Because of this import failure, the system automatically routed extraction to the `FallbackStructureAnalyzer` (PyMuPDF rule-based parser). While the pipeline completed with 41/41 tests passing, it lacked deep structural layout intelligence and table recognition.

### Discovery & Introspection Findings

In accordance with strict discovery requirements, the installed container environment (`paddleocr==3.7.0`) was introspected without making speculative assumptions:

1. **Architecture Shift in PaddleOCR 3.7.0**:
   - PaddleOCR 3.7.0 deprecated and removed the legacy `PPStructure` wrapper.
   - Replaced by `paddleocr.PPStructureV3`, built on top of `PaddleX 3.7.x` (`paddlex.inference.pipelines.layout_parsing`).
2. **PaddlePaddle Compatibility Issue**:
   - `paddlepaddle==2.6.2` lacked `AnalysisConfig.set_optimization_level()` and crashed (SIGSEGV) when parsing Paddle 3.0 Intermediate Representation (PIR `inference.json`) models.
   - Upgrading to `paddlepaddle==3.0.0` resolved the C++ predictor configuration and eliminated the segfault.
3. **PaddleX Pipeline Dependencies**:
   - `PPStructureV3` and `table_recognition_v2` require optional dependencies packaged under `paddlex[ocr]` (`beautifulsoup4`, `jinja2`, `scikit-learn`, `scipy`, `premailer`, `openpyxl`, etc.).
4. **Discovered Output Data Contract**:
   - Running `PPStructureV3.predict(img_np)` returns a list of `LayoutParsingResultV2` objects.
   - Each result contains:
     - `parsing_res_list`: List of `LayoutBlock` objects (`label`, `bbox`, `content`, `order_index`, `image`).
     - `table_res_list`: Extracted HTML tables (`pred_html`), cell bounding boxes (`cell_box_list`), and OCR cell texts.
     - `imgs_in_doc`: Pre-cropped PIL images of visual regions (`img`, `coordinate`, `score`, `label`).

### Technical Implementation

#### A. Core Engine (`backend/app/processors/pp_structure.py`)

- **Engine Initialization**:

  ```python
  import requests  # Pre-imported to avoid zlib C symbol collisions
  import chardet
  from paddleocr import PPStructureV3

  self.engine = PPStructureV3(
      use_doc_orientation_classify=False,
      use_doc_unwarping=False
  )
  ```

- **Dual-Mode Dynamic Parser (`_parse_results`)**:
  - **Native V3 Mode (`_parse_native_v3_results`)**:
    - `table`: Extracts `<table>` HTML, converts to clean Markdown using `_html_to_markdown_table`, preserves raw HTML and text.
    - `image` / `chart` / `figure`: Extracts pre-cropped `PIL.Image` directly from `block.image['img']` (or crops from page image if necessary).
    - `text` / `paragraph_title` / `title` / `header` / `footer`: Ingests reading order (`order_index`) and assigns semantic roles (`title`, `header`, `footer`, `caption`).
  - **Legacy Mode (`_parse_legacy_results`)**:
    - Parses legacy dictionary format `[{"type": ..., "bbox": ..., "res": ...}]` to ensure all existing mock unit tests pass without regression.
- **Engine Invocation Handling**:
  - Dynamically supports direct callable invocation (`self.engine(img_np)`), `.predict(img_np)`, and mock objects.

#### B. Robust Failover in Pipeline Extractor (`backend/app/processors/extractor.py`)

- `PPStructureV3` is configured as the **primary** engine.
- Wrapped in a dedicated `try...except Exception` block:

  ```python
  if use_pp:
      try:
          logger.info("Extracting PDF layout using PP-StructureV3...")
          for page in pages:
              if page.image:
                  page_elems = pp_structure_analyzer.analyze_page(page.image, page.page_number)
                  all_elements.extend(page_elems)
          extracted_with_pp = True
      except Exception as e:
          logger.error(f"PaddleOCR structure analysis failed. Falling back to PyMuPDF analyzer: {e}")
          all_elements = []
          extracted_with_pp = False

  if not extracted_with_pp:
      logger.info("Extracting PDF layout using Fallback / PyMuPDF Structure Analyzer...")
      # PyMuPDF extraction executes cleanly
  ```

#### C. Standardized Structured Logging

- Initialization success: `[INFO] PaddleOCR structure analyzer initialized successfully`
- Processing page: `[INFO] Processing page using PPStructureV3`
- Initialization failure: `[WARNING] PaddleOCR structure analyzer unavailable. Using fallback analyzer.`
- Runtime unexpected error: `[ERROR] PaddleOCR structure analysis failed. Falling back to PyMuPDF analyzer: {e}`

---

## 3. Architecture & Data Contracts

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
        markdown?: string,        // e.g. clean GFM table
        html?: string,            // e.g. raw <table>...</table>
        image_path?: string,      // relative storage path
        confidence?: float,
        reading_order?: int,
        caption?: string,
        table_structure?: {},
        raw_attributes?: {}       // role (header, footer, title), pp_type, layout_index
      }
    }
  ],
  relationships: [],              // Phase 2: entity / cross-element links
  sources: []
}
```

### 2. Database Models (SQLAlchemy 2.0 Async + PostgreSQL + pgvector)

Files: `backend/app/models/`

- **`Document` (`documents` table)**: Stores document metadata, SHA-256 hash, raw file path, page count, and status.
- **`DocumentElement` (`document_elements` table)**: Relational element rows with foreign keys to `Document`, containing bounding box coordinates, element type, text, markdown, image path, reading order, and a `Vector(768)` column for pgvector embeddings.
- **`ProcessingJob` (`processing_jobs` table)**: Asynchronous processing job tracker (`status`, `progress`, `error_message`, `stage`).

---

## 4. Codebase Inventory

```
Ai-Services/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── deps.py                       # DB & Service dependencies
│   │   │   └── v1/
│   │   │       ├── api.py                    # Router aggregation
│   │   │       └── endpoints/
│   │   │           ├── documents.py          # Upload, status, semantic retrieval, download
│   │   │           └── health.py             # Health check & engine availability status
│   │   ├── core/
│   │   │   ├── config.py                     # Pydantic BaseSettings (.env loading)
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
│   │   │   ├── pdf_parser.py                 # PyMuPDF document parser
│   │   │   ├── docx_parser.py                # python-docx document parser
│   │   │   ├── pp_structure.py               # PaddleOCR PPStructureV3 integration
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
│   │   └── main.py                           # FastAPI application & lifespan
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
│       └── test_structure_analyzers.py       # PPStructureV3, fallback, & failover tests
├── docker/
│   ├── Dockerfile                            # Production multi-stage Docker build
│   ├── docker-compose.yml                    # sih_backend + sih_postgres (pgvector)
│   └── .dockerignore                         # Docker build exclusions
├── uploads/
│   ├── raw/                                  # Uploaded original files (.gitkeep)
│   └── extracted/                            # Cropped visual regions (.gitkeep)
├── .env.example                              # Environment configuration template
├── .gitignore                                # Comprehensive Python/runtime ignore rules
├── requirements.txt                          # Python dependencies definition
├── PHASE1_IMPLEMENTATION_SUMMARY.md          # Architectural summary
└── PHASE1_TEST_REPORT.md                     # QA report (41 tests passing)
```

---

## 5. Verification & Test Status

### Structure Analyzer Test Suite (`test_structure_analyzers.py`)

Executed in `sih_backend` with 100% passing tests:

```bash
docker exec sih_backend pytest backend/tests/test_structure_analyzers.py -v
```

```
backend/tests/test_structure_analyzers.py::test_html_to_markdown_table_converter PASSED [ 12%]
backend/tests/test_structure_analyzers.py::test_html_to_markdown_table_empty PASSED [ 25%]
backend/tests/test_structure_analyzers.py::test_pp_structure_analyzer_parsing_mock PASSED [ 37%]
backend/tests/test_structure_analyzers.py::test_fallback_analyzer_direct_page_analysis PASSED [ 50%]
backend/tests/test_structure_analyzers.py::test_fallback_analyzer_when_pp_structure_unavailable PASSED [ 62%]
backend/tests/test_structure_analyzers.py::test_pp_structure_v3_native_parsing PASSED [ 75%]
backend/tests/test_structure_analyzers.py::test_pp_structure_initialization_failure_fallback PASSED [ 87%]
backend/tests/test_structure_analyzers.py::test_extractor_runtime_failure_fallback PASSED [100%]

============================== 8 passed in 202.01s ==============================
```

---

## 6. Commands Reference & Verification Guide

### 1. Docker Build & Start

```bash
docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml up --build -d
```

### 2. Verify Container Environment

```bash
# Check PaddleOCR version (Expected: 3.7.0)
docker exec sih_backend python -c "import paddleocr; print(paddleocr.__version__)"

# Check PaddlePaddle version (Expected: 3.0.0)
docker exec sih_backend python -c "import paddle; print(paddle.__version__)"

# Verify PPStructureV3 import & availability
docker exec sih_backend python -c "from paddleocr import PPStructureV3; print('PPStructureV3 OK')"
```

### 3. Check Service Logs

```bash
docker logs sih_backend
# Expected: [INFO] PaddleOCR structure analyzer initialized successfully
```

### 4. Run Pytest Suite

```bash
docker exec sih_backend pytest backend/tests/ -v
```

### 5. Document Processing Verification

```bash
# Health Check
curl -s http://localhost:8000/api/v1/health | jq .

# Upload Multi-modal Document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@sample.pdf"

# Retrieve Canonical Semantic JSON
curl -s http://localhost:8000/api/v1/documents/{document_id}/semantic | jq .
```

Verify that the output contains categorized element types:

- `table` (with Markdown & HTML representations)
- `image` / `chart` / `figure` (with `image_path` persisted to `/app/uploads/extracted/`)
- `text` (with reading order, headers, footers, and section titles)

---

## 7. Next Milestone: Phase 2 Roadmap

With Phase 1 document intelligence and PPStructureV3 stabilized, the platform is prepared for Phase 2:

1. **Dense Vector Embeddings**: Integrating BGE / sentence-transformers to embed text, table summaries, and visual captions into pgvector (`Vector(768)`).
2. **Hybrid Semantic Search**: BM25 keyword search + pgvector cosine similarity search across document elements.
3. **Qwen2.5-VL Visual Enrichment**: Passing cropped figures, charts, and tables to Qwen2.5-VL for semantic descriptions and knowledge graph entity extraction.
