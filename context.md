# SIH-26154 — Semantic Document Processing & Content Transformation Engine: Technical Context

> **Project**: AI-Powered Content Transformation Platform (SIH-26154)
> **Status**: All 6 Pipeline Stages Complete (Document Intelligence, Visual Reasoning, Knowledge Retrieval, Content Orchestration, Trust & Validation, Multi-Format Export)
> **Updated**: September 8, 2026 (Session 10)
> **Repository**: [https://github.com/Romit-RC7/SIH-26154.git](https://github.com/Romit-RC7/SIH-26154.git)

---

## 1. System Overview & Architecture

The **SIH-26154 AI-Powered Content Transformation Platform** ingests unstructured, multi-page documents (**PDF**, **DOCX**, **PPTX**, **Images**, and **Video MP4**) and converts them into customizable, multi-format communication deliverables.

```
Document / Video Ingestion
       ↓
[Stage 1: Document Intelligence & Semantic Parsing]
  - PP-StructureV3 Layout & OCR Parser
  - PyMuPDF / python-docx / python-pptx / FFmpeg
       ↓
⭐ Unified Semantic Document JSON (System Contract) ⭐
       ↓
[Stage 2: Visual Intelligence]
  - UniChart (Chart-to-Table & Data Extraction)
  - Qwen2.5-VL-3B (Diagrams, Figures, Visual Reasoning)
       ↓
[Stage 3: Knowledge & Retrieval Layer]
  - BGE-small-en-v1.5 (384-dim dense vector embeddings)
  - PostgreSQL 16 + pgvector (sub-second cosine similarity search)
  - Knowledge Engine (Qwen3-4B claims & entity extraction)
       ↓
⭐ KnowledgePackage Payload ⭐
       ↓
[Stage 4: Content Orchestrator & Generation Layer]
  - Format-specific static prompt builder
  - Qwen3-8B Q4 GGUF (VRAM-aware GPU offload / CPU fallback)
       ↓
[Stage 5: Trust, Validation & Schema Enforcement Layer]
  - BGE sentence-level fact verification & hallucination scoring (Trust Score 0-100%)
  - Schema rule enforcement & automated Qwen3-4B repair loop
       ↓
[Stage 6: Multi-Format Output Generation & Export Layer]
  - DocxFormatter (.docx), PptxFormatter (.pptx), PdfFormatter (.pdf)
  - VideoPackageBuilder (.srt, script_storyboard.json, b-roll .zip)
  - InfographicPackageBuilder (interactive HTML preview, metrics .zip)
```

---

## 2. Staged Model Residency & Hardware Allocation

All AI weights run locally offline after staging in `models/`:

| Directory | Model Identifier | Task / Purpose | Memory & Hardware Allocation |
|---|---|---|---|
| `models/pp_structure_v3/` | PP-StructureV3 | OCR, Layout & Table Recognition | $\approx 600\text{ MB RAM}$ |
| `models/unichart_base_960/` | UniChart-Base-960 | Plot/Chart Data Table Extraction | Transformers PyTorch ($\approx 800\text{ MB RAM}$) |
| `models/qwen2.5_vl_3b_q4/` | Qwen2.5-VL-3B Q4 | Visual Diagram & Image Reasoning | GGUF + mmproj via llama-cpp ($\approx 2.5\text{ GB}$) |
| `models/bge_small_en_v1.5/` | BGE-small-en-v1.5 | 384-dim Dense Vector Embeddings | ONNX / Transformers ($133\text{ MB}$) |
| `models/qwen3_4b_q4/` | Qwen3-4B Q4 | Knowledge Engine & Repair Loop | GGUF via llama-cpp ($\approx 2.5\text{ GB}$) |
| `models/qwen3_8b_q4/` | Qwen3-8B Q4 | Content Orchestrator LLM | GGUF via llama-cpp ($\ge 5.5\text{GB GPU}$ or CPU) |
| `models/faster_whisper_small/` | Faster-Whisper-small | Speech-to-Text Transcription | CTranslate2 (loaded only for audio videos) |

---

## 3. Codebase Inventory

```
backend/
├── app/
│   ├── api/
│   │   ├── deps.py                            # DB session dependencies
│   │   └── v1/
│   │       ├── api.py                         # Master v1 API Router aggregator
│   │       └── endpoints/
│   │           ├── documents.py               # POST /upload, GET /{id}, GET /{id}/semantic
│   │           ├── health.py                  # GET /health multi-model diagnostics
│   │           ├── knowledge.py               # POST /embed, POST /search, POST /assemble
│   │           ├── generate.py                # POST /generate/{id} (Stage 4)
│   │           ├── validate.py                # POST /validate/{id} (Stage 5)
│   │           └── export.py                  # POST /export/download (Stage 6)
│   ├── core/
│   │   ├── config.py                          # Settings & environment configuration
│   │   └── logging.py                         # Structured logging
│   ├── database/
│   │   ├── base.py                            # SQLAlchemy DeclarativeBase
│   │   └── session.py                         # Async engine & init_db pgvector setup
│   ├── models/
│   │   ├── document.py                        # Document ORM model
│   │   ├── document_chunk.py                  # DocumentChunk ORM + Vector(384)
│   │   ├── document_element.py                # DocumentElement ORM
│   │   └── processing_job.py                  # ProcessingJob status tracker
│   ├── schemas/
│   │   ├── document.py                        # Document API request/response
│   │   ├── intent.py                          # IntentAndPersonalization & OutputType enums
│   │   ├── knowledge_package.py               # KnowledgePackage contract
│   │   ├── generated_artefact.py              # GeneratedArtefact & format content models
│   │   ├── validation.py                      # ClaimVerification & ValidationReport
│   │   └── semantic_document.py               # Canonical System Contract (Pydantic v2)
│   ├── processors/                            # Document Parsers
│   │   ├── base.py                            # Abstract BaseStructureAnalyzer
│   │   ├── pdf_parser.py                      # PyMuPDF rasterizer
│   │   ├── docx_parser.py                     # Word parser
│   │   ├── ppt_parser.py                      # Presentation parser
│   │   ├── video_parser.py                    # FFmpeg video frame & audio extractor
│   │   ├── image_parser.py                    # Standalone image parser
│   │   ├── pp_structure.py                    # PP-StructureV3 analyzer
│   │   ├── fallback_analyzer.py               # Rule-based layout fallback
│   │   └── extractor.py                       # Visual crop persistence
│   ├── services/
│   │   ├── embedding/                         # BGE Chunking & Embedding package
│   │   ├── model_initializer/                 # Lazy GGUF & AI initializers
│   │   ├── orchestrator/                      # Stage 4 Content Orchestrator
│   │   │   ├── prompt_builder.py              # Static prompt generators
│   │   │   ├── qwen3_generation_service.py    # Qwen3-8B generation wrapper
│   │   │   ├── response_parser.py             # JSON extraction & schema fallbacks
│   │   │   └── orchestrator_service.py        # Master Orchestration Service
│   │   ├── validation/                        # Stage 5 Trust & Validation
│   │   │   ├── fact_checker.py                # BGE grounding fact checker
│   │   │   ├── schema_validator.py            # Format constraint rule engine
│   │   │   ├── repair_service.py              # Automated repair loop
│   │   │   └── trust_service.py               # Master Trust Service
│   │   ├── formatters/                        # Stage 6 Output Export Builders
│   │   │   ├── docx_formatter.py              # Word .docx builder
│   │   │   ├── pptx_formatter.py              # PowerPoint .pptx builder
│   │   │   ├── pdf_formatter.py               # PDF builder
│   │   │   ├── video_package_builder.py       # Video package .zip (.srt, script, b-roll)
│   │   │   ├── infographic_package_builder.py # Infographic package .zip (HTML/SVG)
│   │   │   └── export_coordinator.py          # Master export manager
│   │   ├── knowledge_engine.py                # Stage 3 Knowledge Engine
│   │   ├── retrieval_service.py               # pgvector cosine search
│   │   ├── pipeline_service.py                # End-to-end async processing coordinator
│   │   └── storage_service.py                 # File & visual crop storage
│   └── main.py                                # FastAPI app lifespan
├── docker/
│   ├── Dockerfile                             # Multi-stage Dockerfile with OpenCV & Poppler
│   └── docker-compose.yml                     # sih_backend + sih_postgres (pgvector)
├── tests/                                     # 75+ unit & integration tests
│   ├── test_prompt_builder.py
│   ├── test_response_parser.py
│   ├── test_orchestrator.py
│   ├── test_fact_checker.py
│   ├── test_schema_validator.py
│   ├── test_formatters.py
│   ├── test_export_api.py
│   └── test_generate_api.py
```

---

## 4. Verification & Status

- **Unit & Integration Tests**: All test suites passing.
- **REST Endpoints**: 15 endpoints exposed with interactive Swagger UI.
- **Docker Compose**: Ready for CPU (`docker compose up`) and GPU passthrough (`docker compose --profile gpu up`).
