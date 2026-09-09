# SIH-26154 — Semantic Document Processing & Content Transformation Engine: Technical Context

> **Project**: AI-Powered Content Transformation Platform (SIH-26154)
> **Status**: All 6 Pipeline Stages Complete (Document Intelligence, Visual Reasoning, Knowledge Retrieval, Content Orchestration, Trust & Validation, Multi-Format Export)
> **Updated**: September 9, 2026 (Session 11)
> **Repository**: [https://github.com/Romit-RC7/SIH-26154.git](https://github.com/Romit-RC7/SIH-26154.git)

---

## 0. Problem Statement (SIH-26154)

### Background

Organisations frequently need to convert information available in different forms such as news articles, reports, advisories, threat intelligence, policy documents, research papers, announcements, incident reports or free-form prompts into specific communication artefacts suitable for various purposes. The process of manually analysing the source content, understanding the desired objective and creating the required output format is time-consuming, resource-intensive and often requires expertise in content creation, communication and domain knowledge.

There is a need for an intelligent platform that can transform user-provided content into a desired output format through a simple and configurable interface.

### Description

The system shall act as an AI-powered content transformation engine that converts a common source of information into the specific deliverable requested by the operator, thereby reducing manual effort, improving consistency, accelerating content creation and enhancing operational efficiency.

The platform shall provide a dashboard through which an operator can submit source content in the form of high quality English language text, documents, articles, reports, prompts, images, videos or contextual information. In addition to providing the source content, the operator shall select one or more desired output types through configurable parameters available on the dashboard.

Based on the submitted content and the selected output type(s), the platform shall analyze the input, understand the context and intent, and generate the requested output artefact. The platform should support multiple output formats and allow operators to control generation parameters such as target audience, tone, language, level of detail, communication objective and content style.

In summary, platform shall generate output corresponding to the option(s) selected by the operator on the dashboard.

### Examples

- If **Video** is selected, generate a complete video package including script, storyboard, scene descriptions, narration text, subtitles and visual recommendations.
- If **LinkedIn Post** is selected, generate a professional LinkedIn post suitable for publication.
- If **Twitter/X Post** is selected, generate platform-optimized tweets or tweet threads.
- If **Advisory** is selected, generate a structured advisory document.
- If **Infographic** is selected, generate infographic content, layout recommendations and key messaging.
- If **Executive Summary** is selected, generate a concise executive briefing.
- If **Presentation** is selected, generate presentation slides and speaker notes.
- If multiple output formats are selected, generate all selected deliverables from the same source content.

### Expected Deliverables for Evaluation

- Source Code Link (GitHub/Drive Link)
- Readme with Setup Instructions
- Architecture Document (Max 2 Pages)
- Demo Video (Max 2 Minutes)
- Technical Presentation (Max 5 Slides)

---

## 1. System Overview & Architecture

The **SIH-26154 AI-Powered Content Transformation Platform** ingests direct text/prompt inputs, unstructured multi-page documents (**PDF**, **DOCX**, **PPTX**, **TXT**), standalone images/memes (**PNG**, **JPG**, **JPEG**, **WEBP**, **BMP**, **TIFF**), and **Video MP4/MOV/WebM**, converting them into customizable, multi-format communication deliverables.

```
Pasted Text / Prompt / Document / Image / Meme / Video
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[ Direct Text Ingestion ]     [ File / Visual Asset ]
       │                               │
       ▼                               ▼
[ SourceInputValidator ]      [ Format Router ]
 - Typo & Greeting Filter      - Image/Meme: Qwen2.5-VL Inspection
 - Gibberish & Entropy Check   - Video: Frame-diff Keyframe Extractor
 - Mode: RAW_ARTICLE / PROMPT          │
       │                               ▼
       ▼                      [ Stage 2: Visual Intelligence ]
 [ TextParser ]                - Meme classification & overlay OCR
       │                       - Core theme & concept extraction
       └──────────────┬────────────────┘
                      ▼
⭐ Unified Semantic Document JSON (System Contract) ⭐
                      ▼
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

## 3. Staged Model Residency & Hardware Allocation

All AI weights run locally offline after staging in `models/`:

| Directory | Model Identifier | Task / Purpose | Memory & Hardware Allocation |
|---|---|---|---|
| `models/pp_structure_v3/` | PP-StructureV3 | OCR, Layout & Table Recognition | $\approx 600\text{ MB RAM}$ |
| `models/unichart_base_960/` | UniChart-Base-960 | Chart fallback when PP-Chart2Table cannot run | Transformers PyTorch; local GPU when available |
| `models/qwen2.5_vl_3b_q4/` | Qwen2.5-VL-3B Q4 | Visual Diagram & Image Reasoning | GGUF + mmproj via llama-cpp; 2k context on the 8 GB RTX 4060 |
| `models/bge_small_en_v1.5/` | BGE-small-en-v1.5 | 384-dim Dense Vector Embeddings | ONNX / Transformers ($133\text{ MB}$) |
| `models/qwen3_4b_q4/` | Qwen3-4B Q4 | Knowledge Engine & Repair Loop | GGUF via llama-cpp ($\approx 2.5\text{ GB}$) |
| `models/qwen3_8b_q4/` | Qwen3-8B Q4 | Content Orchestrator LLM | GGUF via llama-cpp; 4k generation context, loaded after Qwen3-4B is released |
| `models/faster_whisper_small/` | Faster-Whisper-small | Speech-to-Text Transcription | CTranslate2 (loaded only for audio videos) |

---

## 4. Codebase Inventory

```
backend/
├── app/
│   ├── api/
│   │   ├── deps.py                            # DB session dependencies
│   │   └── v1/
│   │       ├── api.py                         # Master v1 API Router aggregator
│   │       └── endpoints/
│   │           ├── documents.py               # POST /upload, POST /text, GET /{id}, GET /{id}/semantic
│   │           ├── health.py                  # GET /health multi-model diagnostics
│   │           ├── knowledge.py               # POST /embed, POST /search, POST /assemble
│   │           ├── generate.py                # POST /generate/{id} (Stage 4)
│   │           ├── validate.py                # POST /validate/{id} (Stage 5)
│   │           └── export.py                  # POST /export/download (Stage 6)
│   ├── core/
│   │   ├── config.py                          # Settings & environment configuration (.txt support)
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
│   │   ├── document.py                        # Document API request/response + TextSubmissionRequest
│   │   ├── intent.py                          # IntentAndPersonalization & OutputType enums
│   │   ├── knowledge_package.py               # KnowledgePackage contract
│   │   ├── generated_artefact.py              # GeneratedArtefact & format content models
│   │   ├── validation.py                      # ClaimVerification & ValidationReport
│   │   └── semantic_document.py               # Canonical System Contract (Pydantic v2)
│   ├── processors/                            # Document Parsers
│   │   ├── base.py                            # Abstract BaseStructureAnalyzer
│   │   ├── text_parser.py                     # Direct text & .txt parser
│   │   ├── pdf_parser.py                      # PyMuPDF rasterizer
│   │   ├── docx_parser.py                     # Word parser
│   │   ├── ppt_parser.py                      # Presentation parser
│   │   ├── video_parser.py                    # FFmpeg video frame & audio extractor
│   │   ├── image_parser.py                    # Standalone image parser
│   │   ├── pp_structure.py                    # PP-StructureV3 analyzer
│   │   ├── fallback_analyzer.py               # Rule-based layout fallback
│   │   └── extractor.py                       # Visual crop persistence & format router
│   ├── services/
│   │   ├── input_validator.py                 # Source text guardrail validator (typo/gibberish filter)
│   │   ├── embedding/                         # BGE Chunking & Embedding package
│   │   ├── model_initializer/                 # Lazy GGUF & AI initializers
│   │   ├── recognition/                       # Staged Vision (Qwen2.5-VL meme/chart/image recognition)
│   │   ├── orchestrator/                      # Stage 4 Content Orchestrator
│   │   │   ├── prompt_builder.py              # Static prompt generators
│   │   │   ├── qwen3_generation_service.py    # Qwen3-8B generation wrapper
│   │   │   ├── response_parser.py             # JSON extraction & schema fallbacks
│   │   │   └── orchestrator_service.py        # Master Orchestration Service
│   │   ├── validation/                        # Stage 5 Trust & Validation
│   │   │   ├── fact_checker.py                # BGE grounding fact checker & visual filtering
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
│   │   ├── knowledge_engine.py                # Stage 3 Knowledge Engine (Meme insight formatting)
│   │   ├── retrieval_service.py               # pgvector cosine search
│   │   ├── pipeline_service.py                # End-to-end async processing coordinator
│   │   └── storage_service.py                 # File & visual crop storage
│   └── main.py                                # FastAPI app lifespan
├── docker/
│   ├── Dockerfile                             # Multi-stage Dockerfile with OpenCV & Poppler
│   └── docker-compose.yml                     # sih_backend + sih_postgres (pgvector)
├── tests/                                     # 95+ unit & integration tests
│   ├── test_input_validator.py                # Input validation guardrail unit tests
│   ├── test_text_parser.py                    # Direct text parsing unit tests
│   ├── test_text_ingest_api.py                # POST /text REST API integration tests
│   ├── test_meme_recognition.py              # Qwen2.5-VL meme parsing & insight tests
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

## 5. Verification & Status

- **Unit & Integration Tests**: 36/36 targeted & core test suites passing cleanly (`pytest backend/tests/test_input_validator.py backend/tests/test_text_parser.py backend/tests/test_text_ingest_api.py backend/tests/test_meme_recognition.py ...`).
- **REST Endpoints**: 16 endpoints exposed with interactive Swagger UI (including `POST /api/v1/documents/text`).
- **Docker Compose**: Ready for CPU (`docker compose up`) and GPU passthrough (`docker compose --profile gpu up`).

### Session 15 Runtime Update — September 9, 2026

- GPU runtime verified on RTX 4060 Laptop GPU (8 GB): Torch CUDA, Paddle CUDA, and llama.cpp GPU offload work inside `sih_backend_gpu`.
- All configured Qwen, BGE, Faster-Whisper, UniChart, and PP-Structure weight packages are locally staged. PP-Structure table orientation now reuses the local `doc_ori` package rather than downloading a duplicate at runtime.
- PP-Chart2Table weights are present, but the installed Paddle runtime lacks `fused_rms_norm_ext`; chart requests therefore use the staged UniChart Base 960 local fallback.
- Qwen2.5-VL uses a 2k context; Qwen3-8B generation uses a 4k context. Qwen3-4B is released after knowledge assembly before 8B is loaded, reducing 8 GB VRAM contention.
- `POST /api/v1/generate/{document_id}` can process a queued or failed document inline when semantic JSON is missing, then assemble knowledge, generate, and validate. Its JSON response includes `timings` for document processing (when run), knowledge assembly, generation, and end-to-end elapsed time.
- Operational logs now report model/pipeline durations while suppressing prompts, OCR previews, raw model responses, and llama.cpp native token traces. Qwen3-8B is used only by `/generate`, not by normal upload processing.

