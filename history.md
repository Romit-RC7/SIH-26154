# SIH-26154 — Change History

## Session 14 — 2026-09-08 — Pluggable VLM Engine: Moondream2 (1.6B VLM) Integration

### What was done

- **Implemented Pluggable Vision-Language Model Architecture**:
  - `backend/app/core/config.py`: Added `VLM_ENGINE` (`"qwen2.5_vl"` default, `"moondream2"` alternative) and `MOONDREAM_MODEL_DIR` (`models/moondream2/`).
  - `scripts/download_models.py`: Enabled `moondream2` (`vikhyatk/moondream2`) download specification under `--select moondream2`.
  - `backend/app/services/model_initializer/moondream_initializer.py`: Built `MoondreamInitializer` for lazy loading Moondream2 1.6B VLM onto CUDA GPU / CPU with explicit memory unloading (`gc.collect()`, `empty_cache()`). Exported in `model_initializer/__init__.py`.
  - `backend/app/services/recognition/image_service.py`: Modularized `ImageRecognitionService` to inspect `settings.VLM_ENGINE`. Supports seamless toggling between `_recognize_qwen()` and `_recognize_moondream()`. Formats Moondream2 predictions into the unified JSON contract (`visual_type`, `description`, `visible_text`, `key_details`, `core_concept_or_humor_theme`).
  - `backend/app/services/system_diagnostics.py`: Added `moondream2` readiness check and `vlm_engine` telemetry to system status endpoint (`GET /api/v1/health`).
  - `backend/tests/test_moondream_initializer.py`: Added unit tests for path resolution and VLM engine config toggling.

---

## Session 13 — 2026-09-08 — Document-Wide Visual Deduplication, Logo Filtering & Noise Suppression

### What was done

- **Implemented Visual Asset Deduplicator & Decorative Noise Filter Service**:
  - `backend/app/services/recognition/image_deduplicator.py`: Built `VisualDeduplicator` service. Calculates 64-bit difference hashes (`dhash`) and exact byte hashes for visual crops. Filters micro-icons, bullet emojis, and thin line separators (< 48x48 px or area < 2304 px²), marking them as `is_decorative_noise`. Compares Hamming distance (`VISUAL_DHASH_THRESHOLD = 4`) to identify duplicate visual assets across PPTX slides, DOCX reports, PDFs, and HTML documents. Detects document-wide recurring master slide/header logos (`VISUAL_RECURRING_FREQ_THRESHOLD = 0.30`).
- **Config Updates**:
  - `backend/app/core/config.py`: Added `VISUAL_MIN_DIMENSION_PX` (`48`), `VISUAL_MIN_AREA_PX` (`2304`), `VISUAL_DHASH_THRESHOLD` (`4`), and `VISUAL_RECURRING_FREQ_THRESHOLD` (`0.30`).
- **Specialist Vision & Pipeline Integration**:
  - `backend/app/services/recognition/image_service.py`: Filtered out elements marked `is_decorative_noise` or `is_duplicate` prior to running Qwen2.5-VL inference. Added `_propagate_duplicate_analysis()` to automatically propagate `visual_analysis` attributes from primary elements to all duplicate elements in the document.
  - `backend/app/services/recognition/coordinator.py`: Integrated `visual_deduplicator.process_elements()` into `recognize()` and `recognize_batch()`.
- **Knowledge Engine & Vector Database Search Optimization**:
  - `backend/app/services/embedding/chunker.py`: Skipped chunking and vector embedding generation for elements flagged as `is_decorative_noise`, `is_duplicate`, or `is_recurring_template_asset` to eliminate `pgvector` search noise and context bloat.
  - `backend/app/services/knowledge_engine.py`: Skipped duplicate elements and recurring master logos in `_extract_visual_insights()` so prompt context is not wasted on repetitive slide headers.
- **Added Comprehensive Test Suite**:
  - `backend/tests/test_image_deduplicator.py`: Added unit tests verifying micro-icon detection, dHash perceptual similarity matching, 5-page recurring logo deduplication, and visual analysis attribute propagation.

---

## Session 12 — 2026-09-08 — Refined Input Ingestion & Multimodal Graphic Processing

### What was done

- **Implemented Source Text & Prompt Ingestion Guardrail Service**:
  - `backend/app/services/input_validator.py`: Built `SourceInputValidator` and `ValidationResult`. Enforces minimum bounds (5 words, 30 characters), conversational greeting and typo fuzzy matching using `difflib` (rejecting pleasantries and misspellings such as "good gorming", "goood morning", "helo"), gibberish keyboard mash detection (5+ consonant clusters e.g. "asdfghjkl", vowel deficiency), repeated character and repetitive word spam filtering, and classification between `RAW_ARTICLE` and `FREEFORM_PROMPT` modes.
- **Implemented Text Document Parser**:
  - `backend/app/processors/text_parser.py`: Built `TextParser` to segment raw text inputs and `.txt` files into `ParsedPage` and `RawDocumentElement` objects, preserving markdown headers, paragraph reading orders, and input mode telemetry.
- **Extended Extractor & Storage Integration**:
  - `backend/app/core/config.py`: Added `.txt` to `ALLOWED_EXTENSIONS`.
  - `backend/app/processors/extractor.py`: Added `_extract_text()` to route `.txt` files through `text_parser.parse()`.
  - `backend/app/services/storage_service.py`: Added `save_text_content()` for raw UTF-8 persistence into `uploads/raw/{document_id}.txt`.
- **Exposed Direct Text Ingestion REST API Endpoint**:
  - `backend/app/schemas/document.py`: Added `TextSubmissionRequest` schema with `text` and optional `title`.
  - `backend/app/api/v1/endpoints/documents.py`: Added `POST /api/v1/documents/text`. Runs guardrail validation, creates `Document` and `ProcessingJob` database records, enqueues background processing, and returns `DocumentUploadResponse`.
- **Enhanced Multimodal Meme & Informal Graphic Visual Intelligence**:
  - `backend/app/services/recognition/image_service.py`: Updated Qwen2.5-VL prompt schema to include `meme` and `social_media_graphic` taxonomy, extract overlay OCR text into `overlay_text`, and identify underlying technical or humor themes in `core_concept_or_humor_theme`. Populates element attributes (`is_informal_graphic`, `overlay_text`, `core_concept`).
  - `backend/app/services/knowledge_engine.py`: Enhanced `_extract_visual_insights()` to format meme takeaways as `"Informal Graphic/Meme Overlay Text: '{overlay_text}'. Core Theme: {core_concept}"` so Stage 4 Content Orchestrator can translate informal tech memes into professional advisories and briefings.
  - `backend/app/services/validation/fact_checker.py`: Filtered out irrelevant visuals (`selfie`, `unrelated`, `irrelevant`) from the grounding evidence pool to prevent false evidence matching. Used `bge_initializer.encode` wrapper for robust cosine grounding.
- **Added Comprehensive Test Suites (21 new tests, 36/36 passing across entire test suite)**:
  - `backend/tests/test_input_validator.py`: Tests length bounds, greeting typos ("good gorming"), gibberish clusters, articles, and prompts.
  - `backend/tests/test_text_parser.py`: Tests raw text segmentation, markdown headers, and `.txt` file parsing.
  - `backend/tests/test_text_ingest_api.py`: Tests `POST /api/v1/documents/text` success and 400 error guardrail rejections.
  - `backend/tests/test_meme_recognition.py`: Tests Qwen2.5-VL meme payload parsing, Knowledge Engine insight formatting, and fact-checker visual filtering.

---

## Session 11 — 2026-09-08 — Video Keyframe Sampling Optimization & Frame-Diff Filtering

### What was done

- **Implemented Frame-Diff Keyframe Sampling Optimization**:
  - `backend/app/core/config.py`: Added `VIDEO_FRAME_DIFF_THRESHOLD` (`0.03` / 3% pixel change threshold), `VIDEO_MAX_KEYFRAMES` (`20` hard cap), and `VIDEO_CANDIDATE_FPS` (`0.333` fps / ~3s sampling interval).
  - `backend/app/processors/video_parser.py`: Upgraded video frame extraction with low-resolution (320x180) NumPy mean normalized pixel-difference thresholding. Drops static or near-identical duplicate frames while capturing all visual slide/scene transitions. Enforces a maximum keyframe cap (`VIDEO_MAX_KEYFRAMES: 20`) to bound downstream Qwen2.5-VL inference time.
  - `backend/tests/test_video_parser.py`: Added unit tests for frame difference calculations (identical, distinct, and partial changes) and timestamp sampling.

---

## Session 10 — 2026-09-08 — Trust & Validation Layer (Stage 5) and Multi-Format Export Builders (Stage 6)

### What was done

- **Implemented Stage 5: Trust, Validation & Schema Enforcement**:
  - `backend/app/schemas/validation.py`: Added `ClaimVerification`, `ValidationReport`, `VerifiedArtefact` schemas.
  - `backend/app/services/validation/fact_checker.py`: Fact verification using BGE sentence-level cosine similarity grounding and token overlap fallbacks.
  - `backend/app/services/validation/schema_validator.py`: Format constraint rule engine (LinkedIn word bounds, Twitter 280-char tweet check, Slide counts, Video scene rules).
  - `backend/app/services/validation/repair_service.py`: Automated repair loop (max 2 attempts) with hard truncation fallbacks.
  - `backend/app/services/validation/trust_service.py`: Master Trust coordinator integrating fact checking, schema validation, and auto-repair.
  - `backend/app/api/v1/endpoints/validate.py`: REST endpoint `POST /api/v1/validate/{document_id}`.
  - Integrated `trust_service` directly into `orchestrator_service.py` so all generated content automatically receives Trust Scores and schema enforcement.

- **Implemented Stage 6: Multi-Format Output Generation & Export Layer**:
  - `backend/app/services/formatters/docx_formatter.py`: Professional Word `.docx` generator using `python-docx`.
  - `backend/app/services/formatters/pptx_formatter.py`: 16:9 Widescreen PowerPoint `.pptx` deck builder with speaker notes using `python-pptx`.
  - `backend/app/services/formatters/pdf_formatter.py`: Publication PDF document builder using ReportLab and PyMuPDF.
  - `backend/app/services/formatters/video_package_builder.py`: Video Package `.zip` builder producing `.srt` subtitles, `script_storyboard.json`, and visual cues.
  - `backend/app/services/formatters/infographic_package_builder.py`: Infographic Package `.zip` builder producing interactive HTML preview and key metrics summary.
  - `backend/app/services/formatters/export_coordinator.py`: Master export manager persisting binary files to `uploads/exports/{document_id}/`.
  - `backend/app/api/v1/endpoints/export.py`: REST endpoint `POST /api/v1/export/download`.

- **Added Stage 5 & 6 Test Suites**:
  - `test_fact_checker.py`, `test_schema_validator.py`, `test_formatters.py`, `test_export_api.py`.

---

## Session 9 — 2026-09-08 — Content Orchestrator & Multi-Format Generation Layer (Phase 4)

### What was done

- **Created Schema Models for Generated Artefacts** (`backend/app/schemas/generated_artefact.py`):
  - Added structured content models: `LinkedInPostContent`, `TwitterThreadContent`, `ExecutiveSummaryContent`, `PresentationDeckContent`, `InfographicBriefContent`, `VideoScriptContent`, `BlogPostContent`.
  - Added `GeneratedArtefact`, `GenerateRequest`, `GenerateResponse` contract models.
- **Added Qwen3-8B Orchestrator Model Initializer** (`backend/app/services/model_initializer/qwen_initializers.py`):
  - Implemented `QwenOrchestratorInitializer` with dynamic VRAM detection (`_determine_gpu_layers`).
  - Automatic GPU offload on $\ge 5.5\text{ GB}$ VRAM, partial offload on $3.0\text{--}5.5\text{ GB}$, and graceful fallback to CPU mode / Qwen3-4B on lower-memory environments.
- **Implemented Prompt Builder** (`backend/app/services/orchestrator/prompt_builder.py`):
  - Built static format-specific prompt templates for all 8 output formats consuming `KnowledgePackage.orchestrator_prompt_context`.
- **Implemented Response Parser** (`backend/app/services/orchestrator/response_parser.py`):
  - Added JSON extraction from markdown code fences, trailing comma repair, and deterministic fallback builders.
- **Implemented Qwen3 Generation & Orchestration Services** (`backend/app/services/orchestrator/`):
  - `qwen3_generation_service.py`: Dispatches inference with format-specific temperature and max tokens.
  - `orchestrator_service.py`: Coordinates knowledge assembly and sequential multi-format deliverable generation.
- **Exposed REST API Endpoint** (`backend/app/api/v1/endpoints/generate.py`):
  - `POST /api/v1/generate/{document_id}` with multi-format generation and full Swagger documentation.
- **Added Comprehensive Unit & Integration Tests**:
  - `test_prompt_builder.py`, `test_response_parser.py`, `test_orchestrator.py`, `test_generate_api.py`.

---

# Session 7 — 2026-09-06 — Offline Video, Faster-Whisper, and Build Caching

## What was done

- Added video upload support for `.mp4`, `.webm`, and `.mov` with a 100 MB / two-minute policy exposed in Swagger.
- Added `video_parser.py`, which uses local FFmpeg/FFprobe to extract 16 kHz mono WAV audio and sample one 1280px-capped frame every 10 seconds (six per minute).
- Added a lazy `FasterWhisperInitializer` and `speech_service.py`. The model loads only when a batch contains extracted video audio and unloads before the Qwen vision stage.
- Added `Systran/faster-whisper-small` to the offline model downloader as `--select faster_whisper` and to system diagnostics.
- Added Docker FFmpeg and Faster-Whisper dependencies.
- Enabled BuildKit’s Dockerfile frontend and persistent pip cache mount for faster dependency-layer rebuilds.
- Corrected the downloader completion check so empty folders no longer appear to contain model weights.

## Runtime sequence

```text
PP-Structure -> formula + chart -> Faster-Whisper (audio only) -> Qwen2.5-VL (visual crops only) -> semantic fusion
```

All stages run from local files after the one-time Docker build and model download. No model weights are downloaded at inference time.

---


## Session 8 — 2026-09-06 — Knowledge & Retrieval Layer: Verification, Fixes & Documentation

### What was done

- **Diagnosed and fixed `/api/v1/knowledge/embed/{document_id}` 500 error**:
  - Root cause: `pgvector>=0.3.0` Python package was in `requirements.txt` but NOT installed in the Docker image (image was built before the package was added).
  - Fix: `docker exec sih_backend pip install pgvector>=0.3.0` + `docker restart sih_backend`. No rebuild needed.
  - Going forward: any new `requirements.txt` packages can be hot-installed with `docker exec pip install <pkg>` without a full 30+ minute rebuild.

- **Verified BGE embedding pipeline end-to-end in Docker**:
  - `embed_query("key findings")` → valid 384-dim non-zero float vector confirmed.
  - `retrieval_service.search()` → 3 chunks retrieved with cosine scores 0.61 / 0.59 / 0.58 for document `33a2e458-cb72-462a-a108-19e46dc77d5e`.
  - Confirmed: embeddings in DB are `type=list, len=384` — custom Vector type `bind_processor`/`result_processor` working correctly.

- **Fixed Swagger UX — search returning `[]`**:
  - Cause: Swagger placeholder default `"document_id": "string"` — literal text, not a real UUID.
  - Solution: Directed user to use `GET /api/v1/knowledge/search` which renders individual input fields for `query`, `document_id`, `top_k`, `min_similarity`.

- **Added `use_llm: bool` flag to `AssembleRequest`**:
  - `use_llm: false` → instant deterministic extraction (no Qwen3-4B cold-start wait).
  - `use_llm: true` (default) → full Qwen3-4B GGUF reasoning path.
  - File: `backend/app/api/v1/endpoints/knowledge.py`.

- **Created `knowledge_retrieval.md`**:
  - Full reference document for the Knowledge & Retrieval Layer.
  - Covers: data flow diagram, all sub-components, API endpoints, complete `KnowledgePackage` output contract (TypeScript-style schema), `orchestrator_prompt_context` structure, per-format `suggested_structure` lookup table, file map, and known gotchas.

- **Updated `context.md`**: Added Knowledge & Retrieval Layer sections (Section 6 & updated codebase inventory).

- **Architecture decision — KnowledgePackage as universal orchestrator input**:
  - The Content Orchestrator (Phase 4) will have **separate prompt templates per output format** (LinkedIn, Twitter, Exec Summary, Infographic, etc.).
  - All format-specific prompt templates will consume the same `KnowledgePackage` structure — format constraints (word limits, char limits, section counts) are the orchestrator's responsibility, NOT the knowledge layer's.
  - This is the correct separation of concerns: knowledge layer = WHAT to say; orchestrator prompts = HOW to format it.

### Files changed

| File | Change |
|------|--------|
| `backend/app/api/v1/endpoints/knowledge.py` | Added `use_llm: bool = True` field to `AssembleRequest` |
| `knowledge_retrieval.md` | **NEW** — Complete reference for Knowledge & Retrieval Layer |
| `context.md` | Added Session 8 additions (§6 Knowledge & Retrieval, updated codebase inventory) |
| `history.md` | This entry |

---

## Session 7 — 2026-09-06 — Knowledge & Retrieval Layer (BGE Embeddings + pgvector + Qwen3-4B Knowledge Engine)

### What was done
- **Created Embedding Engine (`backend/app/services/embedding/`)**:
  - `text_cleaner.py`: Unicode NFKC normalization, whitespace collapsing, control character removal, markdown table formatting, and dehyphenation.
  - `chunker.py`: Semantic, structure-aware chunking for text paragraphs, headers, tables (header-preservation across row splits), and visual captions with contextual metadata prefixes (`[Doc: {title} | Page {page} | Type: {type}]`).
  - `bge_initializer.py`: Offline lazy model loader and inference engine for `models/bge_small_en_v1.5/` producing normalized 384-dimensional dense vectors with query prefix support (`Represent this sentence for searching relevant passages: `) and fallback support.
  - `embedding_service.py`: Orchestrates full document cleaning, chunking, BGE dense embedding generation, and atomic database persistence.
- **Created Database Model for Chunks & pgvector (`backend/app/models/document_chunk.py`)**:
  - `DocumentChunk` table storing `(id, document_id, element_id, chunk_index, chunk_type, page, content, cleaned_text, chunk_metadata, embedding Vector(384))`.
  - Added `chunks` relationship to `Document` model.
- **Built Vector Retrieval Service (`backend/app/services/retrieval_service.py`)**:
  - Semantic similarity search using cosine distance ranking over 384-dim BGE embeddings with filtering by `document_id`, `chunk_types`, `page_range`, `top_k`, and `min_similarity`.
- **Created Intent & Personalization Schemas (`backend/app/schemas/intent.py`)**:
  - `IntentAndPersonalization`: Captures user configuration (`output_type` e.g. LinkedIn, PPT, Exec Summary; `audience`, `tone`, `language`, `objective`, `detail_level`, `focus_keywords`, `custom_instructions`).
- **Built Knowledge Engine & Output Contract (`backend/app/services/knowledge_engine.py` & `knowledge_package.py`)**:
  - Consumes Semantic Document JSON, pgvector semantic search retrieval, and user intent.
  - Utilizes `Qwen3-4B` (`models/qwen3_4b_q4/`) for reasoning, entity verification, factual claim extraction with source element citations, metric/table structuring, and content strategy generation.
  - Assembles `KnowledgePackage`: Standardized structured payload containing retrieved evidence, verified claims, key metrics, tables, visual insights, content strategy, and a high-density pre-compiled `orchestrator_prompt_context` Markdown block designed specifically for direct ingestion by Phase 5 Content Orchestrator.
- **Added REST API Endpoints (`backend/app/api/v1/endpoints/knowledge.py`)**:
  - `POST /api/v1/knowledge/embed/{document_id}`: Triggers document chunking and BGE embedding generation into pgvector.
  - `POST /api/v1/knowledge/search`: Vector similarity search endpoint.
  - `POST /api/v1/knowledge/assemble`: Main Knowledge Engine endpoint returning the complete `KnowledgePackage`.
- **Created Test Suite (25 new tests, 66/66 total tests passing)**:
  - `test_text_cleaner.py`, `test_chunker.py`, `test_bge_initializer.py`, `test_embedding_service.py`, `test_retrieval_service.py`, `test_knowledge_engine.py`, `test_knowledge_api.py`.

---

# Session 6 — 2026-09-05 — Staged Recognition and Initializer Architecture

## Architecture decision

The project now uses a batch-oriented offline recognition lifecycle. Documents
are queued and processed in bounded batches so large models are not loaded for
every document or page.

Current stage order:

```text
PP-Structure: layout + OCR + table recognition
    -> formula + chart recognition together
    -> Qwen2.5-VL image/figure recognition
    -> Qwen3 fusion (planned)
```

PP-Structure still owns table recognition because its pipeline requires table
classification, SLANet structure recognition, and wired/wireless cell detection
alongside layout and OCR. Formula and chart recognition are separate services
that share one resident stage.

## Model Initializers

All lazy initializers now live under:

```text
backend/app/services/model_initializer/
├── pp_structure_initializer.py
├── qwen_initializers.py
├── unichart_initializer.py
└── __init__.py
```

They validate local weights, load on demand, and expose explicit unload methods.
`PPStructureAnalyzer` delegates construction and unloading to
`PPStructureInitializer`.

## Recognition Services

The recognition package contains `coordinator.py`, `resource_manager.py`,
`chart_service.py`, and `image_service.py`.

- The coordinator runs formula, chart, and image stages across a batch.
- `chart_service.py` uses local PP-Chart2Table weights.
- `image_service.py` uses Qwen2.5-VL and its local multimodal projector.
- Formula and chart models stay resident together, then both unload before the
  image stage.
- Missing or incompatible optional models are recorded on affected elements.

## Batch Pipeline

1. Extract PDF/DOCX layout, OCR, tables, and crops for the whole batch.
2. Process formulas and charts across all documents in one shared stage.
3. Process images and figures with Qwen2.5-VL.
4. Build and persist each document independently.

Stable document and element IDs preserve provenance across the flattened batch.

## Pending Work

- Qwen3-8B/4B structured fusion is not wired into the pipeline yet.
- UniChart has an initializer but is not yet called by chart recognition.
- Text/table recognition remain owned by PP-Structure rather than separate services.
- Docker runtime validation and hardware-specific RAM/VRAM admission remain pending.

---

> **Purpose**: Quick-reference changelog for LLM continuity.  
> Read this file FIRST to understand what has already been done, what decisions were made, and what is pending.  
> Each entry is a self-contained session summary — no need to read the full codebase.

## Session 4 — 2026-09-06 ~03:00 IST

### What was done

- **Added PowerPoint Parser (`backend/app/processors/ppt_parser.py`)**:
  - Extracts text, tables (rendered as Markdown + HTML + row matrix), and embedded images from `.pptx` slides using `python-pptx`.
  - Maps each slide to a logical `ParsedPage` and shapes to `RawDocumentElement` entries with reading order and bounding boxes.
- **Added Standalone Image Parser (`backend/app/processors/image_parser.py`)**:
  - Ingests direct image formats (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`) as single-page documents with normalized coordinates and full-bleed image element attributes.
- **Updated Extractor Coordinator (`backend/app/processors/extractor.py`)**:
  - Extended extraction router to delegate `.pptx` to `_extract_pptx()` and images to `_extract_image()`.
  - Automatically crops and persists embedded and standalone images to `uploads/extracted/{document_id}/{element_id}.png` and attaches `saved_image_path` to element attributes.
- **Added `save_image_crop` to Storage Service (`backend/app/services/storage_service.py`)**:
  - Handles disk persistence of PIL Images and raw image bytes into `uploads/extracted/<document_id>/`, returning relative web/API paths with forward slashes.
- **Built System Diagnostics Service (`backend/app/services/system_diagnostics.py`)**:
  - Checks live existence and readiness (returning boolean `true`/`false`) of all offline staged models:
    - PP-Structure components: `layout_model`, `table_model`, `ocr_det_model`, `ocr_rec_model`
    - Reasoning & Vision LLMs: `qwen2.5_vl_3b_q4`, `qwen3_4b_q4`, `qwen3_8b_q4`, `unichart_base_960`, and `bge_small_en_v1.5`
  - Verifies engine initialization (`pp_structure_initialized`, `fallback_available`) and storage directory availability (`raw_uploads`, `extracted_uploads`).
- **Integrated System Diagnostics with Health Endpoint (`backend/app/api/v1/endpoints/health.py`)**:
  - Enriched `GET /api/v1/health` to expose database connectivity details and the full model/engine/storage boolean health status report.
- **Config & Requirements Updates**:
  - Added `python-pptx>=1.0.0` to `requirements.txt`.
  - Expanded `ALLOWED_EXTENSIONS` in `backend/app/core/config.py` to support `.pptx`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp` as a `ClassVar[Set[str]]`.

### Files changed

| File                                         | Change                                                                             |
| -------------------------------------------- | ---------------------------------------------------------------------------------- |
| `backend/app/processors/ppt_parser.py`       | New PPTX presentation parser extracting text, tables, and images                   |
| `backend/app/processors/image_parser.py`     | New direct image parser for standalone graphic formats                             |
| `backend/app/services/system_diagnostics.py` | New diagnostics utility checking model existence and engine health                 |
| `backend/app/services/storage_service.py`    | Added `save_image_crop` method to persist extracted images to `uploads/extracted/` |
| `backend/app/processors/extractor.py`        | Routed `.pptx` and image files, wired visual asset persistence                     |
| `backend/app/api/v1/endpoints/health.py`     | Connected `system_diagnostics` to return detailed true/false model availability    |
| `backend/app/core/config.py`                 | Expanded `ALLOWED_EXTENSIONS` to include PPTX and images                           |
| `requirements.txt`                           | Added `python-pptx>=1.0.0` dependency                                              |
| `context.md`                                 | Updated architecture, pipeline diagrams, and codebase inventory                    |
| `history.md`                                 | Recorded Session 4 changelog and architecture updates                              |

---

## Session 3 — 2026-09-05 ~02:30 IST

### What was done

- **Fixed PP-Structure model version mismatch**.  
  The download script was pulling **old V2 weights** from Baidu BOS tarballs:
  - `picodet_lcnet_x1_0_fgd_layout_cdla_infer` (V2 PicoDet layout)
  - `en_ppstructure_mobile_v2.0_SLANet_infer` (V2 SLANet table)

  These are **not compatible** with PaddleOCR 3.7.0 / `PPStructureV3`.

- **Replaced with correct V3 HuggingFace repos** in `scripts/download_models.py`:
  | Component | Old (V2 — wrong) | New (V3 — correct) |
  |-----------|-------------------|---------------------|
  | Layout | `picodet_lcnet_x1_0_fgd_layout_cdla` (BOS tar) | `PaddlePaddle/PP-DocLayout-L` (HF) |
  | Table | `en_ppstructure_mobile_v2.0_SLANet` (BOS tar) | `PaddlePaddle/SLANet_plus` (HF) |
  | Det | `ch_PP-OCRv4_det_infer` (BOS tar) | `PaddlePaddle/PP-OCRv4_server_det` (HF) |
  | Rec | `en_PP-OCRv4_rec_infer` (BOS tar) | `PaddlePaddle/PP-OCRv4_server_rec` (HF) |

- **Added V2-detection guard**: If old V2 weights are found (`model.pdiparams` without `inference.json`), the script auto-removes them and re-downloads V3.

- **Cleaned up dead code**: Removed `download_with_retry`, `tarfile`, `urllib` imports — all downloads now go through `huggingface_hub`.

### Files changed

| File                         | Change                                                                   |
| ---------------------------- | ------------------------------------------------------------------------ |
| `scripts/download_models.py` | Rewrote PP-Structure download to use HF repos; removed BOS tarball logic |
| `history.md`                 | Created (this file)                                                      |

### User action required

- Delete old `models/pp_structure_v3/layout/` and `models/pp_structure_v3/table/` folders (user said they would do this manually).
- Re-run `python scripts/download_models.py --select pp_structure` to fetch correct V3 weights.

---

## Session 2 — 2026-09-05 ~01:00 IST

### What was done

- **PP-Structure initializer** (`backend/app/processors/pp_structure.py`) refactored to load **local model weights** from `models/pp_structure_v3/{layout,table,det,rec}` instead of downloading at runtime.
- **Config** (`backend/app/core/config.py`) — added `MODELS_DIR` and `PP_STRUCTURE_MODEL_DIR` path constants.
- **Dockerfile** (`docker/Dockerfile`) already had correct versions: `paddlepaddle==3.0.0`, `paddleocr==3.7.0`, `paddlex[ocr]>=3.7.0,<3.8.0` — verified, no changes needed.
- **requirements.txt** — PP-Structure deps correctly commented out (Linux/Docker only). Verified.
- **`.gitignore` / `docker/.dockerignore`** — added `models/` and model file masks (`*.gguf`, `*.pd*`, etc.).
- **`context.md`** — updated with model inventory and architecture notes.

### Files changed

| File                                     | Change                                                                                                   |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `backend/app/processors/pp_structure.py` | Uses local model dirs via `layout_model_dir`, `table_model_dir`, `det_model_dir`, `rec_model_dir` kwargs |
| `backend/app/core/config.py`             | Added `MODELS_DIR`, `PP_STRUCTURE_MODEL_DIR`                                                             |
| `.gitignore`                             | Added `models/` and binary file masks                                                                    |
| `docker/.dockerignore`                   | Same exclusions                                                                                          |
| `context.md`                             | Refreshed                                                                                                |

---

## Session 1 — 2026-09-04 ~23:00 IST

### What was done

- **Created `scripts/download_models.py`** — unified model downloader for all project weights.
- **Model inventory established**:
  | Model | Repo | Quant / Variant | Size |
  |-------|------|-----------------|------|
  | UniChart | `ahmed-masry/unichart-base-960` | Full snapshot | ~960 MB |
  | Qwen2.5-VL-3B | `unsloth/Qwen2.5-VL-3B-Instruct-GGUF` | Q4_K_M + mmproj-F16 | ~3.2 GB |
  | Qwen3-4B | `unsloth/Qwen3-4B-GGUF` | Q4_K_M only | ~2.4 GB |
  | Qwen3-8B | `unsloth/Qwen3-8B-GGUF` | Q4_K_M only | ~4.9 GB |
  | BGE-small-en | `BAAI/bge-small-en-v1.5` | Full snapshot | ~130 MB |
  | PP-StructureV3 | PaddleOCR official | layout/table/det/rec | ~60 MB |

- **Fixed Qwen download issues**:
  - Switched from `Qwen/` repos (gated, 401 errors) to `unsloth/` repos (public GGUFs).
  - Limited Q4 downloads to single `Q4_K_M.gguf` file (was downloading all quants = 7+ GB).
  - Added `mmproj-F16.gguf` pattern for Qwen2.5-VL vision projector.

### Key decisions

- **Unsloth GGUFs are drop-in replacements** for official Qwen GGUFs — identical weights, just re-quantized and publicly hosted.
- **Q4_K_M chosen** as the best quality/size tradeoff for 4-bit quantization.
- Models are stored in `models/` (gitignored) and downloaded on-demand via the script.

### Files changed

| File                         | Change               |
| ---------------------------- | -------------------- |
| `scripts/download_models.py` | Created from scratch |

---

## Architecture Quick Reference

```
SIH-26154/
├── backend/app/
│   ├── api/v1/endpoints/
│   │   ├── health.py           # Health check & system diagnostics endpoint
│   │   └── documents.py        # Document upload & semantic retrieval
│   ├── core/config.py          # Settings, allowed extensions, model paths
│   ├── processors/
│   │   ├── pp_structure.py     # PP-StructureV3 layout/table/OCR engine
│   │   ├── fallback_analyzer.py# Rule-based PyMuPDF layout fallback
│   │   ├── pdf_parser.py       # PyMuPDF document rasterizer
│   │   ├── docx_parser.py      # python-docx document parser
│   │   ├── ppt_parser.py       # PowerPoint parser (text, tables, images)
│   │   ├── image_parser.py     # Standalone image parser (PNG, JPG, etc.)
│   │   ├── extractor.py        # Multi-modal extraction coordinator
│   │   └── base.py             # BaseStructureAnalyzer, RawDocumentElement
│   ├── services/
│   │   ├── system_diagnostics.py # Model existence & system diagnostics
│   │   ├── storage_service.py  # Disk storage & image crop persistence
│   │   ├── pipeline_service.py # Full async processing pipeline coordinator
│   │   ├── semantic_fusion.py  # Reading order sorting & caption linking
│   │   └── semantic_builder.py # SemanticDocument JSON builder
│   └── main.py                 # FastAPI app
├── docker/
│   ├── Dockerfile              # paddlepaddle==3.0.0, paddleocr==3.7.0
│   ├── docker-compose.yml
│   └── .dockerignore
├── scripts/
│   └── download_models.py      # Unified model downloader (HF-based)
├── models/                     # .gitignored — local model weights
│   ├── pp_structure_v3/        # Layout, Table, Det, Rec
│   ├── unichart_base_960/      # Chart comprehension
│   ├── qwen2.5_vl_3b_q4/       # Multimodal Vision (GGUF)
│   ├── qwen3_4b_q4/            # Fast reasoning LLM (GGUF)
│   ├── qwen3_8b_q4/            # Deep reasoning LLM (GGUF)
│   └── bge_small_en_v1.5/      # Vector embeddings
├── uploads/
│   ├── raw/                    # Uploaded original files
│   └── extracted/              # Extracted & cropped visual assets
├── requirements.txt            # Dependencies (includes python-pptx, PP deps commented)
├── context.md                  # Full project context
└── history.md                  # THIS FILE — session changelog
```

### Version Pinning

| Package        | Version          | Why                                                               |
| -------------- | ---------------- | ----------------------------------------------------------------- |
| `paddlepaddle` | `3.0.0`          | Required by PP-StructureV3 model format                           |
| `paddleocr`    | `3.7.0`          | Ships `PPStructureV3` class with `layout_model_dir` API           |
| `paddlex[ocr]` | `>=3.7.0,<3.8.0` | PaddleX OCR pipeline integration                                  |
| `python-pptx`  | `>=1.0.0`        | Native extraction of slides, shapes, tables, and images from PPTX |

### Known Gotchas

1. **PP-Structure V2 vs V3 weights** — V2 tarballs use `model.pdiparams` naming; V3 HF repos use `inference.pdiparams`. The download script now auto-detects and cleans V2 leftovers.
2. **PaddlePaddle on Windows** — does not install cleanly; PP-Structure deps are commented out in `requirements.txt` and only installed inside Docker.
3. **HF downloads** — `unsloth` repos are public and don't require auth tokens. Official `Qwen/` repos are gated.
4. **Q4 file bloat** — without `allow_patterns`, HF downloads ALL quant variants (7+ GB). Script restricts to `Q4_K_M.gguf` only.
