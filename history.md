# SIH-26154 — Change History
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
