# SIH-26154 — Change History

> **Purpose**: Quick-reference changelog for LLM continuity.  
> Read this file FIRST to understand what has already been done, what decisions were made, and what is pending.  
> Each entry is a self-contained session summary — no need to read the full codebase.

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
| File | Change |
|------|--------|
| `scripts/download_models.py` | Rewrote PP-Structure download to use HF repos; removed BOS tarball logic |
| `history.md` | Created (this file) |

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
| File | Change |
|------|--------|
| `backend/app/processors/pp_structure.py` | Uses local model dirs via `layout_model_dir`, `table_model_dir`, `det_model_dir`, `rec_model_dir` kwargs |
| `backend/app/core/config.py` | Added `MODELS_DIR`, `PP_STRUCTURE_MODEL_DIR` |
| `.gitignore` | Added `models/` and binary file masks |
| `docker/.dockerignore` | Same exclusions |
| `context.md` | Refreshed |

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
| File | Change |
|------|--------|
| `scripts/download_models.py` | Created from scratch |

---

## Architecture Quick Reference

```
SIH-26154/
├── backend/app/
│   ├── core/config.py          # Settings, model path constants
│   ├── processors/
│   │   ├── pp_structure.py     # PP-StructureV3 layout/table/OCR engine
│   │   └── base.py             # BaseStructureAnalyzer, RawDocumentElement
│   └── main.py                 # FastAPI app
├── docker/
│   ├── Dockerfile              # paddlepaddle==3.0.0, paddleocr==3.7.0
│   ├── docker-compose.yml
│   └── .dockerignore
├── scripts/
│   └── download_models.py      # Unified model downloader (HF-based)
├── models/                     # .gitignored — local model weights
│   └── pp_structure_v3/
│       ├── layout/             # PaddlePaddle/PP-DocLayout-L
│       ├── table/              # PaddlePaddle/SLANet_plus
│       ├── det/                # PaddlePaddle/PP-OCRv4_server_det
│       └── rec/                # PaddlePaddle/PP-OCRv4_server_rec
├── requirements.txt            # PP deps commented (Docker-only)
├── context.md                  # Full project context
└── history.md                  # THIS FILE — session changelog
```

### Version Pinning
| Package | Version | Why |
|---------|---------|-----|
| `paddlepaddle` | `3.0.0` | Required by PP-StructureV3 model format |
| `paddleocr` | `3.7.0` | Ships `PPStructureV3` class with `layout_model_dir` API |
| `paddlex[ocr]` | `>=3.7.0,<3.8.0` | PaddleX OCR pipeline integration |

### Known Gotchas
1. **PP-Structure V2 vs V3 weights** — V2 tarballs use `model.pdiparams` naming; V3 HF repos use `inference.pdiparams`. The download script now auto-detects and cleans V2 leftovers.
2. **PaddlePaddle on Windows** — does not install cleanly; PP-Structure deps are commented out in `requirements.txt` and only installed inside Docker.
3. **HF downloads** — `unsloth` repos are public and don't require auth tokens. Official `Qwen/` repos are gated.
4. **Q4 file bloat** — without `allow_patterns`, HF downloads ALL quant variants (7+ GB). Script restricts to `Q4_K_M.gguf` only.
