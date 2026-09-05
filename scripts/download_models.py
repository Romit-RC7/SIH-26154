#!/usr/bin/env python3
"""
Model Weights Downloader Script for SIH-26154 Content Platform.
Uses native 'huggingface_hub' / new 'hf' CLI with resilient retry logic.

Target Model Directory Layout:
  models/
    ├── pp_structure_v3/       # PP-StructureV3 layout, OCR, tables, formulas, charts
  ├── unichart_base_960/     # ahmed-masry/unichart-base-960 (Universal Chart Reasoning)
  ├── qwen2.5_vl_3b_q4/      # unsloth/Qwen2.5-VL-3B-Instruct-GGUF (Multimodal Vision LLM)
  ├── qwen3_4b_q4/           # unsloth/Qwen3-4B-GGUF (Lightweight High-Efficiency LLM)
  ├── qwen3_8b_q4/           # unsloth/Qwen3-8B-GGUF (Core Reasoning LLM)
  └── bge_small_en_v1.5/     # BAAI/bge-small-en-v1.5 (pgvector Semantic Embeddings)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ==============================================================================
# Hugging Face Model Specifications
# ==============================================================================

HF_MODELS = {
    "unichart": {
        "repo_id": "ahmed-masry/unichart-base-960",
        "patterns": None,  # Full snapshot
        "target_dir": "unichart_base_960",
        "description": "UniChart Base 960 (Universal Chart Comprehension & Reasoning)",
    },
    "qwen2.5_vl": {
        "repo_id": "unsloth/Qwen2.5-VL-3B-Instruct-GGUF",
        "patterns": ["*Q4_K_M.gguf", "*q4_k_m.gguf", "*mmproj-F16.gguf", "*mmproj-f16.gguf"],
        "target_dir": "qwen2.5_vl_3b_q4",
        "description": "Qwen2.5-VL-3B-Instruct Q4_K_M GGUF (Vision Intelligence, ~1.9 GB + mmproj-F16 ~1.3 GB)",
    },
    "qwen3_4b": {
        "repo_id": "unsloth/Qwen3-4B-GGUF",
        "patterns": ["*Q4_K_M.gguf", "*q4_k_m.gguf"],
        "target_dir": "qwen3_4b_q4",
        "description": "Qwen3-4B Q4_K_M GGUF (Fast Compact Reasoning LLM, ~2.4 GB)",
    },
    "qwen3_8b": {
        "repo_id": "unsloth/Qwen3-8B-GGUF",
        "patterns": ["*Q4_K_M.gguf", "*q4_k_m.gguf"],
        "target_dir": "qwen3_8b_q4",
        "description": "Qwen3-8B Q4_K_M GGUF (Primary High-Reasoning LLM, ~4.9 GB)",
    },
    "bge": {
        "repo_id": "BAAI/bge-small-en-v1.5",
        "patterns": None,  # Full repository snapshot
        "target_dir": "bge_small_en_v1.5",
        "description": "BGE-small-en-v1.5 (High-Speed Embeddings for pgvector)",
    },
}

# ---------------------------------------------------------------------------
# PP-StructureV3 components — downloaded from official HuggingFace repos.
# These are the models that PaddleOCR 3.7.0 / PPStructureV3 actually ships
# with.  Older V2 tarballs (PicoDet layout, SLANet v2.0) are NOT compatible
# with PPStructureV3 and must NOT be used.
#
# Each entry uses huggingface_hub snapshot_download so we get the PaddleX
# inference weights and metadata required by PPStructureV3.
# ---------------------------------------------------------------------------
PP_STRUCTURE_HF = {
    "layout": {
        "repo_id": "PaddlePaddle/PP-DocLayout-L",
        "dir_name": "layout",
        "description": "PP-DocLayout-L (RT-DETR layout detection, PP-StructureV3)",
    },
    "table": {
        "repo_id": "PaddlePaddle/SLANet_plus",
        "dir_name": "table",
        "description": "SLANet_plus (Table structure recognition, PP-StructureV3)",
    },
    "det": {
        "repo_id": "PaddlePaddle/PP-OCRv4_server_det",
        "dir_name": "det",
        "description": "PP-OCRv4 server detection model",
    },
    "rec": {
        "repo_id": "PaddlePaddle/PP-OCRv4_server_rec",
        "dir_name": "rec",
        "description": "PP-OCRv4 server recognition model",
    },
    "table_cls": {
        "repo_id": "PaddlePaddle/PP-LCNet_x1_0_table_cls",
        "dir_name": "table_cls",
        "description": "PP-LCNet table classification model",
    },
    "wired_table_cells": {
        "repo_id": "PaddlePaddle/RT-DETR-L_wired_table_cell_det",
        "dir_name": "wired_table_cells",
        "description": "RT-DETR wired table cell detection model",
    },
    "wireless_table_cells": {
        "repo_id": "PaddlePaddle/RT-DETR-L_wireless_table_cell_det",
        "dir_name": "wireless_table_cells",
        "description": "RT-DETR wireless table cell detection model",
    },
    "chart": {
        "repo_id": "PaddlePaddle/PP-Chart2Table",
        "dir_name": "chart",
        "description": "PP-Chart2Table chart recognition model",
        "required_files": ["model_state.pdparams", "inference.yml"],
    },
    "formula": {
        "repo_id": "PaddlePaddle/PP-FormulaNet_plus-M",
        "dir_name": "formula",
        "description": "PP-FormulaNet_plus-M mathematical formula recognition model",
    },
}


# ==============================================================================
# Helper Functions
# ==============================================================================


def run_hf_download(
    repo_id: str,
    target_dir: Path,
    patterns: list = None,
    required_files: list = None,
):
    """
    Download from Hugging Face using huggingface_hub Python API (Primary)
    or fallback to the modern 'hf download' CLI.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # A PP-Structure folder is complete only when its required metadata and
    # weights are present. This avoids treating a partial download as valid.
    if required_files:
        is_complete = all((target_dir / name).exists() for name in required_files)
    else:
        is_complete = any(
            target_dir.glob(pattern)
            for pattern in ("*.gguf", "*.safetensors", "*.bin", "*.pdiparams", "*.pdmodel")
        )
    if is_complete:
        print(f"[*] Target files already exist in {target_dir}. Skipping.")
        return

    # 1. Primary: Use huggingface_hub Python library directly
    try:
        from huggingface_hub import snapshot_download
        print(f"[*] Using huggingface_hub Python API to fetch '{repo_id}'...")

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            allow_patterns=patterns,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print(f"[+] Snapshot complete for {repo_id} -> {target_dir}")
        return
    except ImportError:
        print("[!] 'huggingface_hub' Python library not imported. Falling back to 'hf' CLI...")
    except Exception as e:
        print(f"[!] huggingface_hub error ({e}). Attempting 'hf' CLI fallback...")

    # 2. Fallback: Use new 'hf' CLI command
    hf_binary = shutil.which("hf") or shutil.which("huggingface-cli")
    if not hf_binary:
        print("[!] ERROR: Neither 'huggingface_hub' python package nor 'hf' CLI was found.")
        print("    Please run: pip install -U huggingface_hub")
        return

    cmd = [hf_binary, "download", repo_id, "--local-dir", str(target_dir)]
    if patterns:
        for p in patterns:
            cmd.extend(["--include", p])

    print(f"\n[EXEC] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[!] Warning: '{hf_binary}' exited with code ({result.returncode}) for {repo_id}")
    else:
        print(f"[+] Successfully downloaded {repo_id} -> {target_dir}")


def download_pp_structure(base_dir: Path):
    """Download PP-StructureV3 model weights from official HuggingFace repos.

    Uses the same huggingface_hub / CLI fallback logic as HF_MODELS downloads.
    Each component is downloaded into its own
    subdirectory under models/pp_structure_v3/.

    The correct V3 models are:
      - layout : PaddlePaddle/PP-DocLayout-L     (RT-DETR, NOT the old PicoDet)
      - table  : PaddlePaddle/SLANet_plus         (NOT the old SLANet v2.0)
      - det    : PaddlePaddle/PP-OCRv4_server_det
      - rec    : PaddlePaddle/PP-OCRv4_server_rec
    - table_cls : PaddlePaddle/PP-LCNet_x1_0_table_cls
    - wired_table_cells : PaddlePaddle/RT-DETR-L_wired_table_cell_det
    - wireless_table_cells : PaddlePaddle/RT-DETR-L_wireless_table_cell_det
    - chart : PaddlePaddle/PP-Chart2Table
    - formula : PaddlePaddle/PP-FormulaNet_plus-M
    """
    pp_root = base_dir / "pp_structure_v3"
    pp_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(" Downloading PP-StructureV3 Components (Layout, Table, OCR)")
    print(" Source: Official PaddlePaddle HuggingFace repos")
    print("=" * 80)

    for comp, meta in PP_STRUCTURE_HF.items():
        dest_folder = pp_root / meta["dir_name"]

        required_files = meta.get("required_files", ["inference.pdiparams", "inference.yml"])
        has_complete_model = all((dest_folder / name).exists() for name in required_files)

        if has_complete_model:
            print(f"[*] {comp} ({meta['description']}) already present at {dest_folder}. Skipping.")
            continue

        # Warn and clean up if old V2 weights are detected (wrong model version)
        has_old_v2 = (dest_folder / "model.pdiparams").exists() and not (dest_folder / "inference.json").exists()
        if has_old_v2:
            print(f"[!] WARNING: Old V2 weights detected in {dest_folder}. Removing before V3 download...")
            shutil.rmtree(dest_folder, ignore_errors=True)

        dest_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n---> Fetching {meta['description']}...")
        # Re-use the same HF download logic with pattern filter for paddle inference files
        run_hf_download(
            repo_id=meta["repo_id"],
            target_dir=dest_folder,
            patterns=[
                "*.pdparams",
                "*.pdiparams",
                "*.pdmodel",
                "*.pdiparams.info",
                "*.yml",
                "*.yaml",
                "*.json",
            ],
            required_files=required_files,
        )

    print(f"\n[+] PP-StructureV3 models staged at: {pp_root}")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download model weights for SIH-26154 Content Platform.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "models"),
        help="Base models directory.",
    )
    parser.add_argument(
        "--select",
        nargs="+",
        choices=["all", "pp_structure", "unichart", "qwen2.5_vl", "qwen3_4b", "qwen3_8b", "bge"],
        default=["all"],
        help="Select specific models to download.",
    )

    args = parser.parse_args()
    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.select)
    download_all = "all" in selected

    print("=" * 80)
    print(" SIH-26154 Model Downloader")
    print(f" Destination Directory : {models_dir}")
    print(f" Selected Targets      : {', '.join(args.select)}")
    print("=" * 80)

    # 1. Download PP-Structure (Single Unified Layout + Table + OCR pipeline)
    if download_all or "pp_structure" in selected:
        download_pp_structure(models_dir)

    # 2. Download Hugging Face Models
    hf_keys = [k for k in ["unichart", "qwen2.5_vl", "qwen3_4b", "qwen3_8b", "bge"] if download_all or k in selected]

    if hf_keys:
        for key in hf_keys:
            spec = HF_MODELS[key]
            print(f"\n---> Fetching {spec['description']}...")
            target_path = models_dir / spec["target_dir"]
            run_hf_download(
                repo_id=spec["repo_id"],
                target_dir=target_path,
                patterns=spec["patterns"],
            )

    print("\n" + "=" * 80)
    print(" ALL DOWNLOADS COMPLETED / STAGED SUCCESSFULLY!")
    print(f" Staged models available in: {models_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
