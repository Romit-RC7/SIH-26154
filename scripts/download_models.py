#!/usr/bin/env python3
"""
Model Weights Downloader Script for SIH-26154 Content Platform.
Uses native 'huggingface_hub' / new 'hf' CLI with resilient retry logic.

Target Model Directory Layout:
  models/
  ├── pp_structure_v3/       # Unified Layout (PicoDet), Table (SLANet), and OCR (PP-OCRv4)
  ├── unichart_base_960/     # ahmed-masry/unichart-base-960 (Universal Chart Reasoning)
  ├── qwen2.5_vl_3b_q4/      # bartowski/Qwen2.5-VL-3B-Instruct-GGUF (Multimodal Vision LLM)
  ├── qwen3_4b_q4/           # unsloth/Qwen3-4B-GGUF (Lightweight High-Efficiency LLM)
  ├── qwen3_8b_q4/           # unsloth/Qwen3-8B-GGUF (Core Reasoning LLM)
  └── bge_small_en_v1.5/     # BAAI/bge-small-en-v1.5 (pgvector Semantic Embeddings)
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
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

# PP-Structure v3 official components staged for single unified model call
PP_STRUCTURE_URLS = {
    "layout": {
        "urls": [
            "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar"
        ],
        "dir_name": "layout",
        "archive": "layout.tar",
    },
    "table": {
        "urls": [
            "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar"
        ],
        "dir_name": "table",
        "archive": "table.tar",
    },
    "det": {
        "urls": [
            "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar"
        ],
        "dir_name": "det",
        "archive": "det.tar",
    },
    "rec": {
        "urls": [
            "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
            "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar",
        ],
        "dir_name": "rec",
        "archive": "rec.tar",
    },
}


# ==============================================================================
# Helper Functions
# ==============================================================================

def download_with_retry(urls: list, archive_file: Path, max_retries: int = 4, timeout: int = 30) -> bool:
    """Download a file with multiple mirror support and exponential backoff retry."""
    temp_file = archive_file.with_suffix(".tmp")

    for url in urls:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[*] Downloading from {url} (Attempt {attempt}/{max_retries})...")
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp, open(temp_file, "wb") as out_f:
                    shutil.copyfileobj(resp, out_f)

                if temp_file.exists() and temp_file.stat().st_size > 1024:
                    temp_file.replace(archive_file)
                    print("    [+] Download successful.")
                    return True
            except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as e:
                print(f"    [!] Connection error on attempt {attempt}: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                if attempt < max_retries:
                    sleep_time = attempt * 3
                    print(f"    [*] Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

    return False


def run_hf_download(repo_id: str, target_dir: Path, patterns: list = None):
    """
    Download from Hugging Face using huggingface_hub Python API (Primary)
    or fallback to the modern 'hf download' CLI.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if files already exist inside target directory
    existing_models = list(target_dir.glob("*.gguf")) + list(target_dir.glob("*.safetensors")) + list(target_dir.glob("*.bin"))
    if existing_models:
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
    """Download and stage PP-Structure weights into a single unified directory."""
    pp_root = base_dir / "pp_structure_v3"
    pp_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(" Downloading PP-StructureV3 Components (Layout, Table, OCR)")
    print("=" * 80)

    for comp, meta in PP_STRUCTURE_URLS.items():
        dest_folder = pp_root / meta["dir_name"]
        if dest_folder.exists() and any(dest_folder.iterdir()):
            print(f"[*] {comp} already exists at {dest_folder}. Skipping.")
            continue

        dest_folder.mkdir(parents=True, exist_ok=True)
        archive_file = pp_root / meta["archive"]

        # Download with multi-URL fallback and retry
        success = download_with_retry(meta["urls"], archive_file)
        if not success:
            print(f"[!] Failed to download {comp} after all retry attempts.")
            continue

        print(f"[*] Extracting {meta['archive']} into {dest_folder}...")
        with tarfile.open(archive_file, "r:*") as tar:
            try:
                tar.extractall(path=dest_folder, filter="data")
            except TypeError:
                tar.extractall(path=dest_folder)

        # Flatten nested directory if the tar created an extra inner folder
        subdirs = [d for d in dest_folder.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            inner = subdirs[0]
            for item in inner.iterdir():
                target = dest_folder / item.name
                if not target.exists():
                    item.replace(target)
            try:
                inner.rmdir()
            except OSError:
                pass

        if archive_file.exists():
            archive_file.unlink()

    print(f"[+] PP-Structure models unified at: {pp_root}")


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
