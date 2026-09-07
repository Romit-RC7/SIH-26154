# AI-Powered Content Transformation Platform (SIH-26154)

> **SIH-26154 Solution**: An intelligent, AI-powered content transformation platform that ingests unstructured source documents, articles, reports, prompts, images, or videos, and converts them into customizable, multi-format communication deliverables (LinkedIn Posts, Twitter/X Threads, Executive Summaries, Presentation Decks, Infographic Packages, and Video Storyboards with Subtitles).

---

## System Architecture Pipeline (End-to-End)

```
Multi-Format Ingestion (PDF / DOCX / PPTX / Images / Video MP4)
       ↓
[Stage 1: Document Intelligence & Semantic Parsing]
  - PP-StructureV3 (Layout, OCR, Table SLANet)
  - PyMuPDF / python-docx / python-pptx / FFmpeg
       ↓
⭐ Canonical System Contract: Unified Semantic Document JSON ⭐
       ↓
[Stage 2: Visual Intelligence]
  - UniChart (Chart-to-Table & Data Comprehension)
  - Qwen2.5-VL-3B (Diagrams, Figures, Visual Reasoning)
       ↓
[Stage 3: Knowledge & Retrieval Layer]
  - BGE-small-en-v1.5 (384-dim dense vector embeddings)
  - PostgreSQL 16 + pgvector (sub-second cosine similarity search)
  - Knowledge Engine (Qwen3-4B claims & entity extraction)
       ↓
⭐ KnowledgePackage Payload (Grounding Evidence & Strategy Blueprint) ⭐
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

## Supported Input & Output Deliverables

### Ingestion Formats
- **PDF**: Multi-page layout analysis, table matrix extraction, visual crops.
- **DOCX**: Hierarchical headings, native tables, embedded images.
- **PPTX**: Slide-by-slide shapes, speaker notes, table matrix data.
- **Images**: PNG, JPG, JPEG, WEBP, BMP, TIFF visual elements.
- **Video**: MP4, WebM, MOV (FFmpeg 16kHz mono audio extraction + frame sampling every 10s + Faster-Whisper transcription).

### Generated Output Deliverables

| Deliverable | Key Features | Default Export Format |
|---|---|---|
| **LinkedIn Post** | 150-300 words, opening hook, body, CTA, 3-5 hashtags | JSON / Text / Markdown |
| **Twitter / X Thread** | 5-8 tweets, **strict $\le 280$ characters per tweet**, numbered sequence | JSON / Text |
| **Executive Summary** | Title, overview, key findings, data callouts, recommendations, conclusion | `.pdf` / `.docx` |
| **Presentation Deck** | 5-10 widescreen 16:9 slides, titles, bullet points, speaker notes, visual blueprints | `.pptx` |
| **Infographic Package** | Punchy headline, numeric stats, visual layout sections, interactive HTML preview | `.zip` (HTML/SVG/JSON) |
| **Video Script Package** | Scene storyboard, voiceover narration, visual cues, `.srt` subtitle timing file | `.zip` (.srt/JSON/MD) |
| **Blog Post / Article** | SEO meta description, H2 section headings, takeaways, tags | `.docx` |

---

## Staged Offline Models & Memory Management

All required AI weights run completely offline after initial download. Memory management is hardware-aware:

| Model Directory | Identifier | Hardware Allocation & Fallback |
|---|---|---|
| `models/pp_structure_v3/` | PP-StructureV3 | Offline OCR & Layout Parser ($\approx 600\text{ MB RAM}$) |
| `models/bge_small_en_v1.5/` | BGE-small-en-v1.5 | ONNX / Transformers dense vector embedding ($133\text{ MB}$) |
| `models/qwen2.5_vl_3b_q4/` | Qwen2.5-VL-3B Q4 | Multimodal visual reasoning via llama-cpp |
| `models/qwen3_4b_q4/` | Qwen3-4B Q4 | Knowledge Engine & automated schema repair loop |
| `models/qwen3_8b_q4/` | Qwen3-8B Q4 | Main generation LLM ($\ge 5.5\text{GB VRAM}$ GPU, $3\text{--}5.5\text{GB}$ partial, or CPU fallback) |
| `models/faster_whisper_small/` | Faster-Whisper-small | Audio transcription (loaded only when video contains audio) |

---

## REST API Reference

### 1. Document Upload & Ingestion
```bash
POST /api/v1/documents/upload
Content-Type: multipart/form-data
file: <pdf | docx | pptx | image | mp4>
```

### 2. Semantic Document JSON (System Contract)
```bash
GET /api/v1/documents/{document_id}/semantic
```

### 3. Knowledge Engine Retrieval & Assembly
```bash
POST /api/v1/knowledge/assemble/{document_id}
Content-Type: application/json
{
  "output_type": "executive_summary",
  "audience": "executive",
  "tone": "professional",
  "objective": "Summarize key findings and ROI metrics."
}
```

### 4. Content Orchestrator & Multi-Format Generation
```bash
POST /api/v1/generate/{document_id}
Content-Type: application/json
{
  "output_types": ["linkedin_post", "twitter_thread", "executive_summary", "presentation_deck", "video_script"],
  "audience": "executive",
  "tone": "professional",
  "detail_level": "moderate",
  "objective": "Highlight main security vulnerabilities and patching timelines.",
  "focus_keywords": ["Vulnerability", "Zero-day", "Patching"]
}
```

### 5. Trust & Validation Inspection
```bash
POST /api/v1/validate/{document_id}
Content-Type: application/json
{ ... GeneratedArtefact payload ... }
```

### 6. Binary Deliverable Export & Download
```bash
POST /api/v1/export/download
Content-Type: application/json
{
  "artefact": { ... GeneratedArtefact payload ... },
  "format": "docx"  # docx | pptx | pdf | zip | json
}
```

---

## Quickstart Guide

### Option A: Docker Compose (Recommended)

1. **CPU Default Mode**:
   ```bash
   cd docker
   docker-compose up --build
   ```

2. **NVIDIA GPU Mode**:
   ```bash
   cd docker
   docker-compose --profile gpu up --build
   ```

3. Access interactive Swagger API documentation at: `http://localhost:8000/docs`.

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

3. Download local model weights (selective or full):
   ```bash
   python scripts/download_models.py
   ```

4. Start FastAPI server:
   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```

---

## Running Test Suite

```bash
pytest backend/tests -v
```

All 75+ unit and integration tests execute against isolated test fixtures.
