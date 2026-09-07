# Architecture Implementation Roadmap: AI-Powered Content Transformation Engine (SIH-26154)

All architectural phases of the **SIH-26154 Content Transformation Engine** are fully implemented, verified, and integrated.

---

## Contract-Centric Architecture

```
Raw Multimodal Input (PDF / DOCX / PPTX / Images / Video MP4)
                ↓
    [ Stage 1: Document Processing & Intelligence ]
                ↓
     ⭐ Semantic Document JSON Contract ⭐
                ↓
    [ Stage 2: Visual Intelligence (UniChart + Qwen2.5-VL) ]
                ↓
    [ Stage 3: Knowledge Engine & pgvector (BGE + Qwen3-4B) ]
                ↓
     ⭐ KnowledgePackage Payload ⭐
                ↓
    [ Stage 4: Content Orchestrator (Qwen3-8B Generation) ]
                ↓
    [ Stage 5: Trust, Validation & Schema Enforcement Layer ]
                ↓
    [ Stage 6: Multi-Format Output Export Builders (.docx, .pptx, .pdf, .zip) ]
```

---

## Phase Execution Summary

### Phase 1: Document Processing & Semantic Foundation (Completed)
- FastAPI web server and dependency container.
- Multi-page PDF rasterization (PyMuPDF), DOCX parsing, PPTX parsing, and standalone Image parsing.
- Offline Video Ingestion: MP4, WebM, MOV with FFmpeg 16kHz mono WAV extraction and 10s frame sampling.
- PP-StructureV3 layout analysis, table recognition (SLANet), and OCR (with automatic PyMuPDF fallback).
- Multi-modal extraction: figure/chart visual cropping and storage under `uploads/extracted/`.
- Semantic Fusion Engine: reading order reconciliation and caption linking.
- Assembly & validation of `SemanticDocument` JSON.
- PostgreSQL 16 persistence (JSONB document + relational elements).

---

### Phase 2: Visual Intelligence & Specialist Models (Completed)
- `UniChart-Base-960` for plot and chart data table extraction.
- `Qwen2.5-VL-3B Q4` GGUF for visual diagram, figure, and flowchart explanation.
- `Faster-Whisper-small` for speech-to-text audio transcription.

---

### Phase 3: Embeddings + pgvector + Knowledge Engine (Completed)
- Structure-aware chunking and text cleaning (`text_cleaner.py`, `chunker.py`).
- `BGE-small-en-v1.5` 384-dimensional dense vector embeddings.
- PostgreSQL `pgvector` index and sub-second cosine similarity search (`retrieval_service.py`).
- `KnowledgeEngine`: Qwen3-4B factual claim extraction, named entity linking, and pre-compiled `orchestrator_prompt_context`.

---

### Phase 4: Content Orchestrator & Generation Engine (Completed)
- `IntentAndPersonalization` schema capturing user target persona, tone, language, objective, detail level, and focus keywords.
- Format-specific static prompt builder (`prompt_builder.py`).
- `Qwen3-8B Q4` GGUF generation service with dynamic VRAM-aware GPU layer allocation ($\ge 5.5\text{ GB}$ full GPU, partial, or CPU fallback).
- Multi-format fanout: sequential generation of LinkedIn Posts, Twitter Threads, Executive Summaries, Presentation Decks, Infographic Briefs, Video Scripts, and Blog Posts.

---

### Phase 5: Trust, Validation & Schema Enforcement Layer (Completed)
- **Fact Checker & Grounding Analysis**: Sentence-level BGE cosine similarity matching against `KnowledgePackage` evidence, producing a Trust Score ($0.0 \text{--} 1.0$) and claim verifications.
- **Schema Validator**: Rule engine checking LinkedIn word bounds, Twitter 280-char tweet limits, Slide counts, and Video scene rules.
- **Automated Repair Loop**: Qwen3-4B targeted repair loop (max 2 retries) with hard truncation fallbacks.
- Master `TrustService` integrated directly into `OrchestratorService` pipeline.

---

### Phase 6: Multi-Format Output Export Layer (Completed)
- `DocxFormatter`: Generates Word `.docx` documents using `python-docx`.
- `PptxFormatter`: Generates 16:9 widescreen PowerPoint `.pptx` decks with speaker notes using `python-pptx`.
- `PdfFormatter`: Generates publication-ready PDFs using ReportLab and PyMuPDF.
- `VideoPackageBuilder`: Generates Video Deliverable `.zip` containing `.srt` subtitles, `script_storyboard.json`, and visual cues markdown.
- `InfographicPackageBuilder`: Generates Infographic Deliverable `.zip` containing `render_template.html` interactive preview and key metrics summary.
- Export REST endpoint: `POST /api/v1/export/download`.
