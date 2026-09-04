# Architecture Implementation Roadmap: AI-Powered Content Transformation Engine (SIH-26154)

This roadmap outlines how every future AI module interfaces with the **Semantic Document JSON System Contract** built in Phase 1.

---

## The Contract-Centric Architecture

```
Raw Multimodal Input (PDF / DOCX)
                ↓
    [ Phase 1: Semantic Document Processing ]
                ↓
     ⭐ Semantic Document JSON Contract ⭐
                │
    ┌───────────┼───────────┬───────────┐
    ↓           ↓           ↓           ↓
 Phase 2     Phase 3     Phase 4     Phase 5-8
 Qwen2.5-VL  Knowledge   Intent &    Orchestrator
 Visual      Engine &    Persona     & Generative
 Intel       pgvector    Settings    Multi-Outputs
```

---

## Phase Breakdown

### Phase 1: Document Processing & Semantic Foundation (Completed)
- **Scope**:
  - FastAPI web server and dependency container.
  - Multi-page PDF rasterization (PyMuPDF) and DOCX parsing.
  - PP-StructureV3 layout analysis, table recognition, and OCR (with automatic fallback).
  - Multi-modal extraction: figure/chart visual cropping and storage.
  - Semantic Fusion Engine: reading order reconciliation and caption linking.
  - Assembly & validation of `SemanticDocument` JSON.
  - PostgreSQL 16 persistence (JSONB document + relational elements).

---

### Phase 2: Visual Intelligence (Qwen2.5-VL-3B Q4)
- **Objective**: Multimodal visual understanding of diagrams, charts, infographics, and flowcharts.
- **Contract Interface**:
  - Filters `SemanticDocument.elements` where `type IN ('figure', 'chart', 'image')`.
  - Reads `image_path` from element content.
  - Inferences using Qwen2.5-VL-3B (quantized Q4 via Ollama / llama.cpp / vLLM).
  - Injects generated descriptions, data trends, diagram flows, and key takeaways into `element.content.raw_attributes["visual_analysis"]`.

---

### Phase 3: Embeddings + pgvector + Knowledge Engine
- **Objective**: Semantic search, knowledge graph formation, and factual claim extraction.
- **Contract Interface**:
  - **Embedding Engine**:
    - Chunks text and table elements (`BGE-large-en-v1.5` or `BGE-M3`).
    - Stores vector embeddings in PostgreSQL `pgvector` table linked by `element_id`.
  - **Knowledge Engine**:
    - Extracts named entities -> populates `SemanticDocument.entities`.
    - Extracts factual propositions -> populates `SemanticDocument.claims`.
    - Maps entity triples -> populates `SemanticDocument.relationships`.
    - Tracks source element IDs for end-to-end provenance.

---

### Phase 4: Intent & Personalization Engine
- **Objective**: Translating user configuration into structured content generation parameters.
- **Inputs**:
  - User Configuration: Output Selection (LinkedIn, Exec Summary, PPT, Video), Target Audience, Tone, Language, Style, Detail Level, Objective.
  - Target `document_id`.
- **Outputs**:
  - Execution Plan specifying which semantic elements, claims, and visual insights are prioritized.

---

### Phase 5: Content Orchestrator & Prompt Builder
- **Objective**: Dynamic prompt construction and context window optimization.
- **Contract Interface**:
  - Retrieves prioritized elements, claims, and tables from `SemanticDocument`.
  - Compiles structured prompts containing exact source citations.
  - Coordinates multi-output generation controllers.

---

### Phase 6: Main LLM Integration (Ollama Hosted)
- **Objective**: High-reasoning generation across chosen output targets.
- **Tech**: Ollama (e.g. Llama 3.1 / Qwen 2.5 / DeepSeek).
- **Execution**: Reasoning, synthesis, and creative drafting tailored to the persona.

---

### Phase 7: Guardrails, Validation & Schema Enforcement
- **Objective**: Anti-hallucination, fact verification, and strict schema compliance.
- **Features**:
  - Cross-references generated statements against `SemanticDocument.claims` and `sources`.
  - Hallucination score and confidence metric.
  - Strict JSON schema enforcement for downstream presentation builders.

---

### Phase 8: Multi-Format Output Generation
- **Target Formats**:
  1. **LinkedIn Post**: Engaging copy with hashtags, takeaways, and call-to-action.
  2. **Twitter / X Thread**: Bite-sized insight sequence with data highlights.
  3. **Executive Summary**: Structured PDF / DOCX advisory report.
  4. **Presentation Deck**: Slide outline with titles, bullet points, speaker notes, and diagram references (PPTX / HTML).
  5. **Infographic Package**: Visual design blueprint and key data callouts.
  6. **Video Script Package**: Scene-by-scene storyboard, narration, subtitles, and visual suggestions.
