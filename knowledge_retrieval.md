# Knowledge & Retrieval Layer — Technical Reference

> **Layer**: Phase 3 — Knowledge & Retrieval
> **Status**: ✅ Complete & Verified in Docker
> **Updated**: September 6, 2026
> **Consumed by**: Phase 4 — Content Orchestrator (multi-format output generation)

---

## 1. Purpose

The Knowledge & Retrieval Layer sits between the **Semantic Document JSON** (produced by Phase 2 document parsing) and the **Content Orchestrator** (Phase 4 multi-format generation). Its responsibility is to:

1. Transform raw semantic elements into dense vector embeddings and persist them in pgvector
2. Retrieve the most relevant document sections for any given user intent via cosine similarity search
3. Extract structured knowledge (entities, claims, metrics, tables, visual insights) from retrieved context
4. Assemble a complete, format-ready **KnowledgePackage** that the Content Orchestrator can consume directly without any further document access

The Content Orchestrator **never reads the raw Semantic Document JSON directly** — it only consumes the KnowledgePackage output from this layer.

---

## 2. Data Flow

```
Semantic Document JSON (stored in documents.semantic_json JSONB)
         │
         ▼
┌─────────────────────────────────────┐
│  TEXT CLEANING (text_cleaner.py)    │
│  Unicode NFKC normalization,        │
│  whitespace collapsing, dehyphen-   │
│  ation, control char removal        │
└──────────────┬──────────────────────┘
               │ cleaned text
               ▼
┌─────────────────────────────────────┐
│  SEMANTIC CHUNKER (chunker.py)      │
│  Structure-aware splitting per      │
│  element type (text / table /       │
│  visual). Adds context prefix:      │
│  [Doc: {title} | Page {p} | {type}] │
└──────────────┬──────────────────────┘
               │ chunks[]
               ▼
┌─────────────────────────────────────┐
│  BGE EMBEDDING ENGINE               │
│  (bge_initializer.py)               │
│  model: bge_small_en_v1.5           │
│  backend: transformers (not         │
│  sentence_transformers)             │
│  output: 384-dim float32 vectors    │
│  query prefix: "Represent this      │
│  sentence for searching relevant    │
│  passages: "                        │
└──────────────┬──────────────────────┘
               │ Vector(384) per chunk
               ▼
┌─────────────────────────────────────┐
│  pgvector STORAGE                   │
│  table: document_chunks             │
│  DB: PostgreSQL 16 (sih_postgres)   │
│  extension: vector (pgvector)       │
└──────────────┬──────────────────────┘
               │                     ▲
               │ (at assembly time)  │ stored embeddings
               ▼                     │
┌─────────────────────────────────────┐
│  RETRIEVAL SERVICE                  │
│  (retrieval_service.py)             │
│  - encodes user query with BGE      │
│  - computes cosine similarity in    │
│    Python against all stored vecs   │
│  - filters by document_id /         │
│    chunk_type / page_range          │
│  - returns top_k ranked chunks      │
└──────────────┬──────────────────────┘
               │ RetrievedChunk[]
               ▼
┌─────────────────────────────────────┐
│  KNOWLEDGE ENGINE                   │
│  (knowledge_engine.py)              │
│  - parses SemanticDocument for      │
│    tables + visual elements         │
│  - extracts entities, claims,       │
│    metrics via Qwen3-4B GGUF        │
│    (falls back to deterministic     │
│    extraction on timeout/error)     │
│  - compiles orchestrator_prompt_    │
│    context markdown block           │
└──────────────┬──────────────────────┘
               │
               ▼
        KnowledgePackage
    (input to Content Orchestrator)
```

---

## 3. Sub-Components

### 3.1 Text Cleaner (`backend/app/services/embedding/text_cleaner.py`)

Normalizes raw element text before chunking.

| Operation | Detail |
|-----------|--------|
| Unicode normalization | NFKC — collapses ligatures, full-width chars |
| Whitespace | Multiple spaces/tabs → single space; multiple newlines → max 2 |
| Dehyphenation | `word-\n` → `word` (PDF line-break artifacts) |
| Control chars | Strips `\x00–\x1f` except `\n \t` |
| Markdown table | Ensures pipes are space-padded and header separator exists |

### 3.2 Semantic Chunker (`backend/app/services/embedding/chunker.py`)

Splits cleaned text into fixed-overlap chunks appropriate per element type.

| Element type | Strategy |
|-------------|---------|
| `text` / `heading` / `paragraph` | Sentence-boundary split, 512 char max, 64 char overlap |
| `table` | Header row preserved on every split; each chunk = header + N data rows |
| `figure` / `chart` / `image` | Caption + visual_analysis description as single chunk |

Every chunk gets a **context prefix** prepended before embedding:
```
[Doc: {document_title} | Page {page_number} | Type: {element_type}]
{chunk_text}
```
This prefix improves BGE retrieval accuracy by anchoring each chunk to its document context.

### 3.3 BGE Embedding Engine (`backend/app/services/model_initializer/bge_initializer.py`)

| Property | Value |
|----------|-------|
| Model | `BAAI/bge-small-en-v1.5` |
| Local path | `models/bge_small_en_v1.5/` |
| Backend | `transformers` (AutoTokenizer + AutoModel) |
| Dimension | 384 |
| Pooling | Mean pooling of last hidden state |
| Query prefix | `Represent this sentence for searching relevant passages: ` |
| Fallback | Deterministic hash-based pseudo-vector (for local tests without Docker) |

`sentence_transformers` is **not** used — the raw `transformers` library is called directly (already in Docker for other models). Both approaches produce identical 384-dim embeddings.

### 3.4 Embedding Service (`backend/app/services/embedding/embedding_service.py`)

Orchestrates the full pipeline: clean → chunk → encode → persist.

- Deletes all existing chunks for the document before re-embedding (idempotent, safe to call multiple times)
- Batch-inserts all `DocumentChunk` rows in a single transaction
- Returns the list of created chunks

### 3.5 pgvector Storage (`backend/app/models/document_chunk.py`)

**Table**: `document_chunks`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `document_id` | UUID FK | References `documents.id` |
| `element_id` | String | Source element ID from SemanticDocument |
| `chunk_index` | Integer | Sequential index within document |
| `chunk_type` | Enum | `text` / `table` / `visual` |
| `page` | Integer | Source page number |
| `content` | Text | Original (un-prefixed) chunk text |
| `cleaned_text` | Text | Cleaned version of content |
| `chunk_metadata` | JSONB | Element type, title, reading_order etc. |
| `embedding` | Vector(384) | BGE dense vector |

> **Critical**: Uses a custom `Vector(384)` SQLAlchemy type with `bind_processor` (list → `"[x,y,z]"` string) and `result_processor` (string → list). The official `pgvector.sqlalchemy.Vector` does **not** work with asyncpg. Do not replace it.

### 3.6 Retrieval Service (`backend/app/services/retrieval_service.py`)

Performs semantic search. Computes cosine similarity in Python (not SQL `ORDER BY <=>`) for asyncpg compatibility.

```
score = dot(query_vec, chunk_vec) / (norm(query_vec) * norm(chunk_vec))
```

Supported filters:
- `document_id` — restrict to single document
- `chunk_types` — filter by `ChunkType` enum
- `page_range` — restrict to page range `(min, max)`
- `top_k` — max results (default 5)
- `min_similarity` — minimum cosine score threshold (default 0.0)

### 3.7 Knowledge Engine (`backend/app/services/knowledge_engine.py`)

Main assembly coordinator. Called once per user request.

**Execution order:**
1. Parse `document.semantic_json` → `SemanticDocument`
2. Build search query from `intent.objective + focus_keywords`
3. Retrieve `top_k=8` relevant chunks via `retrieval_service.search()`
4. If no chunks in DB → generate fallback chunks directly from `SemanticDocument.elements`
5. Extract `TableSummaryItem[]` from all `type=table` elements
6. Extract `VisualInsightItem[]` from `type=figure/chart/image` elements (converts dict/list `visual_analysis` to string)
7. Build `EvidenceItem[]` from retrieved chunks
8. Run **Qwen3-4B reasoning** (if available) → extract entities, claims, metrics, strategy
9. If Qwen3-4B fails/times out → **deterministic extraction** fallback
10. Compile `orchestrator_prompt_context` markdown block
11. Return `KnowledgePackage`

**Qwen3-4B path** (Docker, model present):
- Loads `models/qwen3_4b_q4/Qwen3-4B-Q4_K_M.gguf` via `llama-cpp-python`
- Cold start: ~30–60s on CPU (first call only; kept in memory after)
- `max_tokens=300`, `temperature=0.1`
- Output: JSON with `claims[]`, `metrics[]`, `strategy` fields

**Deterministic fallback** (always available, no model needed):
- Claims → first sentence of each top-5 evidence chunk
- Metrics → regex extraction of numeric patterns from evidence text
- Strategy → format-mapped structure templates per `output_type`
- Entities → capitalization-pattern heuristic NER from evidence

---

## 4. API Endpoints

Base: `/api/v1/knowledge/`

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/embed/{document_id}` | Trigger chunking + BGE embedding for a document |
| `GET` | `/search` | Semantic search — individual Swagger input fields per param |
| `POST` | `/search` | Semantic search via JSON body |
| `POST` | `/assemble/{document_id}` | Assemble KnowledgePackage (path param + optional body) |
| `POST` | `/assemble` | Assemble KnowledgePackage (full JSON body with intent) |

### `/assemble/{document_id}` optional body

```json
{
  "output_type": "executive_summary",
  "audience": "executive",
  "tone": "professional",
  "language": "English",
  "objective": "Summarize key insights, data points, and recommendations.",
  "detail_level": "moderate",
  "focus_keywords": [],
  "custom_instructions": null,
  "use_llm": false
}
```

Set `use_llm: false` to skip Qwen3-4B for instant deterministic results. `use_llm: true` (default) enables full reasoning (requires cold-start on first call in new container lifecycle).

---

## 5. KnowledgePackage — Complete Output Contract

**Schema**: `backend/app/schemas/knowledge_package.py`
**Intent schema**: `backend/app/schemas/intent.py`

This is the **exact payload** consumed by the Content Orchestrator. It is fully self-contained — the orchestrator does not need to access any other service or database.

```typescript
KnowledgePackage {
  document_id: string                   // UUID of source document
  document_title: string                // From SemanticDocument.metadata.title or filename

  intent: IntentAndPersonalization {
    document_id: string
    output_type: OutputType             // "linkedin_post" | "twitter_thread" |
                                        // "executive_summary" | "presentation_deck" |
                                        // "infographic_brief" | "video_script"
    audience: AudienceType             // "general" | "student" | "professional" |
                                        // "executive" | "technical" | "investor"
    tone: ToneType                     // "professional" | "casual" | "inspirational" |
                                        // "analytical" | "persuasive" | "educational"
    language: string                   // e.g. "English"
    objective: string                  // core goal / thesis for content generation
    detail_level: DetailLevel          // "brief" | "moderate" | "comprehensive"
    focus_keywords: string[]           // priority topics to emphasize
    custom_instructions?: string       // optional free-text directives
  }

  retrieved_evidence: EvidenceItem[] {
    chunk_id: string
    element_id?: string                // source element ID in SemanticDocument
    page: int
    chunk_type: string                 // "text" | "table" | "visual"
    text: string                       // actual chunk content
    relevance_score: float             // cosine similarity 0.0–1.0
  }[]

  entities: EntityItem[] {
    id: string
    name: string
    category: string                   // e.g. "CONCEPT", "ORG", "METRIC"
    mentions: string[]                 // element_ids where this entity appears
    confidence: float
  }[]

  claims: ClaimItem[] {
    id: string                         // e.g. "claim_1"
    statement: string                  // factual proposition (single sentence)
    source_element_ids: string[]       // provenance — element IDs from SemanticDocument
    confidence: float                  // 0.0–1.0
  }[]

  relationships: RelationshipItem[]    // entity-to-entity triples (from SemanticDocument)

  key_metrics: KeyMetricItem[] {
    label: string                      // e.g. "Data Point 1"
    value: string                      // e.g. "42%", "$1.2 billion"
    context: string                    // surrounding text phrase (~60 chars)
    source_element_id?: string
    page?: int
  }[]

  tables: TableSummaryItem[] {
    element_id: string
    page: int
    caption?: string
    markdown_table: string             // GFM markdown table ready for injection
    key_takeaway: string
  }[]

  visual_insights: VisualInsightItem[] {
    element_id: string
    page: int
    element_type: string               // "figure" | "chart" | "image"
    caption?: string
    image_path?: string                // relative path under uploads/extracted/
    takeaway: string                   // always a plain string (dict/list auto-converted)
  }[]

  strategy: ContentStrategy {
    headline_hook: string              // suggested title / opening hook
    key_themes: string[]              // core themes to thread through the content
    suggested_structure: string[]     // ordered section names for the target format
    recommended_cta: string           // closing call-to-action
    tone_guidelines?: string          // tone + audience delivery directive
  }

  orchestrator_prompt_context: string  // pre-compiled dense markdown block (see §6)

  metadata: {
    retrieval_count: int               // number of chunks retrieved
    tables_count: int
    visuals_count: int
    claims_count: int
    entities_count: int
    processing_time_seconds: float
    qwen3_4b_available: bool
  }
}
```

---

## 6. `orchestrator_prompt_context` — Pre-compiled Markdown Block

This is the primary field the Content Orchestrator uses to build its generation prompt. It is a single dense markdown string with the following structure:

```markdown
# Knowledge Context for Content Orchestration
**Document**: {document_title}
**Target Format**: {output_type} | **Audience**: {audience} | **Tone**: {tone}
**Core Objective**: {objective}

## 1. Content Strategy Blueprint
- **Suggested Hook / Title**: {headline_hook}
- **Tone & Style Directive**: {tone_guidelines}
- **Suggested Section Structure**:
  - {section_1}
  - {section_2}
  ...
- **Recommended Call to Action**: {cta}

## 2. Core Verified Claims & Factual Propositions
- **claim_1**: {statement} [Source: {element_id}]
...

## 3. Key Quantitative Metrics & Evidence
- **{value}** ({label}): "...{context}..." [Source: {element_id}, Page {page}]
...

## 4. Structured Tables
### Table on Page {page} (Element: {element_id})
*Caption: {caption}*
{markdown_table}

## 5. Visual Insights & Chart Interpretations
- **{element_type} (Element: {element_id}, Page {page})**: {takeaway}
  *Image asset*: `{image_path}`

## 6. Retrieved Semantic Context Passages
### [Element: {element_id} | Page {page} | Relevance: {score}]
{chunk_text}
```

---

## 7. How the Content Orchestrator Should Consume This

The Content Orchestrator receives a `KnowledgePackage` and selects a format-specific prompt template based on `intent.output_type`. It should:

1. **Use `orchestrator_prompt_context`** as the primary context block injected into the generation prompt — OR cherry-pick specific fields for tighter token usage on short-form formats
2. **Use `intent.output_type`** to select the correct format prompt template
3. **Use `strategy.suggested_structure`** as the section scaffold
4. **Use `strategy.headline_hook`** as the opening / title candidate
5. **Use `claims[]`** for factual anchors with provenance citations
6. **Use `key_metrics[]`** for numeric/statistical emphasis
7. **Use `tables[]`** for markdown tables to embed or reference
8. **Use `visual_insights[]`** for image paths and chart descriptions
9. **Use `intent.tone` + `strategy.tone_guidelines`** for style enforcement
10. **Use `strategy.recommended_cta`** as the closing directive

> Format-specific constraints (word limits, tweet character limits, slide counts, hashtag rules) are the **Content Orchestrator's responsibility** — they belong in the per-format prompt templates, not in the KnowledgePackage.

---

## 8. `strategy.suggested_structure` by Output Type

Pre-populated by the deterministic path. Qwen3-4B may return richer custom structures.

| `output_type` | `suggested_structure` |
|---------------|-----------------------|
| `linkedin_post` | Attention Grabber / Hook → Key Problem / Data Context → Core Breakthrough / Finding → Actionable Takeaway → Call-to-Action & Hashtags |
| `twitter_thread` | 1/ Hook & Context → 2/ The Core Problem → 3/ Key Data & Stats → 4/ Solution / Insight → 5/ Summary & Takeaway |
| `executive_summary` | Executive Overview → Strategic Context → Key Findings & Quantitative Evidence → Risk & Opportunity Analysis → Recommendations |
| `presentation_deck` | Slide 1: Title & Agenda → Slide 2: Background & Problem → Slide 3: Key Data & Metrics → Slide 4: Strategic Recommendations → Slide 5: Q&A / Next Steps |
| `infographic_brief` | Header & Focal Stat → Key Data Comparison (Table/Chart) → Process / Flow Breakdown → Core Callouts → Source Citations |
| `video_script` | Scene 1: Visual Hook & Intro → Scene 2: Problem Statement → Scene 3: Deep Dive into Insights → Scene 4: Key Takeaway & Closing |

---

## 9. File Map

```
backend/app/
├── services/
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py          # Unicode/whitespace normalization
│   │   ├── chunker.py               # Structure-aware semantic chunker
│   │   └── embedding_service.py     # Clean → chunk → embed → persist pipeline
│   ├── model_initializer/
│   │   ├── bge_initializer.py       # BGE lazy loader (transformers backend)
│   │   └── __init__.py
│   ├── retrieval_service.py         # Cosine similarity vector search
│   └── knowledge_engine.py          # Knowledge assembly + Qwen3-4B / deterministic
├── models/
│   └── document_chunk.py            # DocumentChunk ORM + custom Vector(384) type
├── schemas/
│   ├── intent.py                    # IntentAndPersonalization + all enums
│   └── knowledge_package.py         # KnowledgePackage + all sub-item schemas
└── api/v1/endpoints/
    └── knowledge.py                 # 5 REST endpoints: embed / search (GET+POST) / assemble (path+body)
```

---

## 10. Known Constraints & Gotchas

| Issue | Detail |
|-------|--------|
| Qwen3-4B cold start | First `/assemble` call takes 30–60s on CPU while GGUF loads. Use `use_llm: false` for instant results. After first load, model stays resident in memory. |
| Custom Vector type | Must use the custom `Vector(384)` class in `document_chunk.py`. The official `pgvector.sqlalchemy.Vector` breaks with asyncpg — sends Python list instead of the required `"[x,y,z]"` string format. |
| `pgvector` pip package | Hot-installed via `docker exec sih_backend pip install pgvector>=0.3.0`. Now in `requirements.txt` for future image rebuilds. Does not need a Docker rebuild — just `docker exec pip install`. |
| BGE backend | `sentence_transformers` is NOT installed in Docker. BGE uses raw `transformers` (AutoTokenizer + AutoModel + mean pooling). Produces identical 384-dim output. |
| Search returns `[]` | Most common cause: Swagger `document_id` placeholder left as literal `"string"`. Always use real UUID. Also ensure `/embed/{document_id}` was called first to populate `document_chunks`. |
| Embedding is idempotent | Calling `/embed/{document_id}` again deletes and recreates all chunks for that document. Safe to re-call. |
| New pip packages | Never requires a full Docker rebuild. `docker exec sih_backend pip install <pkg>` + `docker restart sih_backend` is sufficient. Only OS-level apt packages or Dockerfile changes need a rebuild. |
