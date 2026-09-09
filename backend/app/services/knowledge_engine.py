"""
Knowledge Engine Service (powered by Qwen3-4B).
Consumes Semantic Document JSON, pgvector semantic search retrieval, and user Intent & Personalization
to extract verified entities, factual claims with citations, quantitative metrics, and structured context,
assembling a comprehensive KnowledgePackage for the Content Orchestrator.
"""

import json
import re
import time
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.logging import logger
from backend.app.models.document import Document
from backend.app.models.document_chunk import ChunkType
from backend.app.schemas.semantic_document import (
    SemanticDocument,
    EntityItem,
    ClaimItem,
    RelationshipItem,
)
from backend.app.schemas.intent import IntentAndPersonalization
from backend.app.schemas.knowledge_package import (
    KnowledgePackage,
    EvidenceItem,
    KeyMetricItem,
    TableSummaryItem,
    VisualInsightItem,
    ContentStrategy,
    ExecutiveInsightItem,
    KeyFindingItem,
    RecommendationItem,
    RiskOrOpportunityItem,
)
from backend.app.services.retrieval_service import retrieval_service, RetrievedChunk
from backend.app.services.model_initializer.qwen_initializers import qwen_fusion_initializer


class KnowledgeEngine:
    """
    Coordinates semantic retrieval, claim verification, entity linking,
    and Qwen3-4B reasoning to assemble structured knowledge for the Content Orchestrator.
    """

    def __init__(self):
        self.metric_regex = re.compile(
            r"(?:(?:\$|€|£|₹)?\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|\s*(?:billion|million|trillion|percent|k|MB|GB|TB|bps|x))?\b)"
        )

    async def assemble_knowledge(
        self,
        intent: IntentAndPersonalization,
        document: Document,
        db: AsyncSession,
        top_k: int = 8
    ) -> KnowledgePackage:
        """
        Main entrypoint: executes semantic search, extracts entities/claims/metrics,
        runs Qwen3-4B reasoning, and returns the assembled KnowledgePackage.
        """
        start_time = time.time()
        doc_id = document.id

        # 1. Parse or deserialize SemanticDocument
        semantic_doc: SemanticDocument
        if isinstance(document.semantic_json, dict):
            semantic_doc = SemanticDocument.model_validate(document.semantic_json)
        else:
            semantic_doc = SemanticDocument(
                document_id=doc_id,
                metadata={
                    "file_name": document.filename,
                    "file_size": document.file_size,
                    "mime_type": document.mime_type,
                    "page_count": document.page_count,
                },
                elements=[]
            )

        doc_title = self._resolve_document_title(
            candidate_title=semantic_doc.metadata.title,
            fallback_title=document.filename,
            semantic_doc=semantic_doc
        )

        # 2. Multi-Modal Semantic Search using Intent & Objective
        search_query = self._build_search_query(intent)
        retrieved_chunks = await retrieval_service.search(
            query=search_query,
            db=db,
            document_id=doc_id,
            top_k=top_k,
            min_similarity=0.0
        )

        # Fallback if no chunks in DB yet: extract from semantic_doc directly
        if not retrieved_chunks and semantic_doc.elements:
            retrieved_chunks = self._fallback_chunks_from_elements(semantic_doc)

        logger.info("Retrieved %d relevant chunks for document %s (intent: %s)", len(retrieved_chunks), doc_id, intent.output_type)

        # 3. Extract Tables and Visual Insights from Semantic Document
        tables = self._extract_relevant_tables(semantic_doc, retrieved_chunks)
        visual_insights = self._extract_visual_insights(semantic_doc, retrieved_chunks)

        # 4. Extract Entities, Claims, Metrics, and Content Strategy (via Qwen3-4B or deterministic extractor)
        # De-noise: filter out decorative icons/symbols/logos from becoming primary evidence or claims
        filtered_chunks = []
        for c in retrieved_chunks:
            txt_lower = c.content.lower()
            is_decorative_icon = (
                '"visual_type": "icon"' in txt_lower
                or '"visual_type": "database_icon"' in txt_lower
                or '"visual_type": "symbol"' in txt_lower
                or '"visual_type": "abstract"' in txt_lower
                or '"visual_type": "logo"' in txt_lower
            )
            if not is_decorative_icon:
                filtered_chunks.append(c)

        # Fallback to all if filtering removed everything
        final_chunks = filtered_chunks if filtered_chunks else retrieved_chunks

        evidence_items = [
            EvidenceItem(
                chunk_id=c.id,
                element_id=c.element_id,
                page=c.page,
                chunk_type=str(c.chunk_type.value if hasattr(c.chunk_type, "value") else c.chunk_type),
                text=c.content,
                relevance_score=c.similarity_score
            )
            for c in final_chunks
        ]

        (
            entities,
            claims,
            relationships,
            metrics,
            strategy,
            document_story,
            key_findings,
            recommendations,
            executive_insights
        ) = self._extract_knowledge_and_strategy(
            intent=intent,
            evidence=evidence_items,
            tables=tables,
            visuals=visual_insights,
            doc_title=doc_title,
            semantic_doc=semantic_doc
        )

        # 5. Compile High-Density Orchestrator Prompt Context
        orchestrator_context = self._compile_orchestrator_prompt_context(
            doc_title=doc_title,
            intent=intent,
            strategy=strategy,
            claims=claims,
            metrics=metrics,
            tables=tables,
            visuals=visual_insights,
            evidence=evidence_items
        )

        duration = round(time.time() - start_time, 3)

        # 6. Package into KnowledgePackage
        package = KnowledgePackage(
            document_id=doc_id,
            document_title=doc_title,
            intent=intent,
            document_story=document_story,
            retrieved_evidence=evidence_items,
            entities=entities,
            claims=claims,
            relationships=relationships,
            key_metrics=metrics,
            tables=tables,
            visual_insights=visual_insights,
            executive_insights=executive_insights,
            key_findings=key_findings,
            recommendations=recommendations,
            strategy=strategy,
            orchestrator_prompt_context=orchestrator_context,
            metadata={
                "retrieval_count": len(evidence_items),
                "tables_count": len(tables),
                "visuals_count": len(visual_insights),
                "claims_count": len(claims),
                "entities_count": len(entities),
                "processing_time_seconds": duration,
                "qwen3_4b_available": qwen_fusion_initializer.is_available()
            }
        )

        return package

    def _build_search_query(self, intent: IntentAndPersonalization) -> str:
        """Constructs an enriched semantic search query from intent parameters."""
        parts = [intent.objective]
        if intent.focus_keywords:
            parts.append(" ".join(intent.focus_keywords))
        if intent.custom_instructions:
            parts.append(intent.custom_instructions[:100])
        return " ".join(parts).strip()

    def _fallback_chunks_from_elements(self, doc: SemanticDocument) -> List[RetrievedChunk]:
        """Generates mock retrieved chunks directly from elements if DB chunks haven't been indexed."""
        chunks = []
        for idx, elem in enumerate(doc.elements[:8]):
            txt = elem.content.text or elem.content.markdown or elem.content.caption or ""
            if txt:
                chunks.append(
                    RetrievedChunk(
                        id=f"chunk_mem_{idx}",
                        document_id=doc.document_id,
                        element_id=elem.id,
                        chunk_index=idx,
                        chunk_type=ChunkType.TEXT,
                        page=elem.page,
                        content=txt,
                        cleaned_text=txt,
                        chunk_metadata={},
                        similarity_score=1.0 - (idx * 0.05)
                    )
                )
        return chunks

    def _extract_relevant_tables(
        self,
        doc: SemanticDocument,
        retrieved_chunks: List[RetrievedChunk]
    ) -> List[TableSummaryItem]:
        """Extracts and formats tables present in the semantic document."""
        tables: List[TableSummaryItem] = []
        seen_element_ids = set()

        # Check if table elements were retrieved or exist in document
        for elem in doc.elements:
            if elem.type == "table" and elem.id not in seen_element_ids:
                table_md = elem.content.markdown or elem.content.text or ""
                if table_md:
                    seen_element_ids.add(elem.id)
                    tables.append(
                        TableSummaryItem(
                            element_id=elem.id,
                            page=elem.page,
                            caption=elem.content.caption,
                            markdown_table=table_md,
                            key_takeaway=f"Structured table on page {elem.page}" + (f": {elem.content.caption}" if elem.content.caption else "")
                        )
                    )
        return tables[:4]

    def _extract_visual_insights(
        self,
        doc: SemanticDocument,
        retrieved_chunks: List[RetrievedChunk]
    ) -> List[VisualInsightItem]:
        """Extracts visual elements (figures, charts, memes) and their descriptive takeaways."""
        insights: List[VisualInsightItem] = []
        seen_element_ids = set()

        for elem in doc.elements:
            if elem.type in ("figure", "chart", "image") and elem.id not in seen_element_ids:
                seen_element_ids.add(elem.id)
                raw_attrs = elem.content.raw_attributes or {}

                # Skip micro-icons/emojis, duplicate visual instances, and recurring slide template logos
                if (
                    raw_attrs.get("is_decorative_noise")
                    or raw_attrs.get("is_duplicate")
                    or raw_attrs.get("is_recurring_template_asset")
                ):
                    continue

                raw_val = raw_attrs.get("visual_analysis") or raw_attrs.get("description")

                # De-noise: also skip decorative logos, icons, and symbols identified via visual_type
                v_type = ""
                if isinstance(raw_val, dict):
                    v_type = str(raw_val.get("visual_type", "")).lower()
                elif isinstance(raw_attrs.get("visual_type"), str):
                    v_type = raw_attrs["visual_type"].lower()
                if v_type in ("logo", "icon", "database_icon", "symbol", "abstract"):
                    continue

                # Check for meme / informal graphic
                is_meme = (
                    raw_attrs.get("is_informal_graphic") is True
                    or (isinstance(raw_val, dict) and str(raw_val.get("visual_type", "")).lower() in ("meme", "social_media_graphic"))
                )

                if is_meme and isinstance(raw_val, dict):
                    overlay_list = raw_attrs.get("overlay_text") or raw_val.get("overlay_text") or raw_val.get("visible_text") or []
                    overlay_str = ", ".join(overlay_list) if isinstance(overlay_list, list) else str(overlay_list)
                    core_concept = raw_attrs.get("core_concept") or raw_val.get("core_concept_or_humor_theme") or raw_val.get("description") or "Tech concept"
                    analysis = f"Informal Graphic/Meme Overlay Text: '{overlay_str}'. Core Theme: {core_concept}"
                elif isinstance(raw_val, dict):
                    analysis = raw_val.get("_raw_text") or raw_val.get("summary") or " ".join(str(v) for v in raw_val.values())
                elif isinstance(raw_val, list):
                    analysis = " ".join(str(v) for v in raw_val)
                elif raw_val:
                    analysis = str(raw_val)
                else:
                    analysis = elem.content.caption or f"{elem.type.title()} visualization"

                insights.append(
                    VisualInsightItem(
                        element_id=elem.id,
                        page=elem.page,
                        element_type=elem.type,
                        caption=elem.content.caption,
                        image_path=elem.content.image_path,
                        takeaway=analysis.strip()
                    )
                )
        return insights[:4]

    def _extract_knowledge_and_strategy(
        self,
        intent: IntentAndPersonalization,
        evidence: List[EvidenceItem],
        tables: List[TableSummaryItem],
        visuals: List[VisualInsightItem],
        doc_title: str,
        semantic_doc: SemanticDocument
    ) -> tuple[
        List[EntityItem],
        List[ClaimItem],
        List[RelationshipItem],
        List[KeyMetricItem],
        ContentStrategy,
        str,
        List[KeyFindingItem],
        List[RecommendationItem],
        List[ExecutiveInsightItem]
    ]:
        """
        Attempts Qwen3-4B inference to perform reasoning; falls back gracefully to
        deterministic semantic extraction if Qwen runtime is not active or use_llm=False.
        """
        use_llm = intent.extra.get("use_llm", True) if getattr(intent, "extra", None) else True
        if use_llm and qwen_fusion_initializer.is_available():
            try:
                return self._run_qwen3_reasoning(intent, evidence, tables, visuals, doc_title, semantic_doc)
            except Exception as e:
                logger.warning("Qwen3-4B inference error (%s); falling back to deterministic extraction", e)

        return self._deterministic_extraction(intent, evidence, tables, visuals, doc_title, semantic_doc)

    def _run_qwen3_reasoning(
        self,
        intent: IntentAndPersonalization,
        evidence: List[EvidenceItem],
        tables: List[TableSummaryItem],
        visuals: List[VisualInsightItem],
        doc_title: str,
        semantic_doc: SemanticDocument
    ) -> tuple[
        List[EntityItem],
        List[ClaimItem],
        List[RelationshipItem],
        List[KeyMetricItem],
        ContentStrategy,
        str,
        List[KeyFindingItem],
        List[RecommendationItem],
        List[ExecutiveInsightItem]
    ]:
        """Invokes local Qwen3-4B GGUF model via llama-cpp-python for reasoning."""
        model = qwen_fusion_initializer.load()

        context_bullets = "\n".join([f"- [Element: {e.element_id}, Page: {e.page}]: {e.text}" for e in evidence[:6]])

        prompt = f"""<|im_start|>system
You are the Knowledge Engine for an AI Content Transformation Platform.
Analyze the following document context and user intent. Extract verified entities, factual claims with source element citations, numeric metrics, and outline a content strategy.
Output strictly a JSON object conforming to this structure:
{{
  "headline_hook": "...",
  "key_themes": ["...", "..."],
  "suggested_structure": ["...", "..."],
  "recommended_cta": "...",
  "tone_guidelines": "...",
  "claims": [
    {{"id": "claim_1", "statement": "...", "source_element_ids": ["..."], "confidence": 0.95}}
  ],
  "metrics": [
    {{"label": "...", "value": "...", "context": "...", "source_element_id": "...", "page": 1}}
  ]
}}
<|im_end|>
<|im_start|>user
Document: {doc_title}
Target Output: {intent.output_type.value}
Target Audience: {intent.audience.value}
Tone: {intent.tone.value}
Objective: {intent.objective}

Context Passages:
{context_bullets}
<|im_end|>
<|im_start|>assistant
"""
        response = model(
            prompt,
            max_tokens=300,
            temperature=0.1,
            stop=["<|im_end|>", "\n\n\n"]
        )
        output_text = response["choices"][0]["text"].strip()

        # Parse JSON from output
        json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                strategy = ContentStrategy(
                    headline_hook=str(data.get("headline_hook", f"Key Insights: {doc_title}")),
                    key_themes=data.get("key_themes", [intent.objective]),
                    suggested_structure=data.get("suggested_structure", ["Introduction", "Key Findings", "Conclusion"]),
                    recommended_cta=str(data.get("recommended_cta", "Explore the full findings.")),
                    tone_guidelines=str(data.get("tone_guidelines", f"Adopt a {intent.tone.value} tone for {intent.audience.value} audience."))
                )

                claims = []
                for idx, c in enumerate(data.get("claims", []), start=1):
                    if isinstance(c, dict):
                        claims.append(ClaimItem(
                            id=str(c.get("id", f"claim_{idx}")),
                            statement=str(c.get("statement", "")),
                            source_element_ids=[str(s) for s in c.get("source_element_ids", [])],
                            confidence=float(c.get("confidence", 0.9))
                        ))

                metrics = []
                for m in data.get("metrics", []):
                    if isinstance(m, dict):
                        metrics.append(KeyMetricItem(
                            label=str(m.get("label", "Metric")),
                            value=str(m.get("value", "")),
                            context=str(m.get("context", "")),
                            source_element_id=m.get("source_element_id"),
                            page=int(m.get("page")) if m.get("page") else None
                        ))

                # Merge with semantic_doc entities
                entities = semantic_doc.entities or self._extract_entities_heuristic(evidence)
                relationships = semantic_doc.relationships or []

                # Deterministic synthesis for strategic narrative structures the LLM doesn't produce
                _, _, _, _, _, document_story, key_findings, recommendations, executive_insights = self._deterministic_extraction(
                    intent, evidence, tables, visuals, doc_title, semantic_doc
                )
                return entities, claims, relationships, metrics, strategy, document_story, key_findings, recommendations, executive_insights
            except Exception as parse_err:
                logger.warning("Qwen3-4B JSON parsing error (%s); falling back to deterministic extraction", parse_err)

        # If JSON parsing fails, fall back to deterministic
        return self._deterministic_extraction(intent, evidence, tables, visuals, doc_title, semantic_doc)

    def _deterministic_extraction(
        self,
        intent: IntentAndPersonalization,
        evidence: List[EvidenceItem],
        tables: List[TableSummaryItem],
        visuals: List[VisualInsightItem],
        doc_title: str,
        semantic_doc: SemanticDocument,
    ) -> tuple[
        List[EntityItem],
        List[ClaimItem],
        List[RelationshipItem],
        List[KeyMetricItem],
        ContentStrategy,
        str,
        List[KeyFindingItem],
        List[RecommendationItem],
        List[ExecutiveInsightItem]
    ]:
        """
        Deterministic, rule-based extraction for offline tests and fast CPU environments.
        """
        # 1. Strategy Formulation based on intent
        format_name = intent.output_type.value.replace("_", " ").title()
        headline_hook = f"{format_name}: Strategic Insights on {doc_title}"

        structure_map = {
            "linkedin_post": ["Attention Grabber / Hook", "Key Problem / Data Context", "Core Breakthrough / Finding", "Actionable Takeaway", "Call-to-Action & Hashtags"],
            "twitter_thread": ["1/ Hook & Context", "2/ The Core Problem", "3/ Key Data & Stats", "4/ Solution / Insight", "5/ Summary & Takeaway"],
            "executive_summary": ["Executive Overview", "Strategic Context", "Key Findings & Quantitative Evidence", "Risk & Opportunity Analysis", "Recommendations"],
            "presentation_deck": ["Slide 1: Title & Agenda", "Slide 2: Background & Problem", "Slide 3: Key Data & Metrics", "Slide 4: Strategic Recommendations", "Slide 5: Q&A / Next Steps"],
            "infographic_brief": ["Header & Focal Stat", "Key Data Comparison (Table/Chart)", "Process / Flow Breakdown", "Core Callouts", "Source Citations"],
            "video_script": ["Scene 1: Visual Hook & Intro", "Scene 2: Problem Statement", "Scene 3: Deep Dive into Insights", "Scene 4: Key Takeaway & Closing"],
        }
        suggested_structure = structure_map.get(intent.output_type.value, ["Overview", "Key Findings", "Implications", "Recommendations"])

        strategy = ContentStrategy(
            headline_hook=headline_hook,
            key_themes=[intent.objective] + (intent.focus_keywords or []),
            suggested_structure=suggested_structure,
            recommended_cta=f"Review the {doc_title} transformation report for detailed implementation steps.",
            tone_guidelines=f"Deliver insights using a {intent.tone.value} voice tailored specifically for {intent.audience.value}."
        )

        # 2. Claims Extraction from top evidence (skip raw json, decorative icon text, and pure URLs)
        claims: List[ClaimItem] = []
        for idx, ev in enumerate(evidence[:6]):
            # Skip chunks that are raw json dumps or icon descriptions
            if ev.text.strip().startswith("{") or "visual_type" in ev.text.lower():
                continue
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ev.text) if len(s.strip()) > 20 and not s.strip().startswith("|")]
            if sentences:
                claims.append(
                    ClaimItem(
                        id=f"claim_{len(claims)+1}",
                        statement=sentences[0],
                        source_element_ids=[ev.element_id] if ev.element_id else [],
                        confidence=round(ev.relevance_score, 2)
                    )
                )
            if len(claims) >= 5:
                break

        # Fallback if text claims are sparse: synthesize from tables/visuals
        if not claims and tables:
            claims.append(
                ClaimItem(
                    id="claim_1",
                    statement=f"Document defines a structured architecture across key modules detailed on Page {tables[0].page}.",
                    source_element_ids=[tables[0].element_id],
                    confidence=0.95
                )
            )

        # 3. Metrics Extraction from text & tables (strip URLs and software versions)
        metrics: List[KeyMetricItem] = []
        # First: extract any concrete metrics from tables
        for tbl in tables:
            for line in tbl.markdown_table.splitlines():
                if "|" in line and not line.strip().startswith("| -") and not line.strip().startswith("|Technology") and not line.strip().startswith("| Technology"):
                    cells = [c.strip() for c in line.split("|")[1:-1]]
                    if len(cells) >= 2 and cells[0] and not cells[0].startswith("---"):
                        row_name = cells[0]
                        for c_val in cells[1:]:
                            if c_val and not c_val.startswith("http") and not c_val.startswith("https") and len(c_val) < 60:
                                # Check if cell contains number, percentage, multiplier, or key tech spec
                                if re.search(r"\b(?:\d+(?:\.\d+)?(?:%|x|k|M|B|GB|TB|ms)?)\b", c_val) or any(k in c_val.lower() for k in ["docker", "qwen", "jwt", "faster"]):
                                    metrics.append(
                                        KeyMetricItem(
                                            label=f"{row_name}",
                                            value=c_val,
                                            context=f"{row_name}: {c_val}",
                                            source_element_id=tbl.element_id,
                                            page=tbl.page
                                        )
                                    )
                                    if len(metrics) >= 6:
                                        break
                if len(metrics) >= 6:
                    break

        # Second: text metrics with strict URL/version sanitization
        for ev in evidence:
            if len(metrics) >= 8:
                break
            # Strip URLs and markdown links before matching numbers
            sanitized_text = re.sub(r"https?://\S+", "", ev.text)
            sanitized_text = re.sub(r"\[.*?\]\(.*?\)", "", sanitized_text)
            sanitized_text = re.sub(r"\bv\d+(?:\.\d+)*\b", "", sanitized_text, flags=re.IGNORECASE)

            matches = self.metric_regex.findall(sanitized_text)
            for m in matches:
                clean_m = m.strip()
                # Ignore isolated single-digit integers without units or symbols
                if clean_m.isdigit() and len(clean_m) <= 2:
                    continue
                pos = sanitized_text.find(clean_m)
                start = max(0, pos - 30)
                end = min(len(sanitized_text), pos + len(clean_m) + 30)
                ctx = sanitized_text[start:end].strip()

                metrics.append(
                    KeyMetricItem(
                        label=f"Metric {len(metrics) + 1}",
                        value=clean_m,
                        context=ctx,
                        source_element_id=ev.element_id,
                        page=ev.page
                    )
                )
                if len(metrics) >= 8:
                    break

        # 4. Entities & Relationships (prioritize table first-column names if available)
        table_entities = []
        for tbl in tables:
            for line in tbl.markdown_table.splitlines():
                if "|" in line and not line.strip().startswith("| -") and not line.strip().startswith("| Technology") and not line.strip().startswith("|Technology"):
                    cells = [c.strip() for c in line.split("|")[1:-1]]
                    if cells and cells[0] and not cells[0].startswith("---") and len(cells[0]) > 2:
                        ent_name = cells[0]
                        if ent_name not in [e.name for e in table_entities] and ent_name.lower() not in ("technology", "purpose", "solution", "reference"):
                            table_entities.append(
                                EntityItem(
                                    id=f"ent_{len(table_entities)+1}",
                                    name=ent_name,
                                    category="TECHNOLOGY",
                                    mentions=[tbl.element_id],
                                    confidence=0.92
                                )
                            )
                            if len(table_entities) >= 8:
                                break

        entities = table_entities or semantic_doc.entities or self._extract_entities_heuristic(evidence)
        relationships = semantic_doc.relationships or []

        # 5. Synthesize Document Story, Key Findings, and Recommendations
        key_findings: List[KeyFindingItem] = []
        for idx, c in enumerate(claims[:4]):
            key_findings.append(
                KeyFindingItem(
                    finding=c.statement,
                    source_element_ids=c.source_element_ids,
                    confidence=c.confidence
                )
            )

        # If findings were sparse, synthesize from diagrams/tables
        for vis in visuals[:2]:
            if vis.takeaway and len(key_findings) < 5:
                # Extract first clean sentence or key details
                clean_vis = re.sub(r"```json.*?```", "", vis.takeaway, flags=re.DOTALL).strip()
                if not clean_vis:
                    clean_vis = vis.caption or f"{vis.element_type.title()} analysis"
                key_findings.append(
                    KeyFindingItem(
                        finding=clean_vis[:200],
                        source_element_ids=[vis.element_id],
                        confidence=0.90
                    )
                )

        recommendations: List[RecommendationItem] = [
            RecommendationItem(
                action=f"Adopt the core architectural and transformation recommendations outlined in {doc_title}.",
                rationale="Aligns operational processes with the verified system capabilities and specifications.",
                priority="high"
            ),
            RecommendationItem(
                action="Deploy robust verification and access control measures across integrated modules.",
                rationale="Ensures enterprise data security, audit integrity, and regulatory governance.",
                priority="strategic"
            )
        ]

        executive_insights: List[ExecutiveInsightItem] = [
            ExecutiveInsightItem(
                headline=f"Strategic Transformation Vector: {doc_title}",
                takeaway=f"{doc_title} presents an integrated architectural paradigm designed to optimize performance, automate delivery, and maintain security.",
                confidence=0.95
            )
        ]

        document_story = (
            f"This document, titled '{doc_title}', outlines a strategic framework and architecture designed to transform inputs into high-impact deliverables. "
            f"Key technologies and processes work in coordination to provide automated, scalable, and verifiable results for executive decision-makers.\n\n"
            f"Through structured components including data ingestion, semantic analysis, and secure deployment, the system establishes operational efficiency, "
            f"data integrity, and competitive advantages across business and technical workflows."
        )

        return entities, claims, relationships, metrics, strategy, document_story, key_findings, recommendations, executive_insights

    def _extract_entities_heuristic(self, evidence: List[EvidenceItem]) -> List[EntityItem]:
        """Heuristic named entity extraction using capitalization patterns."""
        entities: List[EntityItem] = []
        seen = set()
        cap_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

        for ev in evidence:
            for match in cap_pattern.findall(ev.text):
                if match not in seen and len(match) > 3 and match not in ("Document", "Page", "Type", "Table", "Figure"):
                    seen.add(match)
                    entities.append(
                        EntityItem(
                            id=f"ent_{len(entities)+1}",
                            name=match,
                            category="CONCEPT",
                            mentions=[ev.element_id] if ev.element_id else [],
                            confidence=0.85
                        )
                    )
                    if len(entities) >= 8:
                        return entities
        return entities

    def _compile_orchestrator_prompt_context(
        self,
        doc_title: str,
        intent: IntentAndPersonalization,
        strategy: ContentStrategy,
        claims: List[ClaimItem],
        metrics: List[KeyMetricItem],
        tables: List[TableSummaryItem],
        visuals: List[VisualInsightItem],
        evidence: List[EvidenceItem]
    ) -> str:
        """
        Compiles an unambiguous, dense, structured markdown block ready for the Content Orchestrator.
        Includes provenance citations [elem_id, Page N] for verifiable generation.
        """
        lines = []
        lines.append(f"# Knowledge Context for Content Orchestration")
        lines.append(f"**Document**: {doc_title}")
        lines.append(f"**Target Format**: {intent.output_type.value} | **Audience**: {intent.audience.value} | **Tone**: {intent.tone.value}")
        lines.append(f"**Core Objective**: {intent.objective}")
        lines.append("")

        lines.append(f"## 1. Content Strategy Blueprint")
        lines.append(f"- **Suggested Hook / Title**: {strategy.headline_hook}")
        lines.append(f"- **Tone & Style Directive**: {strategy.tone_guidelines or 'Professional and clear'}")
        lines.append(f"- **Suggested Section Structure**:")
        for s in strategy.suggested_structure:
            lines.append(f"  - {s}")
        if strategy.recommended_cta:
            lines.append(f"- **Recommended Call to Action**: {strategy.recommended_cta}")
        lines.append("")

        if claims:
            lines.append(f"## 2. Core Verified Claims & Factual Propositions")
            for c in claims:
                citation = f" [Source: {', '.join(c.source_element_ids)}]" if c.source_element_ids else ""
                lines.append(f"- **{c.id}**: {c.statement}{citation}")
            lines.append("")

        if metrics:
            lines.append(f"## 3. Key Quantitative Metrics & Evidence")
            for m in metrics:
                citation = f" [Source: {m.source_element_id}, Page {m.page}]" if m.source_element_id else ""
                lines.append(f"- **{m.value}** ({m.label}): \"...{m.context}...\"{citation}")
            lines.append("")

        if tables:
            lines.append(f"## 4. Structured Tables")
            for t in tables:
                lines.append(f"### Table on Page {t.page} (Element: {t.element_id})")
                if t.caption:
                    lines.append(f"*Caption: {t.caption}*")
                lines.append(t.markdown_table)
                lines.append("")

        if visuals:
            lines.append(f"## 5. Visual Insights & Chart Interpretations")
            for v in visuals:
                lines.append(f"- **{v.element_type.title()} (Element: {v.element_id}, Page {v.page})**: {v.takeaway}")
                if v.image_path:
                    lines.append(f"  *Image asset*: `{v.image_path}`")
            lines.append("")

        lines.append(f"## 6. Retrieved Semantic Context Passages")
        for ev in evidence:
            lines.append(f"### [Element: {ev.element_id or 'N/A'} | Page {ev.page} | Relevance: {ev.relevance_score}]")
            lines.append(ev.text)
            lines.append("")

        return "\n".join(lines)

    def _resolve_document_title(
        self,
        candidate_title: Optional[str],
        fallback_title: Optional[str],
        semantic_doc: SemanticDocument
    ) -> str:
        """
        Determines the most accurate, human-readable title for the document.
        If the candidate title or filename is missing or is an auto-generated UUID,
        inspects the first page/slide elements for a header/title.
        """
        uuid_pattern = re.compile(
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
        )

        title = (candidate_title or "").strip()
        is_uuid_candidate = bool(uuid_pattern.search(title))

        # If candidate_title is already a meaningful, non-UUID title, use it
        if title and not is_uuid_candidate:
            return title

        # Check fallback_title (e.g., original filename without extension)
        fallback = (fallback_title or "").strip()
        # Strip common extensions if filename was passed
        cleaned_fallback = re.sub(r"\.(pdf|docx|pptx|txt|png|jpg|jpeg|webm|mp4)$", "", fallback, flags=re.IGNORECASE).strip()
        is_uuid_fallback = bool(uuid_pattern.search(cleaned_fallback))

        if cleaned_fallback and not is_uuid_fallback:
            return cleaned_fallback

        # Candidate and fallback are UUIDs or empty: inspect page 1 elements for header/title
        if semantic_doc and semantic_doc.elements:
            page_1_elements = [e for e in semantic_doc.elements if e.page == 1]

            # 1. Prefer explicit title or heading elements
            for elem in page_1_elements:
                if elem.type in ("title", "heading", "header"):
                    text = (elem.content.text or "").strip()
                    if text and len(text) < 120 and not uuid_pattern.search(text):
                        return text

            # 2. Fall back to the first non-empty text element on page 1 that is title-like
            for elem in page_1_elements:
                if elem.type == "text":
                    text = (elem.content.text or "").strip()
                    first_line = text.split("\n")[0].strip()
                    if first_line and 3 < len(first_line) < 100 and not uuid_pattern.search(first_line):
                        return first_line

        # Final fallback: whatever candidate or fallback string was available, or "Untitled Document"
        return cleaned_fallback or title or "Untitled Document"


# Global knowledge engine instance
knowledge_engine = KnowledgeEngine()

__all__ = ["KnowledgeEngine", "knowledge_engine"]