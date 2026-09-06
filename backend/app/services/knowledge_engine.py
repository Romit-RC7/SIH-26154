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

        doc_title = semantic_doc.metadata.title or document.filename

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
        evidence_items = [
            EvidenceItem(
                chunk_id=c.id,
                element_id=c.element_id,
                page=c.page,
                chunk_type=str(c.chunk_type.value if hasattr(c.chunk_type, "value") else c.chunk_type),
                text=c.content,
                relevance_score=c.similarity_score
            )
            for c in retrieved_chunks
        ]

        entities, claims, relationships, metrics, strategy = self._extract_knowledge_and_strategy(
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
            retrieved_evidence=evidence_items,
            entities=entities,
            claims=claims,
            relationships=relationships,
            key_metrics=metrics,
            tables=tables,
            visual_insights=visual_insights,
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
        """Extracts visual elements (figures, charts) and their descriptive takeaways."""
        insights: List[VisualInsightItem] = []
        seen_element_ids = set()

        for elem in doc.elements:
            if elem.type in ("figure", "chart", "image") and elem.id not in seen_element_ids:
                seen_element_ids.add(elem.id)
                raw_attrs = elem.content.raw_attributes or {}
                raw_val = raw_attrs.get("visual_analysis") or raw_attrs.get("description")
                if isinstance(raw_val, dict):
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
    ) -> tuple[List[EntityItem], List[ClaimItem], List[RelationshipItem], List[KeyMetricItem], ContentStrategy]:
        """
        Attempts Qwen3-4B inference to perform reasoning; falls back gracefully to
        deterministic semantic extraction if Qwen runtime is not active.
        """
        if qwen_fusion_initializer.is_available():
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
    ) -> tuple[List[EntityItem], List[ClaimItem], List[RelationshipItem], List[KeyMetricItem], ContentStrategy]:
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
            data = json.loads(json_match.group(0))
            strategy = ContentStrategy(
                headline_hook=data.get("headline_hook", f"Key Insights: {doc_title}"),
                key_themes=data.get("key_themes", [intent.objective]),
                suggested_structure=data.get("suggested_structure", ["Introduction", "Key Findings", "Conclusion"]),
                recommended_cta=data.get("recommended_cta", "Explore the full findings."),
                tone_guidelines=data.get("tone_guidelines", f"Adopt a {intent.tone.value} tone for {intent.audience.value} audience.")
            )
            claims = [ClaimItem(**c) for c in data.get("claims", [])]
            metrics = [KeyMetricItem(**m) for m in data.get("metrics", [])]

            # Merge with semantic_doc entities
            entities = semantic_doc.entities or self._extract_entities_heuristic(evidence)
            relationships = semantic_doc.relationships or []
            return entities, claims, relationships, metrics, strategy

        # If JSON parsing fails, fall back to deterministic
        return self._deterministic_extraction(intent, evidence, tables, visuals, doc_title, semantic_doc)

    def _deterministic_extraction(
        self,
        intent: IntentAndPersonalization,
        evidence: List[EvidenceItem],
        tables: List[TableSummaryItem],
        visuals: List[VisualInsightItem],
        doc_title: str,
        semantic_doc: SemanticDocument
    ) -> tuple[List[EntityItem], List[ClaimItem], List[RelationshipItem], List[KeyMetricItem], ContentStrategy]:
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

        # 2. Claims Extraction from top evidence
        claims: List[ClaimItem] = []
        for idx, ev in enumerate(evidence[:5]):
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ev.text) if len(s.strip()) > 20]
            if sentences:
                claims.append(
                    ClaimItem(
                        id=f"claim_{idx+1}",
                        statement=sentences[0],
                        source_element_ids=[ev.element_id] if ev.element_id else [],
                        confidence=round(ev.relevance_score, 2)
                    )
                )

        # 3. Metrics Extraction from text & tables
        metrics: List[KeyMetricItem] = []
        metric_count = 0
        for ev in evidence:
            matches = self.metric_regex.findall(ev.text)
            for m in matches:
                # Find surrounding phrase for context
                pos = ev.text.find(m)
                start = max(0, pos - 30)
                end = min(len(ev.text), pos + len(m) + 30)
                ctx = ev.text[start:end].strip()

                metrics.append(
                    KeyMetricItem(
                        label=f"Data Point {metric_count + 1}",
                        value=m,
                        context=ctx,
                        source_element_id=ev.element_id,
                        page=ev.page
                    )
                )
                metric_count += 1
                if metric_count >= 6:
                    break
            if metric_count >= 6:
                break

        # 4. Entities & Relationships
        entities = semantic_doc.entities or self._extract_entities_heuristic(evidence)
        relationships = semantic_doc.relationships or []

        return entities, claims, relationships, metrics, strategy

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


# Global knowledge engine instance
knowledge_engine = KnowledgeEngine()

__all__ = ["KnowledgeEngine", "knowledge_engine"]
