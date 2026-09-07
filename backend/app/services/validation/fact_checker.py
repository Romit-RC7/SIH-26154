"""
Fact Checker Service.
Evaluates factual grounding of generated text against source document evidence
using sentence-level vector similarity (BGE-small) or semantic keyword overlap.
"""

import re
from typing import List, Dict, Any, Tuple
from backend.app.core.logging import logger
from backend.app.schemas.generated_artefact import GeneratedArtefact
from backend.app.schemas.knowledge_package import KnowledgePackage
from backend.app.schemas.validation import ClaimVerification
from backend.app.services.model_initializer.bge_initializer import bge_initializer


class FactChecker:
    """
    Sentence-level fact verification and hallucination detection.
    """

    GROUNDING_THRESHOLD = 0.45  # Cosine similarity threshold for grounded claims

    def verify_factuality(
        self,
        artefact: GeneratedArtefact,
        kp: KnowledgePackage
    ) -> Tuple[float, bool, List[ClaimVerification]]:
        """
        Extracts claim statements from artefact content, matches against KnowledgePackage evidence,
        and computes overall trust score (ratio of grounded claims).
        """
        statements = self._extract_statements(artefact.content)
        if not statements:
            logger.info("No text statements found for factuality verification in artefact %s", artefact.artefact_id)
            return 1.0, True, []

        evidence_texts = [item.text for item in kp.retrieved_evidence]
        if not evidence_texts and kp.orchestrator_prompt_context:
            evidence_texts = [kp.orchestrator_prompt_context]

        verifications: List[ClaimVerification] = []
        grounded_count = 0

        for stmt in statements:
            best_score, matched_chunk, matched_text = self._match_statement(stmt, evidence_texts, kp)
            is_grounded = best_score >= self.GROUNDING_THRESHOLD
            if is_grounded:
                grounded_count += 1

            verifications.append(
                ClaimVerification(
                    statement=stmt,
                    is_grounded=is_grounded,
                    confidence_score=round(best_score, 3),
                    matched_chunk_id=matched_chunk,
                    matched_source_text=matched_text[:150] if matched_text else None,
                )
            )

        trust_score = round(grounded_count / len(statements), 3) if statements else 1.0
        is_fully_grounded = trust_score >= 0.85

        logger.info(
            "Fact verification complete for artefact %s: trust_score=%.2f, grounded=%d/%d",
            artefact.artefact_id, trust_score, grounded_count, len(statements)
        )
        return trust_score, is_fully_grounded, verifications

    def _extract_statements(self, content: Dict[str, Any]) -> List[str]:
        """Extracts individual sentences or claims from structured content dictionary."""
        text_blobs: List[str] = []

        def recurse_extract(val: Any):
            if isinstance(val, str):
                if len(val.strip()) > 15:
                    text_blobs.append(val.strip())
            elif isinstance(val, list):
                for item in val:
                    recurse_extract(item)
            elif isinstance(val, dict):
                for k, v in val.items():
                    if k not in ("hashtags", "visual_type", "index", "slide_number", "scene_number", "word_count", "char_count", "duration_sec"):
                        recurse_extract(v)

        recurse_extract(content)

        statements: List[str] = []
        for blob in text_blobs:
            # Split sentences by punctuation
            sentences = re.split(r"(?<=[.!?])\s+", blob)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) > 20 and not s_clean.startswith("#"):
                    statements.append(s_clean)

        return statements[:15]  # Limit to top 15 main statements for verification performance

    def _match_statement(
        self,
        statement: str,
        evidence_texts: List[str],
        kp: KnowledgePackage
    ) -> Tuple[float, str, str]:
        """Matches a single statement against evidence pool."""
        if not evidence_texts:
            return 0.5, "fallback", "Context block fallback"

        # 1. Try BGE vector similarity if initialized
        if bge_initializer.is_available():
            try:
                model = bge_initializer.load()
                stmt_emb = model.encode(statement, normalize_embeddings=True)
                evidence_embs = model.encode(evidence_texts, normalize_embeddings=True)
                
                # Dot product cosine similarity
                sims = (evidence_embs @ stmt_emb).tolist()
                best_idx = max(range(len(sims)), key=lambda i: sims[i])
                best_score = float(sims[best_idx])
                
                chunk_id = kp.retrieved_evidence[best_idx].chunk_id if best_idx < len(kp.retrieved_evidence) else "ctx_block"
                return best_score, chunk_id, evidence_texts[best_idx]
            except Exception as exc:
                logger.warning("BGE encoding failed during fact check (%s); using token overlap fallback", exc)

        # 2. Token overlap fallback
        stmt_tokens = set(re.findall(r"\w+", statement.lower()))
        if not stmt_tokens:
            return 0.5, "fallback", ""

        best_score = 0.0
        best_text = ""
        best_chunk = "fallback"

        for idx, ev_text in enumerate(evidence_texts):
            ev_tokens = set(re.findall(r"\w+", ev_text.lower()))
            if not ev_tokens:
                continue
            intersection = stmt_tokens.intersection(ev_tokens)
            score = len(intersection) / len(stmt_tokens)
            if score > best_score:
                best_score = score
                best_text = ev_text
                best_chunk = kp.retrieved_evidence[idx].chunk_id if idx < len(kp.retrieved_evidence) else "ctx_block"

        # Scale token score to cosine range
        scaled_score = min(1.0, best_score * 1.2)
        return scaled_score, best_chunk, best_text


fact_checker = FactChecker()
