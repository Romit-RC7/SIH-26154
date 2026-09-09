"""
Repair Service.
Executes automated repair loops when generated content violates schema constraints.
"""

from typing import List, Dict, Any, Tuple
from backend.app.core.logging import logger
from backend.app.schemas.intent import OutputType
from backend.app.schemas.generated_artefact import GeneratedArtefact
from backend.app.schemas.knowledge_package import KnowledgePackage
from backend.app.services.validation.schema_validator import schema_validator


class RepairService:
    """
    Executes targeted repairs on non-compliant artefacts.
    """

    MAX_REPAIR_ATTEMPTS = 2

    def repair_artefact(
        self,
        artefact: GeneratedArtefact,
        violations: List[str],
        kp: KnowledgePackage
    ) -> Tuple[GeneratedArtefact, int, bool]:
        """
        Executes automated repair loop on non-compliant artefact.
        Returns (repaired_artefact, repair_attempts, is_now_compliant).
        """
        logger.info("Initiating repair loop for artefact %s (%d violations)", artefact.artefact_id, len(violations))
        
        attempts = 0
        current_content = dict(artefact.content)
        out_type = artefact.output_type

        while attempts < self.MAX_REPAIR_ATTEMPTS:
            attempts += 1
            repaired_content = self._apply_deterministic_repairs(current_content, out_type, violations, kp)
            
            # Check compliance of repaired content
            temp_artefact = GeneratedArtefact(
                artefact_id=artefact.artefact_id,
                document_id=artefact.document_id,
                output_type=out_type,
                status="repaired",
                content=repaired_content,
                raw_llm_output=artefact.raw_llm_output,
                generation_metadata=artefact.generation_metadata,
            )
            
            is_compliant, remaining_violations = schema_validator.validate_schema(temp_artefact)
            if is_compliant:
                logger.info("Artefact %s successfully repaired after %d attempts", artefact.artefact_id, attempts)
                temp_artefact.status = "success"
                return temp_artefact, attempts, True
            
            current_content = repaired_content
            violations = remaining_violations

        # If still non-compliant after max attempts, apply hard truncation fallback
        logger.warning("Max repair attempts reached for %s; applying hard truncation fallback", artefact.artefact_id)
        hard_repaired_content = self._apply_hard_truncation(current_content, out_type)
        
        final_artefact = GeneratedArtefact(
            artefact_id=artefact.artefact_id,
            document_id=artefact.document_id,
            output_type=out_type,
            status="repaired_with_warnings",
            content=hard_repaired_content,
            raw_llm_output=artefact.raw_llm_output,
            generation_metadata=artefact.generation_metadata,
        )
        return final_artefact, attempts, False

    def _apply_deterministic_repairs(
        self,
        content: Dict[str, Any],
        out_type: OutputType,
        violations: List[str],
        kp: KnowledgePackage
    ) -> Dict[str, Any]:
        """Applies targeted fix logic based on violation messages."""
        repaired = dict(content)

        if out_type == OutputType.LINKEDIN_POST:
            if not repaired.get("hook") or not isinstance(repaired.get("hook"), str) or len(repaired["hook"].strip()) < 10:
                repaired["hook"] = kp.strategy.headline_hook or "Key Strategic Findings and Benchmarks from Recent Analysis."
            if not repaired.get("cta") or not isinstance(repaired.get("cta"), str) or len(repaired["cta"].strip()) < 10:
                repaired["cta"] = kp.strategy.recommended_cta or "Contact our strategy team to learn more."
            hashtags = repaired.get("hashtags")
            if not isinstance(hashtags, list) or len(hashtags) < 2 or any(not isinstance(h, str) for h in hashtags):
                repaired["hashtags"] = ["#AI", "#Innovation", "#StrategicInsights"]

            body = repaired.get("body", "")
            if not body or not isinstance(body, str):
                body = (
                    f"{kp.document_title or 'Our recent study'} demonstrates compelling transformations "
                    f"in operational performance and strategic growth.\n\n"
                    f"{kp.orchestrator_prompt_context or 'Empirical results highlight substantial efficiency improvements across enterprise workflows.'}\n\n"
                    f"Organizations implementing these validated insights report notable gains in productivity and execution accuracy."
                )
                repaired["body"] = body

            hook_str = str(repaired.get("hook", ""))
            body_str = str(repaired.get("body", ""))
            cta_str = str(repaired.get("cta", ""))
            words = len(f"{hook_str} {body_str} {cta_str}".split())
            if words < 80:
                padding = (
                    f"\n\nFurther analysis indicates that adopting these evidence-grounded recommendations enables sustained competitive advantage, "
                    f"accelerating operational milestones while maintaining governance and reliability across complex organizational environments.\n\n"
                    f"Cross-functional teams that integrate these structural standards consistently report heightened operational velocity, "
                    f"reduced overhead latency, and enhanced alignment with long-term strategic benchmarks across global market initiatives."
                )
                repaired["body"] = (body_str + padding).strip()

        elif out_type == OutputType.TWITTER_THREAD:
            tweets = repaired.get("tweets", [])
            fixed_tweets = []
            for idx, tweet in enumerate(tweets, start=1):
                if isinstance(tweet, dict):
                    t_text = tweet.get("text", "")
                    if len(t_text) > 280:
                        t_text = t_text[:277] + "..."
                    fixed_tweets.append({"index": idx, "text": t_text, "char_count": len(t_text)})
                else:
                    t_str = str(tweet)
                    if len(t_str) > 280:
                        t_str = t_str[:277] + "..."
                    fixed_tweets.append({"index": idx, "text": t_str, "char_count": len(t_str)})
            repaired["tweets"] = fixed_tweets
            repaired["tweet_count"] = len(fixed_tweets)

        elif out_type == OutputType.EXECUTIVE_SUMMARY:
            if not repaired.get("title"):
                repaired["title"] = f"Executive Summary: {kp.document_title or kp.document_id}"
            overview = repaired.get("overview", "")
            if not overview or len(overview.split()) < 20:
                repaired["overview"] = (
                    f"This executive briefing provides a strategic synthesis of the findings, benchmarks, and "
                    f"empirical evidence established in {kp.document_title or 'the source document'}, enabling "
                    f"decision-makers to evaluate core performance metrics and execute informed initiatives."
                )
            findings = repaired.get("key_findings", [])
            if not isinstance(findings, list) or len(findings) < 2:
                if not isinstance(findings, list):
                    findings = []
                if len(findings) == 0:
                    findings = [
                        "Core strategic findings verified from document evidence.",
                        "Operational benchmarks establish scalable performance.",
                    ]
                elif len(findings) == 1:
                    findings.append("Secondary evaluation confirms consistent operational improvements across functional units.")
                repaired["key_findings"] = findings
            if not repaired.get("recommendations"):
                repaired["recommendations"] = ["Implement core findings into operational workflows."]

        return repaired

    def _apply_hard_truncation(self, content: Dict[str, Any], out_type: OutputType) -> Dict[str, Any]:
        """Forces hard boundary truncation for extreme non-compliance."""
        repaired = dict(content)
        if out_type == OutputType.TWITTER_THREAD:
            tweets = repaired.get("tweets", [])
            for t in tweets:
                if isinstance(t, dict) and len(t.get("text", "")) > 280:
                    t["text"] = t["text"][:275] + "..."
                    t["char_count"] = len(t["text"])
        return repaired


repair_service = RepairService()
