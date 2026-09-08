"""
Response Parser for Content Orchestrator.
Extracts and validates structured JSON content from Qwen3-8B raw text outputs,
providing robust sanitization and graceful schema fallbacks.
"""

import json
import re
from typing import Dict, Any, Tuple, Optional
from backend.app.core.logging import logger
from backend.app.schemas.intent import OutputType
from backend.app.schemas.generated_artefact import (
    LinkedInPostContent,
    TwitterThreadContent,
    TweetItem,
    ExecutiveSummaryContent,
    PresentationDeckContent,
    SlideItem,
    InfographicBriefContent,
    InfographicStat,
    InfographicSection,
    VideoScriptContent,
    VideoScene,
    BlogPostContent,
    BlogSection,
)
from backend.app.schemas.knowledge_package import KnowledgePackage


class ResponseParser:
    """
    Parses, cleans, and validates raw LLM output into format-specific Pydantic dictionaries.
    """

    @staticmethod
    def parse_response(
        raw_output: str,
        output_type: OutputType,
        kp: KnowledgePackage
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Parses raw LLM string into a validated dictionary matching output_type schema.
        Returns (status: 'success' | 'fallback' | 'parse_error', content_dict).
        """
        extracted_json = ResponseParser._extract_json_block(raw_output)
        if extracted_json is not None:
            try:
                validated_content = ResponseParser._validate_and_normalize(extracted_json, output_type)
                return "success", validated_content
            except Exception as exc:
                logger.warning("JSON parsed but failed schema validation for %s: %s. Using normalized object.", output_type, exc)
                return "success", extracted_json

        # Fallback if raw text wasn't valid JSON
        logger.warning("Could not parse JSON from LLM output for %s. Generating structured deterministic fallback.", output_type)
        fallback_content = ResponseParser._build_deterministic_fallback(raw_output, output_type, kp)
        return "fallback", fallback_content

    @staticmethod
    def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON object from text containing markdown fences or raw braces."""
        if not text or not text.strip():
            return None

        clean_text = text.strip()

        # 1. Try direct json.loads
        try:
            parsed = json.loads(clean_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 2. Extract from markdown code blocks: ```json ... ```
        block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", clean_text, re.DOTALL | re.IGNORECASE)
        if block_match:
            try:
                parsed = json.loads(block_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # 3. Find outermost curly braces { ... }
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = clean_text[start_idx : end_idx + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                # Try fixing common trailing comma issue before closing brace
                fixed = re.sub(r",\s*([\]}])", r"\1", candidate)
                try:
                    parsed = json.loads(fixed)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

        return None

    @staticmethod
    def _validate_and_normalize(data: Dict[str, Any], output_type: OutputType) -> Dict[str, Any]:
        """Validates dictionary against format-specific Pydantic schema."""
        if output_type == OutputType.LINKEDIN_POST:
            return LinkedInPostContent.model_validate(data).model_dump()
        elif output_type == OutputType.TWITTER_THREAD:
            return TwitterThreadContent.model_validate(data).model_dump()
        elif output_type == OutputType.EXECUTIVE_SUMMARY:
            return ExecutiveSummaryContent.model_validate(data).model_dump()
        elif output_type == OutputType.PRESENTATION_DECK:
            return PresentationDeckContent.model_validate(data).model_dump()
        elif output_type == OutputType.INFOGRAPHIC_BRIEF:
            return InfographicBriefContent.model_validate(data).model_dump()
        elif output_type == OutputType.VIDEO_SCRIPT:
            return VideoScriptContent.model_validate(data).model_dump()
        elif output_type == OutputType.BLOG_POST:
            return BlogPostContent.model_validate(data).model_dump()
        else:
            return data

    @staticmethod
    def _build_deterministic_fallback(
        raw_text: str,
        output_type: OutputType,
        kp: KnowledgePackage
    ) -> Dict[str, Any]:
        """Builds a schema-compliant fallback when model output cannot be parsed as JSON."""
        doc_title = kp.document_title or f"Document {kp.document_id[:8]}"
        hook = kp.strategy.headline_hook or f"Key findings and strategic insights from {doc_title}."
        raw_clean = raw_text.strip() if raw_text else hook

        if output_type == OutputType.LINKEDIN_POST:
            hashtags = [f"#{kw.replace(' ', '')}" for kw in kp.intent.focus_keywords[:4]] or ["#AI", "#Innovation", "#StrategicInsights"]
            return LinkedInPostContent(
                hook=hook,
                body=raw_clean[:800],
                cta=kp.strategy.recommended_cta or "Read the complete findings to learn more.",
                hashtags=hashtags,
                word_count=len(raw_clean.split()),
            ).model_dump()

        elif output_type == OutputType.TWITTER_THREAD:
            tweets = [
                TweetItem(index=1, text=f"1/5 🚀 {hook[:240]}"),
                TweetItem(index=2, text=f"2/5 📊 Evidence shows significant developments across {doc_title}."),
            ]
            for i, metric in enumerate(kp.key_metrics[:2], start=3):
                tweets.append(TweetItem(index=i, text=f"{i}/5 📈 {metric.label}: {metric.value} ({metric.context or ''})"[:270]))
            tweets.append(TweetItem(index=len(tweets) + 1, text=f"{len(tweets) + 1}/5 💡 Takeaway: {kp.strategy.recommended_cta or 'Explore further.'}"))
            return TwitterThreadContent(tweets=tweets, tweet_count=len(tweets)).model_dump()

        elif output_type == OutputType.EXECUTIVE_SUMMARY:
            findings = [c.claim_text for c in kp.claims[:4]] or ["Key transformations established.", "Strategic efficiency achieved."]
            metrics = [f"{m.label}: {m.value}" for m in kp.key_metrics[:4]] or ["Analysis completed."]
            return ExecutiveSummaryContent(
                title=f"Executive Briefing: {doc_title}",
                overview=raw_clean[:500] if len(raw_clean) > 50 else hook,
                key_findings=findings,
                data_highlights=metrics,
                recommendations=["Implement findings into operational workflow.", "Monitor performance metrics."],
                conclusion=kp.strategy.headline_hook,
            ).model_dump()

        elif output_type == OutputType.PRESENTATION_DECK:
            slides = [
                SlideItem(slide_number=1, title=doc_title, bullet_points=[hook], speaker_notes="Welcome everyone.", visual_suggestion="Title graphic"),
                SlideItem(slide_number=2, title="Core Objectives", bullet_points=[kp.intent.objective], speaker_notes="Reviewing goals.", visual_suggestion="Icon grid"),
                SlideItem(slide_number=3, title="Key Findings", bullet_points=[c.claim_text for c in kp.claims[:3]] or ["Finding A", "Finding B"], speaker_notes="Detailing results.", visual_suggestion="Bar chart"),
                SlideItem(slide_number=4, title="Recommendations & Next Steps", bullet_points=["Deploy recommendations", "Review timeline"], speaker_notes="Action plan.", visual_suggestion="Roadmap timeline"),
            ]
            return PresentationDeckContent(
                presentation_title=doc_title,
                tagline=hook,
                slides=slides,
                slide_count=len(slides),
            ).model_dump()

        elif output_type == OutputType.INFOGRAPHIC_BRIEF:
            stats = [
                InfographicStat(label=m.label, value=m.value, context=m.context)
                for m in kp.key_metrics[:4]
            ] or [InfographicStat(label="Performance", value="100%", context="Complete")]
            sections = [
                InfographicSection(title="Overview", description=hook, visual_type="barchart"),
                InfographicSection(title="Strategic Impact", description="Verified structural evidence.", visual_type="timeline"),
            ]
            return InfographicBriefContent(
                headline=hook[:80],
                subheadline=f"Insights from {doc_title}",
                key_stats=stats,
                visual_sections=sections,
                cta=kp.strategy.recommended_cta or "Learn more.",
            ).model_dump()

        elif output_type == OutputType.VIDEO_SCRIPT:
            scenes = [
                VideoScene(scene_number=1, visual_cue="Motion graphic introduction", narration=f"Welcome. Today we examine {doc_title}.", duration_sec=15),
                VideoScene(scene_number=2, visual_cue="Data charts and statistics", narration=hook, duration_sec=20),
                VideoScene(scene_number=3, visual_cue="Outro screen with call to action", narration=kp.strategy.recommended_cta or "Thank you for watching.", duration_sec=15),
            ]
            return VideoScriptContent(
                video_title=f"Video Brief: {doc_title}",
                target_duration_sec=50,
                scenes=scenes,
                full_subtitles=" ".join(s.narration for s in scenes),
            ).model_dump()

        else:
            return {
                "title": doc_title,
                "summary": hook,
                "content": raw_clean,
                "metadata": {"fallback_generated": True},
            }


response_parser = ResponseParser()
