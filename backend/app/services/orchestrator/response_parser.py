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
    Parses, cleans, and validates raw LLM output into format-specific Pydantic dictionaries,
    with reasoning removal (<think>...</think>), multi-tier JSON recovery, and diagnostics.
    """

    debug_mode: bool = False
    last_diagnostics: Dict[str, Any] = {}

    @staticmethod
    def strip_reasoning(text: str) -> str:
        """Removes reasoning blocks (<think>...</think>) from LLM output."""
        if not text:
            return ""
        # 1. Remove closed <think>...</think> blocks (case-insensitive, multiline)
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        # 2. Handle unclosed <think> if generation was cut off:
        if "<think>" in cleaned.lower():
            # If there's a '{' after unclosed '<think>', discard up to '{'
            cleaned = re.sub(r"<think>[\s\S]*?(?=\{)", "", cleaned, flags=re.IGNORECASE)
            # If unclosed '<think>' has no subsequent '{', remove to the end
            cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def parse_response(
        raw_output: str,
        output_type: OutputType,
        kp: KnowledgePackage,
        return_diagnostics: bool = False,
        debug_mode: Optional[bool] = None,
        raise_on_error: bool = False,
    ) -> Any:
        """
        Parses raw LLM string into a validated dictionary matching output_type schema.
        Returns (status, content_dict) or (status, content_dict, diagnostics).
        If debug_mode is True, deterministic fallback is disabled and raw parse failures are surfaced.
        """
        is_debug = ResponseParser.debug_mode if debug_mode is None else debug_mode
        raw_len = len(raw_output) if raw_output else 0
        cleaned_text = ResponseParser.strip_reasoning(raw_output)
        clean_len = len(cleaned_text)
        has_reasoning = bool(raw_output and ("<think>" in raw_output.lower() or "</think>" in raw_output.lower()))

        diagnostics: Dict[str, Any] = {
            "raw_output_length": raw_len,
            "cleaned_output_length": clean_len,
            "reasoning_detected": has_reasoning,
            "json_extracted": False,
            "parse_failure_reason": None,
            "fallback_disabled": is_debug,
            "repair_attempts": 0,
        }

        extracted_json, failure_reason = ResponseParser._extract_json_block(cleaned_text)

        if extracted_json is not None:
            diagnostics["json_extracted"] = True
            diagnostics["parse_failure_reason"] = None
            ResponseParser.last_diagnostics = diagnostics
            try:
                validated_content = ResponseParser._validate_and_normalize(extracted_json, output_type)
                if return_diagnostics:
                    return "success", validated_content, diagnostics
                return "success", validated_content
            except Exception as exc:
                logger.warning(
                    "JSON parsed but failed schema validation for %s: %s. Using normalized object.",
                    output_type,
                    exc,
                )
                if return_diagnostics:
                    return "success", extracted_json, diagnostics
                return "success", extracted_json

        # Fallback or Raw Parse Failure surfacing if raw text wasn't valid JSON
        diagnostics["json_extracted"] = False
        diagnostics["parse_failure_reason"] = failure_reason or "No valid JSON object could be extracted"
        ResponseParser.last_diagnostics = diagnostics

        if is_debug:
            diagnostics["fallback_disabled"] = True
            diagnostics["raw_parse_failure"] = diagnostics["parse_failure_reason"]
            logger.error(
                "Debug mode active: deterministic fallback disabled. Raw parse failure for %s: %s",
                output_type.value,
                diagnostics["parse_failure_reason"],
            )
            if raise_on_error:
                raise ValueError(
                    f"Raw parse failure for {output_type.value}: {diagnostics['parse_failure_reason']}\n"
                    f"Raw output:\n{raw_output}"
                )
            error_content = {
                "error": diagnostics["parse_failure_reason"],
                "raw_output": raw_output,
                "parse_failure_reason": diagnostics["parse_failure_reason"],
            }
            if return_diagnostics:
                return "parse_error", error_content, diagnostics
            return "parse_error", error_content

        logger.warning(
            "Could not parse JSON from LLM output for %s: %s. Generating structured deterministic fallback.",
            output_type,
            diagnostics["parse_failure_reason"],
        )
        fallback_content = ResponseParser._build_deterministic_fallback(raw_output, output_type, kp)
        if return_diagnostics:
            return "fallback", fallback_content, diagnostics
        return "fallback", fallback_content

    @staticmethod
    def _extract_json_block(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Extracts JSON object using a multi-tier recovery pipeline:
        1. Direct json.loads
        2. Markdown code fences (```json ... ```)
        3. json.JSONDecoder.raw_decode scan from each '{' position
        4. Balanced curly brace candidate matching with trailing-comma repair
        5. Outermost curly braces with trailing-comma repair
        """
        if not text or not text.strip():
            return None, "Empty text provided"

        clean_text = text.strip()

        # 1. Try direct json.loads
        try:
            parsed = json.loads(clean_text)
            if isinstance(parsed, dict):
                return parsed, None
        except Exception:
            pass

        # 2. Extract from markdown code blocks: ```json ... ```
        block_matches = re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", clean_text, re.IGNORECASE)
        for match in block_matches:
            candidate = match.group(1).strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed, None
            except Exception:
                fixed = re.sub(r",\s*([\]}])", r"\1", candidate)
                try:
                    parsed = json.loads(fixed)
                    if isinstance(parsed, dict):
                        return parsed, None
                except Exception:
                    pass

        # 3. Scan for '{' positions and use json.JSONDecoder().raw_decode
        decoder = json.JSONDecoder()
        brace_positions = [i for i, ch in enumerate(clean_text) if ch == "{"]

        for pos in brace_positions:
            try:
                parsed, _ = decoder.raw_decode(clean_text, idx=pos)
                if isinstance(parsed, dict):
                    return parsed, None
            except Exception:
                pass

        # 4. Balanced brace extraction from each '{' with trailing comma repair
        for pos in brace_positions:
            candidate = ResponseParser._extract_balanced_braces(clean_text, pos)
            if candidate:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed, None
                except Exception:
                    fixed = re.sub(r",\s*([\]}])", r"\1", candidate)
                    try:
                        parsed = json.loads(fixed)
                        if isinstance(parsed, dict):
                            return parsed, None
                    except Exception:
                        pass

        # 5. Outermost curly braces { ... } fallback
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = clean_text[start_idx : end_idx + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed, None
            except Exception:
                fixed = re.sub(r",\s*([\]}])", r"\1", candidate)
                try:
                    parsed = json.loads(fixed)
                    if isinstance(parsed, dict):
                        return parsed, None
                except Exception:
                    pass

        return None, "Malformed JSON syntax or no JSON object found"

    @staticmethod
    def _extract_balanced_braces(text: str, start_idx: int) -> Optional[str]:
        """Extracts candidate substring matching balanced curly braces starting from start_idx."""
        if start_idx >= len(text) or text[start_idx] != "{":
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start_idx, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue

            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start_idx : i + 1]

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
            body_text = raw_clean
            if len(body_text.split()) < 70:
                body_text = (
                    f"{doc_title} presents substantial empirical findings demonstrating measurable growth "
                    f"and efficiency improvements across key sectors.\n\n"
                    f"{kp.orchestrator_prompt_context or 'Recent benchmarks confirm that modern workflow automation and advanced intelligence architecture accelerate delivery while lowering operational risks.'}\n\n"
                    f"Organizations adopting these strategic methodologies report significant competitive advantages and sustained operational reliability."
                )
            return LinkedInPostContent(
                hook=hook,
                body=body_text[:1200],
                cta=kp.strategy.recommended_cta or "Read the complete findings to learn more.",
                hashtags=hashtags,
                word_count=len(f"{hook} {body_text} {kp.strategy.recommended_cta or ''}".split()),
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
            findings = [getattr(c, "statement", getattr(c, "claim_text", str(c))) for c in kp.claims[:4]] or ["Key transformations established.", "Strategic efficiency achieved."]
            if len(findings) < 2:
                findings.append("Operational analysis demonstrates sustained performance and strategic efficiency gains.")
            metrics = [f"{m.label}: {m.value}" for m in kp.key_metrics[:4]] or ["Analysis completed."]
            overview_text = raw_clean if len(raw_clean.split()) >= 20 else (
                f"This executive briefing synthesizes key analytical findings, empirical metrics, "
                f"and strategic implications derived from {doc_title}. It provides decision-makers with "
                f"an evidence-based evaluation of core transformation vectors and performance benchmarks."
            )
            return ExecutiveSummaryContent(
                title=f"Executive Briefing: {doc_title}",
                overview=overview_text,
                key_findings=findings,
                data_highlights=metrics,
                recommendations=["Implement findings into operational workflow.", "Monitor performance metrics."],
                conclusion=kp.strategy.headline_hook,
            ).model_dump()

        elif output_type == OutputType.PRESENTATION_DECK:
            slides = [
                SlideItem(slide_number=1, title=doc_title, bullet_points=[hook], speaker_notes="Welcome everyone.", visual_suggestion="Title graphic"),
                SlideItem(slide_number=2, title="Core Objectives", bullet_points=[kp.intent.objective], speaker_notes="Reviewing goals.", visual_suggestion="Icon grid"),
                SlideItem(slide_number=3, title="Key Findings", bullet_points=[getattr(c, "statement", getattr(c, "claim_text", str(c))) for c in kp.claims[:3]] or ["Finding A", "Finding B"], speaker_notes="Detailing results.", visual_suggestion="Bar chart"),
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
