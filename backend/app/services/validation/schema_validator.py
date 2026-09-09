"""
Schema Validator Service.
Enforces format-specific structural rules and constraints across generated content.
"""

from typing import List, Dict, Any, Tuple
from backend.app.core.logging import logger
from backend.app.schemas.intent import OutputType
from backend.app.schemas.generated_artefact import GeneratedArtefact


class SchemaValidator:
    """
    Validates structural format constraints (word counts, char limits, required fields).
    """

    def validate_schema(self, artefact: GeneratedArtefact) -> Tuple[bool, List[str]]:
        """
        Validates an artefact against format-specific rules.
        Returns (is_compliant: bool, violations: List[str]).
        """
        violations: List[str] = []
        content = artefact.content or {}
        out_type = artefact.output_type

        validators = {
            OutputType.LINKEDIN_POST: self._validate_linkedin,
            OutputType.TWITTER_THREAD: self._validate_twitter,
            OutputType.EXECUTIVE_SUMMARY: self._validate_executive_summary,
            OutputType.PRESENTATION_DECK: self._validate_presentation_deck,
            OutputType.INFOGRAPHIC_BRIEF: self._validate_infographic,
            OutputType.VIDEO_SCRIPT: self._validate_video_script,
            OutputType.BLOG_POST: self._validate_blog_post,
        }

        val_func = validators.get(out_type, self._validate_generic)
        violations = val_func(content)

        is_compliant = len(violations) == 0
        if not is_compliant:
            logger.warning(
                "Schema validation failed for artefact %s (%s) with %d violation(s):\n%s",
                artefact.artefact_id,
                out_type.value,
                len(violations),
                "\n".join(f"  - {viol}" for viol in violations),
            )
        else:
            logger.info(
                "Schema validation for artefact %s (%s): compliant=True, violations=0",
                artefact.artefact_id,
                out_type.value,
            )
        return is_compliant, violations

    def _validate_linkedin(self, content: Dict[str, Any]) -> List[str]:
        v: List[str] = []

        # 1. Field presence and type checks
        if "hook" not in content or not content.get("hook"):
            v.append("Missing field: hook")
        elif not isinstance(content["hook"], str):
            v.append("Invalid field type: hook")
        elif len(content["hook"].strip()) < 10:
            v.append("LinkedIn post is missing a strong hook sentence (minimum 10 characters).")

        if "body" not in content or not content.get("body"):
            v.append("Missing field: body")
        elif not isinstance(content["body"], str):
            v.append("Invalid field type: body")

        if "cta" not in content or not content.get("cta"):
            v.append("Missing field: cta")
        elif not isinstance(content["cta"], str):
            v.append("Invalid field type: cta")
        elif len(content["cta"].strip()) < 10:
            v.append("LinkedIn post is missing a clear call-to-action (CTA).")

        if "hashtags" not in content:
            v.append("Missing field: hashtags")
        elif not isinstance(content["hashtags"], list):
            v.append("Invalid field type: hashtags")
        elif len(content["hashtags"]) < 2:
            v.append("LinkedIn post must contain at least 2 hashtags.")
        elif any(not isinstance(h, str) for h in content["hashtags"]):
            v.append("Invalid field type: hashtags")

        # 2. Word count bounds
        hook_str = str(content.get("hook", "")) if isinstance(content.get("hook"), str) else ""
        body_str = str(content.get("body", "")) if isinstance(content.get("body"), str) else ""
        cta_str = str(content.get("cta", "")) if isinstance(content.get("cta"), str) else ""
        words = len(f"{hook_str} {body_str} {cta_str}".split())

        if words < 80:
            v.append(f"Word count below minimum: LinkedIn post is too short ({words} words; minimum 80 words).")
        elif words > 400:
            v.append(f"Word count above maximum: LinkedIn post is too long ({words} words; maximum 400 words).")

        return v

    def _validate_twitter(self, content: Dict[str, Any]) -> List[str]:
        v: List[str] = []
        tweets = content.get("tweets", [])

        if not tweets or len(tweets) < 3:
            v.append(f"Twitter thread contains too few tweets ({len(tweets)}; minimum 3 required).")

        for idx, tweet in enumerate(tweets, start=1):
            text = tweet.get("text", "") if isinstance(tweet, dict) else str(tweet)
            if not text:
                v.append(f"Tweet #{idx} is empty.")
            elif len(text) > 280:
                v.append(f"Tweet #{idx} exceeds 280 characters limit (length: {len(text)} chars).")

        return v

    def _validate_executive_summary(self, content: Dict[str, Any]) -> List[str]:
        v: List[str] = []
        title = content.get("title", "")
        overview = content.get("overview", "")
        key_findings = content.get("key_findings", [])
        recommendations = content.get("recommendations", [])

        if not title:
            v.append("Executive summary is missing a title.")
        if not overview or len(overview.split()) < 20:
            v.append("Executive summary overview is insufficient or missing.")
        if not key_findings or len(key_findings) < 2:
            v.append("Executive summary requires at least 2 key findings.")
        if not recommendations or len(recommendations) < 1:
            v.append("Executive summary requires at least 1 recommendation.")

        return v

    def _validate_presentation_deck(self, content: Dict[str, Any]) -> List[str]:
        v: List[str] = []
        slides = content.get("slides", [])

        if not slides or len(slides) < 3:
            v.append(f"Presentation deck has too few slides ({len(slides)}; minimum 3 required).")

        for idx, slide in enumerate(slides, start=1):
            if isinstance(slide, dict):
                s_title = slide.get("title", "")
                bullets = slide.get("bullet_points", [])
                if not s_title:
                    v.append(f"Slide #{idx} is missing a title.")
                elif len(s_title.split()) > 14:
                    v.append(f"Slide #{idx} title is too long ({len(s_title.split())} words; max 14).")
                if not bullets or len(bullets) < 1:
                    v.append(f"Slide #{idx} contains no bullet points.")

        return v

    def _validate_infographic(self, content: Dict[str, Any]) -> List[str]:
        v: List[str] = []
        headline = content.get("headline", "")
        key_stats = content.get("key_stats", [])
        visual_sections = content.get("visual_sections", [])

        if not headline:
            v.append("Infographic brief missing a headline.")
        elif len(headline.split()) > 15:
            v.append(f"Infographic headline is too long ({len(headline.split())} words; max 15).")

        if not key_stats or len(key_stats) < 2:
            v.append("Infographic brief must contain at least 2 key statistics.")

        if not visual_sections or len(visual_sections) < 1:
            v.append("Infographic brief must contain at least 1 visual section.")

        return v

    def _validate_video_script(self, content: Dict[str, Any]) -> List[str]:
        v: List[str] = []
        scenes = content.get("scenes", [])

        if not scenes or len(scenes) < 3:
            v.append(f"Video script contains too few scenes ({len(scenes)}; minimum 3 required).")

        for idx, scene in enumerate(scenes, start=1):
            if isinstance(scene, dict):
                narration = scene.get("narration", "")
                visual_cue = scene.get("visual_cue", "")
                if not narration:
                    v.append(f"Scene #{idx} missing narration script.")
                if not visual_cue:
                    v.append(f"Scene #{idx} missing visual cue.")

        return v

    def _validate_blog_post(self, content: Dict[str, Any]) -> List[str]:
        v: List[str] = []
        title = content.get("title", "")
        sections = content.get("sections", [])

        if not title:
            v.append("Blog post missing title.")
        if not sections or len(sections) < 2:
            v.append("Blog post must contain at least 2 sections.")

        return v

    def _validate_generic(self, content: Dict[str, Any]) -> List[str]:
        if not content or len(content) == 0:
            return ["Artefact content is empty."]
        return []


schema_validator = SchemaValidator()
