"""
Generated Artefact Schemas for Content Orchestrator (Phase 4).
Defines contracts for multi-format output artefacts produced by Qwen3-8B.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.intent import (
    OutputType,
    AudienceType,
    ToneType,
    DetailLevel,
    IntentAndPersonalization,
)


# --- Format-Specific Content Schemas ---

class LinkedInPostContent(BaseModel):
    hook: str = Field(..., description="Compelling opening hook sentence")
    body: str = Field(..., description="Main narrative body in clean, professional prose")
    cta: str = Field(..., description="Closing call-to-action")
    hashtags: List[str] = Field(default_factory=list, description="3-5 relevant industry hashtags")
    word_count: Optional[int] = Field(default=None, description="Estimated word count")


class TweetItem(BaseModel):
    index: int = Field(..., description="1-indexed tweet order number")
    text: str = Field(..., description="Tweet content text (under 280 characters)")
    char_count: Optional[int] = Field(default=None, description="Character count")


class TwitterThreadContent(BaseModel):
    tweets: List[TweetItem] = Field(default_factory=list, description="Sequence of 5-8 numbered tweets")
    tweet_count: Optional[int] = Field(default=None, description="Total number of tweets in thread")


class ExecutiveSummaryContent(BaseModel):
    title: str = Field(..., description="Formal executive briefing title")
    overview: str = Field(..., description="High-level strategic overview and purpose")
    key_findings: List[str] = Field(default_factory=list, description="Key empirical findings and facts")
    data_highlights: List[str] = Field(default_factory=list, description="Crucial quantitative metrics with citations")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    conclusion: str = Field(..., description="Concluding strategic assessment")


class SlideItem(BaseModel):
    slide_number: int = Field(..., description="Slide sequence number")
    title: str = Field(..., description="Concise slide title (max 8 words)")
    bullet_points: List[str] = Field(default_factory=list, description="3-5 scannable bullet points")
    speaker_notes: Optional[str] = Field(default=None, description="Detailed talking points for the presenter")
    visual_suggestion: Optional[str] = Field(default=None, description="Recommended diagram/chart/image layout")


class PresentationDeckContent(BaseModel):
    presentation_title: str = Field(..., description="Deck title")
    tagline: Optional[str] = Field(default=None, description="Subtitle or audience hook")
    slides: List[SlideItem] = Field(default_factory=list, description="5-10 structured presentation slides")
    slide_count: Optional[int] = Field(default=None, description="Total number of slides")


class InfographicStat(BaseModel):
    label: str = Field(..., description="Metric label (e.g. Model Accuracy)")
    value: str = Field(..., description="Prominent figure/number (e.g. 98.4%)")
    context: Optional[str] = Field(default=None, description="Brief explanatory context")


class InfographicSection(BaseModel):
    title: str = Field(..., description="Section title")
    description: str = Field(..., description="1-2 sentence scannable summary")
    visual_type: Optional[str] = Field(default="barchart", description="Suggested visual element (barchart, piechart, timeline, icon_grid)")


class InfographicBriefContent(BaseModel):
    headline: str = Field(..., description="Punchy infographic headline (max 10 words)")
    subheadline: Optional[str] = Field(default=None, description="Contextual subtitle")
    key_stats: List[InfographicStat] = Field(default_factory=list, description="3-6 focal quantitative statistics")
    visual_sections: List[InfographicSection] = Field(default_factory=list, description="3-5 thematic visual sections")
    cta: Optional[str] = Field(default=None, description="Call to action or source note")


class VideoScene(BaseModel):
    scene_number: int = Field(..., description="Scene index")
    narration: str = Field(..., description="Voiceover/spoken narration script (30-60 words)")
    visual_cue: str = Field(..., description="Visual scene description / B-roll instructions")
    duration_sec: int = Field(default=15, description="Estimated scene duration in seconds")


class VideoScriptContent(BaseModel):
    video_title: str = Field(..., description="Video title")
    target_duration_sec: int = Field(default=90, description="Total estimated video duration in seconds")
    scenes: List[VideoScene] = Field(default_factory=list, description="4-8 scene-by-scene storyboard components")
    full_subtitles: Optional[str] = Field(default=None, description="Complete concatenated subtitle text")


class BlogSection(BaseModel):
    heading: str = Field(..., description="Section H2 header")
    content: str = Field(..., description="Rich narrative section content")


class BlogPostContent(BaseModel):
    title: str = Field(..., description="Engaging blog title")
    meta_description: str = Field(..., description="SEO meta description under 160 characters")
    sections: List[BlogSection] = Field(default_factory=list, description="Structured thematic sections")
    conclusion: str = Field(..., description="Concluding remarks and takeaway")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")


# --- Top-level Artefact Container ---

class GeneratedArtefact(BaseModel):
    """
    Standardized Generated Artefact payload.
    Represents one transformational output generated by the Content Orchestrator.
    """
    artefact_id: str = Field(..., description="Unique UUID for this generated artefact")
    document_id: str = Field(..., description="Source document ID")
    output_type: OutputType = Field(..., description="Format type of this artefact")
    status: str = Field(default="success", description="Status: 'success', 'fallback', or 'parse_error'")
    content: Dict[str, Any] = Field(..., description="Structured content matching format schema")
    raw_llm_output: str = Field(default="", description="Unmodified LLM output for validation and audit")
    generation_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Telemetry metadata (latency_seconds, model_used, tokens, temperature)"
    )


# --- REST API Request & Response Schemas ---

class GenerateRequest(BaseModel):
    """Request payload for POST /api/v1/generate/{document_id}."""
    output_types: List[OutputType] = Field(
        default_factory=lambda: [OutputType.EXECUTIVE_SUMMARY],
        description="One or more target content deliverables to generate"
    )
    audience: AudienceType = Field(default=AudienceType.EXECUTIVE, description="Target reader persona")
    tone: ToneType = Field(default=ToneType.PROFESSIONAL, description="Tone of voice")
    detail_level: DetailLevel = Field(default=DetailLevel.MODERATE, description="Depth of coverage")
    language: str = Field(default="English", description="Target output language")
    objective: str = Field(
        default="Summarize key insights, data points, and strategic takeaways.",
        description="Primary communication objective"
    )
    focus_keywords: List[str] = Field(default_factory=list, description="Prioritized entities or themes")
    custom_instructions: Optional[str] = Field(default=None, description="Custom guidelines or stylistic constraints")
    debug_mode: Optional[bool] = Field(
        default=False,
        description="When True, disables deterministic fallback and surfaces raw parse failures for debugging"
    )


class GenerateResponse(BaseModel):
    """Response payload for POST /api/v1/generate/{document_id}."""
    document_id: str = Field(..., description="Target document ID")
    artefacts: List[GeneratedArtefact] = Field(default_factory=list, description="Generated deliverables")
    total_generation_time_seconds: float = Field(..., description="Total wall-clock generation time in seconds")
    model_name: str = Field(default="Qwen3-8B", description="Model engine used for generation")
