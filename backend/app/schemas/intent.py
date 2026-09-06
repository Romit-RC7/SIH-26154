"""
Intent and Personalization Schema.
Captures user configuration (format, audience, tone, language, objective, detail level)
to guide Knowledge Engine retrieval and Content Orchestration context assembly.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class OutputType(str, Enum):
    LINKEDIN_POST = "linkedin_post"
    TWITTER_THREAD = "twitter_thread"
    EXECUTIVE_SUMMARY = "executive_summary"
    PRESENTATION_DECK = "presentation_deck"
    INFOGRAPHIC_BRIEF = "infographic_brief"
    VIDEO_SCRIPT = "video_script"
    BLOG_POST = "blog_post"
    CUSTOM = "custom"


class AudienceType(str, Enum):
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    GENERAL_PUBLIC = "general_public"
    INVESTORS = "investors"
    STUDENTS = "students"
    PRACTITIONERS = "practitioners"
    POLICY_MAKERS = "policy_makers"
    DEVELOPERS = "developers"


class ToneType(str, Enum):
    PROFESSIONAL = "professional"
    PERSUASIVE = "persuasive"
    ACADEMIC = "academic"
    CASUAL = "casual"
    AUTHORITATIVE = "authoritative"
    INSPIRATIONAL = "inspirational"
    ANALYTICAL = "analytical"
    ENGAGING = "engaging"


class DetailLevel(str, Enum):
    CONCISE = "concise"
    MODERATE = "moderate"
    COMPREHENSIVE = "comprehensive"
    DEEP_DIVE = "deep_dive"


class IntentAndPersonalization(BaseModel):
    """
    User-specified intent and personalization parameters for transforming document content.
    """
    document_id: str = Field(..., description="Target document ID to transform")
    output_type: OutputType = Field(
        default=OutputType.EXECUTIVE_SUMMARY,
        description="Target content output format"
    )
    audience: AudienceType = Field(
        default=AudienceType.EXECUTIVE,
        description="Target reader or viewer persona"
    )
    tone: ToneType = Field(
        default=ToneType.PROFESSIONAL,
        description="Tone of voice and stylistic delivery"
    )
    language: str = Field(
        default="English",
        description="Target output language"
    )
    objective: str = Field(
        default="Summarize key insights, data points, and recommendations.",
        description="Primary goal or core thesis for the generated content"
    )
    detail_level: DetailLevel = Field(
        default=DetailLevel.MODERATE,
        description="Depth and granularity of generated content"
    )
    focus_keywords: List[str] = Field(
        default_factory=list,
        description="Key topics, entities, or keywords to prioritize"
    )
    custom_instructions: Optional[str] = Field(
        default=None,
        description="Specific guidelines, constraints, or stylistic rules"
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom parameters for specialized templates"
    )
