"""
Prompt Builder for Content Orchestration.
Constructs format-specific, evidence-grounded prompt payloads for Qwen3-8B generation.
"""

from typing import Dict, Any, Optional
from backend.app.schemas.intent import OutputType, IntentAndPersonalization
from backend.app.schemas.knowledge_package import KnowledgePackage


class PromptBuilder:
    """
    Constructs high-density, structured prompts tailored to the specified OutputType,
    injecting the compiled KnowledgePackage context, audience constraints, and JSON schemas.
    """

    @staticmethod
    def build_prompt(output_type: OutputType, kp: KnowledgePackage) -> str:
        """Dispatches to the appropriate format prompt generator."""
        generators = {
            OutputType.LINKEDIN_POST: PromptBuilder._build_linkedin_prompt,
            OutputType.TWITTER_THREAD: PromptBuilder._build_twitter_prompt,
            OutputType.EXECUTIVE_SUMMARY: PromptBuilder._build_executive_summary_prompt,
            OutputType.PRESENTATION_DECK: PromptBuilder._build_presentation_deck_prompt,
            OutputType.INFOGRAPHIC_BRIEF: PromptBuilder._build_infographic_prompt,
            OutputType.VIDEO_SCRIPT: PromptBuilder._build_video_script_prompt,
            OutputType.BLOG_POST: PromptBuilder._build_blog_post_prompt,
            OutputType.CUSTOM: PromptBuilder._build_custom_prompt,
        }
        gen_func = generators.get(output_type, PromptBuilder._build_custom_prompt)
        return gen_func(kp)

    @staticmethod
    def _base_context_header(kp: KnowledgePackage, format_name: str) -> str:
        intent: IntentAndPersonalization = kp.intent
        keywords_str = ", ".join(intent.focus_keywords) if intent.focus_keywords else "General Key Insights"
        custom_rules = f"\n- Additional Directives: {intent.custom_instructions}" if intent.custom_instructions else ""

        return (
            f"You are an expert AI content strategist and transformer. Your goal is to transform the provided "
            f"document evidence into a top-tier {format_name}.\n\n"
            f"=== EVIDENCE & FACTUAL CONTEXT (STRICT GROUNDING SOURCE) ===\n"
            f"Document: {kp.document_title or kp.document_id}\n"
            f"{kp.orchestrator_prompt_context}\n\n"
            f"=== USER INTENT & STRATEGY ===\n"
            f"- Target Audience: {intent.audience.value}\n"
            f"- Tone of Voice: {intent.tone.value}\n"
            f"- Detail Level: {intent.detail_level.value}\n"
            f"- Primary Objective: {intent.objective}\n"
            f"- Priority Keywords: {keywords_str}\n"
            f"- Strategy Hook: {kp.strategy.headline_hook}\n"
            f"- Recommended CTA: {kp.strategy.recommended_cta or 'Explore further.'}"
            f"{custom_rules}\n\n"
            f"=== CRITICAL INSTRUCTIONS ===\n"
            f"1. Factual Grounding: Use ONLY facts, metrics, and entities from the provided evidence. DO NOT hallucinate.\n"
            f"2. Format: Return ONLY a single raw JSON object matching the requested schema. No markdown backticks (no ```json), no intro or outro commentary.\n"
        )

    @staticmethod
    def _build_linkedin_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(kp, "LinkedIn Thought Leadership Post")
        rules = (
            f"=== FORMAT SPECIFIC RULES ===\n"
            f"- Length: 150 to 300 words total.\n"
            f"- Hook: Punchy, arresting first line that grabs attention.\n"
            f"- Body: 2-3 short, highly readable paragraphs highlighting core findings and quantitative data.\n"
            f"- CTA: 1 clear actionable call-to-action at the end.\n"
            f"- Hashtags: 3 to 5 relevant industry hashtags.\n\n"
            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "hook": "Single line opening hook",\n'
            f'  "body": "Multi-paragraph post text without hashtags",\n'
            f'  "cta": "Final sentence call to action",\n'
            f'  "hashtags": ["#AI", "#Innovation", "#Tech"],\n'
            f'  "word_count": 220\n'
            f"}}\n\n"
            f"JSON Output:"
        )
        return f"{header}\n{rules}"

    @staticmethod
    def _build_twitter_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(kp, "Twitter / X Insight Thread")
        rules = (
            f"=== FORMAT SPECIFIC RULES ===\n"
            f"- Number of tweets: Exactly 5 to 8 tweets.\n"
            f"- Length limit: EVERY tweet MUST be under 280 characters.\n"
            f"- Tweet 1: Headline hook + thread intro.\n"
            f"- Intermediate Tweets: One core metric, takeaway, or chart finding per tweet.\n"
            f"- Final Tweet: Summary takeaway or Call To Action.\n"
            f"- Include numbering at the beginning of each tweet (e.g. '1/6', '2/6').\n\n"
            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "tweets": [\n'
            f'    {{"index": 1, "text": "1/6 🚀 Headline hook with key finding... [under 280 chars]"}},\n'
            f'    {{"index": 2, "text": "2/6 📊 Key data point with specific numbers..."}}\n'
            f'  ],\n'
            f'  "tweet_count": 6\n'
            f"}}\n\n"
            f"JSON Output:"
        )
        return f"{header}\n{rules}"

    @staticmethod
    def _build_executive_summary_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(kp, "Executive Strategic Briefing")
        rules = (
            f"=== FORMAT SPECIFIC RULES ===\n"
            f"- Professional, authoritative tone suitable for decision-makers and C-suite leaders.\n"
            f"- Overview: Concise paragraph summarizing context, methodology, and primary theme.\n"
            f"- Key Findings: 3 to 5 comprehensive bullet statements.\n"
            f"- Data Highlights: 3 to 5 quantitative statements citing specific statistics.\n"
            f"- Recommendations: 3 to 5 actionable strategic steps.\n"
            f"- Conclusion: Forward-looking summary.\n\n"
            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "title": "Concise Strategic Title",\n'
            f'  "overview": "Comprehensive overview paragraph",\n'
            f'  "key_findings": ["Finding 1...", "Finding 2...", "Finding 3..."],\n'
            f'  "data_highlights": ["Metric 1: 98.4% accuracy achieved...", "Metric 2: $12M efficiency..."],\n'
            f'  "recommendations": ["Actionable step 1...", "Actionable step 2..."],\n'
            f'  "conclusion": "Final concluding perspective"\n'
            f"}}\n\n"
            f"JSON Output:"
        )
        return f"{header}\n{rules}"

    @staticmethod
    def _build_presentation_deck_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(kp, "Presentation Slide Deck")
        rules = (
            f"=== FORMAT SPECIFIC RULES ===\n"
            f"- Generate 5 to 10 structured slides.\n"
            f"- Slide 1: Title slide with compelling tagline.\n"
            f"- Body Slides: Focus on Problem, Data/Findings, Architecture/Methods, Impact.\n"
            f"- Final Slide: Conclusion / Next Steps / Call to Action.\n"
            f"- Each slide must contain: Title (max 8 words), 3-5 concise bullet points, speaker_notes (talking points), and visual_suggestion (e.g. 'Dual column bar chart').\n\n"
            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "presentation_title": "Executive Presentation Title",\n'
            f'  "tagline": "Strategic Subtitle / Hook",\n'
            f'  "slides": [\n'
            f'    {{\n'
            f'      "slide_number": 1,\n'
            f'      "title": "Title of Slide",\n'
            f'      "bullet_points": ["Point 1", "Point 2", "Point 3"],\n'
            f'      "speaker_notes": "Talking points for presenter...",\n'
            f'      "visual_suggestion": "Suggested visual layout"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "slide_count": 6\n'
            f"}}\n\n"
            f"JSON Output:"
        )
        return f"{header}\n{rules}"

    @staticmethod
    def _build_infographic_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(kp, "Infographic Blueprint & Key Takeaways")
        rules = (
            f"=== FORMAT SPECIFIC RULES ===\n"
            f"- Headline: Catchy, maximum 10 words.\n"
            f"- Key Stats: 3 to 6 high-impact quantitative callouts with short labels and bold values (e.g. label: 'Latency Drop', value: '45%').\n"
            f"- Visual Sections: 3 to 5 thematic sections with 1-2 sentence descriptions and a visual_type (barchart, piechart, timeline, comparison_table, flow_diagram).\n"
            f"- CTA: Bottom callout / takeaway.\n\n"
            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "headline": "Bold Infographic Headline",\n'
            f'  "subheadline": "Subtitle explaining the scope",\n'
            f'  "key_stats": [\n'
            f'    {{"label": "Model Accuracy", "value": "98.4%", "context": "On evaluation set"}},\n'
            f'    {{"label": "Processing Speed", "value": "2.4x", "context": "Faster than baseline"}}\n'
            f'  ],\n'
            f'  "visual_sections": [\n'
            f'    {{"title": "Core Breakthrough", "description": "Short explanation...", "visual_type": "barchart"}}\n'
            f'  ],\n'
            f'  "cta": "Explore the full report"\n'
            f"}}\n\n"
            f"JSON Output:"
        )
        return f"{header}\n{rules}"

    @staticmethod
    def _build_video_script_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(kp, "Video Package (Script, Storyboard & Visual Cues)")
        rules = (
            f"=== FORMAT SPECIFIC RULES ===\n"
            f"- Generate 4 to 8 storyboard scenes totaling 60-180 seconds.\n"
            f"- Scene 1: Hook and Problem statement.\n"
            f"- Intermediate Scenes: Evidence, Technology/Process, Quantitative Data Points, Visual Demonstrations.\n"
            f"- Final Scene: Conclusion, Takeaway, and Call to Action.\n"
            f"- Each scene must include narration (spoken script, 30-50 words), visual_cue (on-screen graphics/b-roll), and duration_sec (10-25s).\n"
            f"- Include full_subtitles (concatenation of all narrations).\n\n"
            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "video_title": "Engaging Video Title",\n'
            f'  "target_duration_sec": 90,\n'
            f'  "scenes": [\n'
            f'    {{\n'
            f'      "scene_number": 1,\n'
            f'      "visual_cue": "Fast-paced motion graphic introducing...",\n'
            f'      "narration": "Spoken voiceover script for scene 1...",\n'
            f'      "duration_sec": 15\n'
            f'    }}\n'
            f'  ],\n'
            f'  "full_subtitles": "Complete subtitle text..."\n'
            f"}}\n\n"
            f"JSON Output:"
        )
        return f"{header}\n{rules}"

    @staticmethod
    def _build_blog_post_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(kp, "In-Depth Blog Post / Technical Article")
        rules = (
            f"=== FORMAT SPECIFIC RULES ===\n"
            f"- Title: Engaging, SEO-friendly headline.\n"
            f"- Meta Description: Under 160 characters summary.\n"
            f"- Sections: 3 to 5 comprehensive thematic sections with descriptive H2 headings.\n"
            f"- Conclusion: Final perspective.\n"
            f"- Tags: 3 to 5 categorization tags.\n\n"
            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "title": "Blog Post Title",\n'
            f'  "meta_description": "SEO meta description...",\n'
            f'  "sections": [\n'
            f'    {{"heading": "Section Heading", "content": "Detailed paragraph text..."}}\n'
            f'  ],\n'
            f'  "conclusion": "Final thoughts...",\n'
            f'  "tags": ["AI", "Transformation"]\n'
            f"}}\n\n"
            f"JSON Output:"
        )
        return f"{header}\n{rules}"

    @staticmethod
    def _build_custom_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(kp, "Structured Content Deliverable")
        rules = (
            f"=== FORMAT SPECIFIC RULES ===\n"
            f"- Deliverable format: {kp.intent.output_type.value}\n"
            f"- Synthesize the document findings into a clean, structured output.\n\n"
            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "title": "Deliverable Title",\n'
            f'  "summary": "Core summary",\n'
            f'  "details": "Comprehensive content text",\n'
            f'  "key_points": ["Point 1", "Point 2"],\n'
            f'  "recommended_actions": ["Action 1"]\n'
            f"}}\n\n"
            f"JSON Output:"
        )
        return f"{header}\n{rules}"


prompt_builder = PromptBuilder()
