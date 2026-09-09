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

        keywords_str = (
            ", ".join(intent.focus_keywords)
            if intent.focus_keywords
            else "General Key Insights"
        )

        custom_rules = (
            f"\n- Additional Directives: {intent.custom_instructions}"
            if intent.custom_instructions
            else ""
        )

        return (
            f"You are a senior industry analyst, executive communications advisor, "
            f"and content strategist.\n\n"

            f"Your task is to transform evidence into a high-quality "
            f"{format_name} suitable for the intended audience.\n"
            f"Every statement must be grounded in the supplied evidence.\n\n"

            f"=== DOCUMENT CONTEXT ===\n"
            f"Document Title: {kp.document_title or kp.document_id}\n"
            f"IMPORTANT: The document title is metadata only. "
            f"Generate content from the evidence below, not from the title.\n\n"

            f"=== EVIDENCE & FACTUAL CONTEXT (PRIMARY SOURCE OF TRUTH) ===\n"
            f"{kp.orchestrator_prompt_context}\n\n"

            f"=== USER INTENT & STRATEGY ===\n"
            f"- Target Audience: {intent.audience.value}\n"
            f"- Tone of Voice: {intent.tone.value}\n"
            f"- Detail Level: {intent.detail_level.value}\n"
            f"- Primary Objective: {intent.objective}\n"
            f"- Priority Keywords: {keywords_str}\n"
            f"- Strategy Hook: {kp.strategy.headline_hook}\n"
            f"- Recommended CTA: "
            f"{kp.strategy.recommended_cta or 'Explore further.'}"
            f"{custom_rules}\n\n"

            f"=== EVIDENCE PRIORITY ORDER ===\n"
            f"1. Quantitative findings and metrics\n"
            f"2. Named entities, organizations, technologies, products\n"
            f"3. Key findings and conclusions\n"
            f"4. Strategic implications\n"
            f"5. Recommendations and actions\n\n"

            f"=== CONTENT QUALITY REQUIREMENTS ===\n"
            f"- Write like a domain expert, analyst, or executive advisor.\n"
            f"- Avoid generic summaries.\n"
            f"- Avoid repeating the document title.\n"
            f"- Explain why findings matter.\n"
            f"- Focus on insight, evidence, impact, and actionability.\n"
            f"- Prefer specific facts over broad statements.\n"
            f"- Use concrete evidence whenever available.\n"
            f"- If evidence is weak or missing, do not invent information.\n\n"

            f"=== FACTUAL SAFETY RULES ===\n"
            f"- Use ONLY facts present in the evidence.\n"
            f"- Do NOT invent statistics.\n"
            f"- Do NOT invent percentages.\n"
            f"- Do NOT invent financial values.\n"
            f"- Do NOT invent dates, organizations, or initiatives.\n"
            f"- If a metric is not present in the evidence, do not create one.\n"
            f"- If evidence is insufficient, use a grounded qualitative statement instead.\n\n"

            f"=== OUTPUT REQUIREMENTS ===\n"
            f"- Return ONLY valid JSON.\n"
            f"- Do not include explanations.\n"
            f"- Do not include markdown.\n"
            f"- Do not include code fences.\n"
            f"- Do not include reasoning.\n"
            f"- Do not include <think> blocks.\n"
            f"- Do not include analysis.\n"
            f"- Output must exactly match the required schema.\n"
        )

    @staticmethod
    def _build_linkedin_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(
            kp,
            "LinkedIn Thought Leadership Post"
        )

        rules = (
            f"=== FORMAT SPECIFIC RULES ===\n"
            f"- Length: 150 to 300 words total.\n\n"

            f"=== CONTENT OBJECTIVE ===\n"
            f"Generate a professional LinkedIn thought-leadership post.\n"
            f"Do NOT simply summarize the document.\n"
            f"Identify the most important insight, finding, trend, opportunity, risk, or strategic takeaway contained in the evidence.\n"
            f"Explain why it matters and what action leaders should consider.\n\n"

            f"=== HOOK REQUIREMENTS ===\n"
            f"- Start with a compelling insight, challenge, trend, opportunity, or observation.\n"
            f"- Must be interesting to a professional audience.\n"
            f"- Must NOT begin with document titles, filenames, project names, IDs, codes, or metadata.\n\n"

            f"=== BODY REQUIREMENTS ===\n"
            f"- Use 2-3 short, highly readable paragraphs.\n"
            f"- Focus on business impact, operational impact, strategic implications, risks, opportunities, or key lessons.\n"
            f"- Include quantitative evidence only when it strengthens the insight.\n"
            f"- Explain why the finding matters.\n"
            f"- Include one actionable takeaway.\n"
            f"- Do NOT list technologies, tools, frameworks, or implementation details unless they are directly relevant to the insight.\n"
            f"- Avoid repeating raw document metadata.\n\n"

            f"=== QUALITY RUBRIC ===\n"
            f"A high-quality LinkedIn post:\n"
            f"- Provides insight, not a document summary.\n"
            f"- Focuses on outcomes rather than technology lists.\n"
            f"- Explains implications and recommendations.\n"
            f"- Reads naturally as professional thought leadership.\n"
            f"- Avoids internal identifiers, ticket numbers, filenames, and project codes.\n\n"

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

            f"JSON Output (Return ONLY a single valid JSON object. "
            f"No explanations, no markdown, no <think> blocks):"
        )

        return f"{header}\n{rules}"

    @staticmethod
    def _build_twitter_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(
            kp,
            "Twitter / X Insight Thread"
        )

        rules = (
            f"=== CONTENT OBJECTIVE ===\n"
            f"Generate a high-signal Twitter/X thread that delivers insights, not a document summary.\n"
            f"Identify the most important finding, trend, opportunity, risk, lesson, or recommendation from the evidence.\n"
            f"Focus on why it matters and what readers should learn from it.\n\n"

            f"=== FORMAT RULES ===\n"
            f"- Number of tweets: Exactly 5 to 8 tweets.\n"
            f"- EVERY tweet MUST be under 280 characters.\n"
            f"- Include numbering at the beginning of each tweet (e.g. '1/6', '2/6').\n"
            f"- Tweet 1: Strong hook and thread introduction.\n"
            f"- Tweets 2 through N-1: One meaningful insight, lesson, statistic, implication, risk, opportunity, or recommendation per tweet.\n"
            f"- Final Tweet: Strategic takeaway or clear call to action.\n\n"

            f"=== QUALITY REQUIREMENTS ===\n"
            f"- Do NOT summarize the document sequentially.\n"
            f"- Do NOT create a technology list.\n"
            f"- Do NOT repeat document metadata, IDs, filenames, ticket numbers, or project codes.\n"
            f"- Use quantitative evidence only when it strengthens the insight.\n"
            f"- Prioritize clarity, usefulness, and shareability.\n"
            f"- Every tweet should add new value.\n\n"

            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "tweets": [\n'
            f'    {{"index": 1, "text": "1/6 Strong opening insight or hook..."}},\n'
            f'    {{"index": 2, "text": "2/6 Supporting insight, evidence, or implication..."}}\n'
            f'  ],\n'
            f'  "tweet_count": 6\n'
            f"}}\n\n"

            f"JSON Output (Return ONLY a single valid JSON object. "
            f"No explanations, no markdown, no <think> blocks):"
        )

        return f"{header}\n{rules}"

    @staticmethod
    def _build_executive_summary_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(
            kp,
            "Executive Strategic Briefing"
        )

        rules = (
            f"=== CONTENT OBJECTIVE ===\n"
            f"Create a decision-oriented executive briefing for senior leaders.\n"
            f"Focus on findings that influence strategy, investment, operations, risk, growth, or competitive advantage.\n"
            f"Do NOT simply summarize the source material.\n"
            f"Prioritize insights and implications over implementation details.\n\n"

            f"=== FORMAT SPECIFIC RULES ===\n"
            f"- Professional and authoritative tone suitable for executives and decision-makers.\n"
            f"- Overview: Concise context and primary theme.\n"
            f"- Key Findings: 3 to 5 high-impact observations.\n"
            f"- Data Highlights: 3 to 5 evidence-backed quantitative findings.\n"
            f"- Recommendations: 3 to 5 actionable strategic recommendations.\n"
            f"- Conclusion: Forward-looking executive perspective.\n\n"

            f"=== QUALITY REQUIREMENTS ===\n"
            f"- Separate findings from recommendations.\n"
            f"- Explain why findings matter.\n"
            f"- Focus on business impact, risk, efficiency, growth, innovation, or opportunity.\n"
            f"- Avoid document metadata, IDs, filenames, and project codes.\n"
            f"- Avoid low-level implementation details unless strategically important.\n\n"

            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "title": "Concise Strategic Title",\n'
            f'  "overview": "Comprehensive overview paragraph",\n'
            f'  "key_findings": ["Finding 1...", "Finding 2...", "Finding 3..."],\n'
            f'  "data_highlights": ["Metric 1...", "Metric 2..."],\n'
            f'  "recommendations": ["Actionable step 1...", "Actionable step 2..."],\n'
            f'  "conclusion": "Final concluding perspective"\n'
            f"}}\n\n"

            f"JSON Output (Return ONLY a single valid JSON object. "
            f"No explanations, no markdown, no <think> blocks):"
        )

        return f"{header}\n{rules}"

    @staticmethod
    def _build_presentation_deck_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(
            kp,
            "Presentation Slide Deck"
        )

        rules = (
            f"=== CONTENT OBJECTIVE ===\n"
            f"Create a presentation that tells a compelling strategic story.\n"
            f"The audience should understand:\n"
            f"1. Why this matters.\n"
            f"2. What was discovered.\n"
            f"3. Why it is important.\n"
            f"4. What actions should be taken.\n\n"

            f"=== NARRATIVE FLOW ===\n"
            f"- Slide 1: Executive hook and context.\n"
            f"- Slide 2: Problem, challenge, or opportunity.\n"
            f"- Slide 3-5: Key findings and supporting evidence.\n"
            f"- Slide 6-8: Strategic implications, risks, opportunities, or impact.\n"
            f"- Final Slide: Recommendations, next steps, and call to action.\n\n"

            f"=== FORMAT SPECIFIC RULES ===\n"
            f"- Generate 5 to 10 structured slides.\n"
            f"- Each slide title must be 8 words or fewer.\n"
            f"- Each slide must contain 3-5 concise bullet points.\n"
            f"- Speaker notes should explain the key message, not repeat bullets.\n"
            f"- Visual suggestions should reinforce the insight being presented.\n\n"

            f"=== QUALITY REQUIREMENTS ===\n"
            f"- Do NOT create slides that merely mirror document sections.\n"
            f"- Focus on decisions, insights, evidence, and recommendations.\n"
            f"- Avoid document metadata, IDs, filenames, and project codes.\n"
            f"- Prioritize business impact over technical implementation details.\n"
            f"- Ensure slides form a coherent narrative.\n\n"

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

            f"JSON Output (Return ONLY a single valid JSON object. "
            f"No explanations, no markdown, no <think> blocks):"
        )

        return f"{header}\n{rules}"


    @staticmethod
    def _build_infographic_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(
            kp,
            "Executive Infographic Blueprint"
        )

        rules = (
            f"=== OBJECTIVE ===\n"
            f"Convert the provided evidence into an infographic blueprint suitable for executives and decision-makers.\n"
            f"Highlight only the most important insights, metrics, trends, risks, opportunities, and outcomes.\n\n"

            f"=== CONTENT RULES ===\n"
            f"- Headline must communicate the primary insight, not just the topic.\n"
            f"- Subheadline should explain why the information matters.\n"
            f"- Use only metrics explicitly found in the source evidence.\n"
            f"- Do not invent percentages, revenue figures, timelines, accuracy numbers, or KPIs.\n"
            f"- Prioritize business impact, strategic value, efficiency gains, risk reduction, adoption trends, or performance outcomes.\n"
            f"- Each visual section should tell a distinct story.\n"
            f"- Visual sections should collectively answer:\n"
            f"  1. What problem exists?\n"
            f"  2. What was discovered?\n"
            f"  3. Why does it matter?\n"
            f"  4. What should happen next?\n"
            f"- Avoid repeating information across sections.\n\n"

            f"=== VISUAL DESIGN GUIDANCE ===\n"
            f"- Use visual types only when appropriate.\n"
            f"- visual_type must be one of:\n"
            f"  barchart, piechart, timeline, comparison_table, flow_diagram\n"
            f"- Select the visual type that best represents the evidence.\n\n"

            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "headline": "Insight-driven headline",\n'
            f'  "subheadline": "Why this matters",\n'
            f'  "key_stats": [\n'
            f'    {{"label": "Metric Name", "value": "Value", "context": "Meaning of metric"}}\n'
            f'  ],\n'
            f'  "visual_sections": [\n'
            f'    {{\n'
            f'      "title": "Section title",\n'
            f'      "description": "Executive-level explanation",\n'
            f'      "visual_type": "barchart"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "cta": "Final takeaway or recommendation"\n'
            f"}}\n\n"

            f"Return ONLY valid JSON."
        )

        return f"{header}\n{rules}"

    @staticmethod
    def _build_video_script_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(
            kp,
            "Video Package (Executive Storyboard)"
        )

        rules = (
            f"=== OBJECTIVE ===\n"
            f"Transform the evidence into a compelling professional video narrative.\n"
            f"The viewer should understand the problem, evidence, solution, impact, and next action.\n\n"

            f"=== STORY STRUCTURE ===\n"
            f"- Scene 1: Attention-grabbing hook.\n"
            f"- Scene 2: Problem or challenge.\n"
            f"- Scene 3-6: Evidence, findings, technology, process, or key insights.\n"
            f"- Final Scene: Strategic takeaway and call to action.\n\n"

            f"=== CONTENT RULES ===\n"
            f"- Every scene must move the story forward.\n"
            f"- Narration should sound natural when spoken aloud.\n"
            f"- Avoid bullet-point narration.\n"
            f"- Use concise professional language.\n"
            f"- Include quantitative evidence only if present in source material.\n"
            f"- Never invent metrics or claims.\n"
            f"- Explain why findings matter, not just what happened.\n"
            f"- Visual cues should clearly support the narration.\n\n"

            f"=== SCENE RULES ===\n"
            f"- Generate 4 to 8 scenes.\n"
            f"- Each narration: 30-60 words.\n"
            f"- Each scene duration: 10-25 seconds.\n"
            f"- Total duration: 60-180 seconds.\n\n"

            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "video_title": "Compelling title",\n'
            f'  "target_duration_sec": 90,\n'
            f'  "scenes": [\n'
            f'    {{\n'
            f'      "scene_number": 1,\n'
            f'      "visual_cue": "Visual direction",\n'
            f'      "narration": "Professional voiceover text",\n'
            f'      "duration_sec": 15\n'
            f'    }}\n'
            f'  ],\n'
            f'  "full_subtitles": "Combined narration text"\n'
            f"}}\n\n"

            f"Return ONLY valid JSON."
        )

        return f"{header}\n{rules}"

    @staticmethod
    def _build_blog_post_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(
            kp,
            "In-Depth Blog Post / Technical Article"
        )

        rules = (
            f"=== OBJECTIVE ===\n"
            f"Create a professional long-form article that educates the reader while clearly explaining the significance of the findings.\n\n"

            f"=== CONTENT RULES ===\n"
            f"- Write for an informed professional audience.\n"
            f"- Focus on insights, implications, lessons, challenges, and opportunities.\n"
            f"- Use only facts present in the provided evidence.\n"
            f"- Do not invent statistics, benchmarks, performance figures, timelines, or claims.\n"
            f"- Each section should build logically on the previous section.\n"
            f"- Avoid repeating the same information.\n"
            f"- Explain not only what happened, but why it matters.\n"
            f"- Use clear and authoritative language.\n\n"

            f"=== ARTICLE STRUCTURE ===\n"
            f"- Strong SEO-friendly title.\n"
            f"- Meta description under 160 characters.\n"
            f"- 3 to 5 sections.\n"
            f"- Each section should have a meaningful H2 heading.\n"
            f"- Each section should contain 2-4 well-developed paragraphs.\n"
            f"- End with a strategic conclusion.\n\n"

            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "title": "SEO-friendly article title",\n'
            f'  "meta_description": "SEO summary under 160 characters",\n'
            f'  "sections": [\n'
            f'    {{\n'
            f'      "heading": "Section heading",\n'
            f'      "content": "Detailed section content"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "conclusion": "Strategic conclusion",\n'
            f'  "tags": ["Tag1", "Tag2"]\n'
            f"}}\n\n"

            f"Return ONLY valid JSON."
        )

        return f"{header}\n{rules}"

    @staticmethod
    def _build_custom_prompt(kp: KnowledgePackage) -> str:
        header = PromptBuilder._base_context_header(
            kp,
            "Structured Content Deliverable"
        )

        rules = (
            f"=== OBJECTIVE ===\n"
            f"Generate the highest-quality structured deliverable possible based on the provided evidence.\n\n"

            f"=== CONTENT RULES ===\n"
            f"- Focus on the most important findings, insights, recommendations, risks, opportunities, and outcomes.\n"
            f"- Use only information supported by the source material.\n"
            f"- Never invent metrics, statistics, claims, or conclusions.\n"
            f"- Prioritize clarity, usefulness, and business value.\n"
            f"- Summarize information intelligently rather than copying source text.\n"
            f"- Highlight implications and recommended actions whenever appropriate.\n\n"

            f"=== WRITING STYLE ===\n"
            f"- Professional and concise.\n"
            f"- Insight-driven rather than descriptive.\n"
            f"- Suitable for business, technical, or executive audiences.\n\n"

            f"=== REQUIRED JSON SCHEMA ===\n"
            f"{{\n"
            f'  "title": "Deliverable title",\n'
            f'  "summary": "Executive summary of the most important insights",\n'
            f'  "details": "Comprehensive structured content",\n'
            f'  "key_points": [\n'
            f'    "Key point 1",\n'
            f'    "Key point 2"\n'
            f'  ],\n'
            f'  "recommended_actions": [\n'
            f'    "Action 1",\n'
            f'    "Action 2"\n'
            f'  ]\n'
            f"}}\n\n"

            f"Return ONLY valid JSON."
        )

        return f"{header}\n{rules}"

prompt_builder = PromptBuilder()
