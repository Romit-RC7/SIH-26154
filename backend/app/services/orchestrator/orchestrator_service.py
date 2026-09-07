"""
Content Orchestrator Service (Phase 4).
Coordinates multi-format content generation from KnowledgePackage using Qwen3-8B.
"""

import time
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.logging import logger
from backend.app.models.document import Document
from backend.app.schemas.intent import IntentAndPersonalization, OutputType
from backend.app.schemas.knowledge_package import KnowledgePackage
from backend.app.schemas.generated_artefact import (
    GeneratedArtefact,
    GenerateRequest,
    GenerateResponse,
)
from backend.app.services.knowledge_engine import knowledge_engine
from backend.app.services.orchestrator.prompt_builder import prompt_builder
from backend.app.services.orchestrator.qwen3_generation_service import qwen3_generation_service
from backend.app.services.orchestrator.response_parser import response_parser


class OrchestratorService:
    """
    Transforms documents and KnowledgePackages into multi-format deliverable artefacts.
    """

    async def generate_artefacts(
        self,
        document: Document,
        request: GenerateRequest,
        db: AsyncSession,
        kp_override: Optional[KnowledgePackage] = None,
    ) -> GenerateResponse:
        """
        Main orchestration entry point:
        1. Assembles KnowledgePackage (if not provided).
        2. Iterates over requested output types sequentially.
        3. Builds prompt, runs Qwen3-8B generation, and parses structured output.
        4. Returns all generated artefacts.
        """
        overall_start = time.time()
        output_types = request.output_types or [OutputType.EXECUTIVE_SUMMARY]

        # 1. Assemble or retrieve KnowledgePackage
        if kp_override is not None:
            kp = kp_override
        else:
            # Use first output type for intent configuration
            intent = IntentAndPersonalization(
                document_id=document.id,
                output_type=output_types[0],
                audience=request.audience,
                tone=request.tone,
                language=request.language,
                objective=request.objective,
                detail_level=request.detail_level,
                focus_keywords=request.focus_keywords,
                custom_instructions=request.custom_instructions,
            )
            logger.info("Assembling KnowledgePackage for document %s across %d output formats", document.id, len(output_types))
            kp = await knowledge_engine.assemble_knowledge(intent, document, db)

        # 2. Sequential generation for each selected output format
        artefacts: List[GeneratedArtefact] = []
        for out_type in output_types:
            logger.info("Generating deliverable artefact for format: %s", out_type.value)
            
            # Update intent output_type in KnowledgePackage for current format
            kp.intent.output_type = out_type
            prompt = prompt_builder.build_prompt(out_type, kp)

            raw_text, meta = qwen3_generation_service.generate(prompt, out_type)
            status, content_dict = response_parser.parse_response(raw_text, out_type, kp)

            artefact = GeneratedArtefact(
                artefact_id=f"art_{uuid.uuid4().hex[:12]}",
                document_id=document.id,
                output_type=out_type,
                status=status,
                content=content_dict,
                raw_llm_output=raw_text,
                generation_metadata=meta,
            )

            # --- Stage 5: Trust, Validation & Schema Enforcement ---
            try:
                from backend.app.services.validation.trust_service import trust_service
                verified = trust_service.validate_and_enforce(artefact, kp, auto_repair=True)
                artefact = verified.artefact
                artefact.generation_metadata["trust_score"] = verified.validation_report.trust_score
                artefact.generation_metadata["schema_compliance"] = verified.validation_report.schema_compliance
                artefact.generation_metadata["violations"] = verified.validation_report.violations
            except Exception as exc:
                logger.warning("Stage 5 Trust & Validation failed for %s (%s); proceeding with unverified artefact", artefact.artefact_id, exc)

            artefacts.append(artefact)

        total_time = round(time.time() - overall_start, 2)
        model_name = artefacts[0].generation_metadata.get("model", "Qwen3-8B") if artefacts else "Qwen3-8B"

        logger.info("Successfully generated %d artefacts for doc %s in %.2fs", len(artefacts), document.id, total_time)
        return GenerateResponse(
            document_id=document.id,
            artefacts=artefacts,
            total_generation_time_seconds=total_time,
            model_name=model_name,
        )


orchestrator_service = OrchestratorService()
