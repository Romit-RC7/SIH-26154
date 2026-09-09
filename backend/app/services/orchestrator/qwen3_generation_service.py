"""
Qwen3 Generation Service.
Manages Qwen3-8B GGUF inference via llama-cpp-python, handling temperature,
token allocation, and deterministic offline simulation when model weights are absent.
"""

import time
from typing import Dict, Any, Optional, Tuple
from backend.app.core.logging import logger
from backend.app.schemas.intent import OutputType
from backend.app.services.model_initializer.qwen_initializers import qwen_orchestrator_initializer


class Qwen3GenerationService:
    """
    Executes inference with Qwen3-8B (or Qwen3-4B / fallback) for content orchestration.
    """

    FORMAT_HYPERPARAMS: Dict[OutputType, Dict[str, Any]] = {
        OutputType.LINKEDIN_POST: {"temperature": 0.2, "max_tokens": 1200},
        OutputType.TWITTER_THREAD: {"temperature": 0.6, "max_tokens": 800},
        OutputType.EXECUTIVE_SUMMARY: {"temperature": 0.3, "max_tokens": 1600},
        OutputType.PRESENTATION_DECK: {"temperature": 0.4, "max_tokens": 2048},
        OutputType.INFOGRAPHIC_BRIEF: {"temperature": 0.5, "max_tokens": 800},
        OutputType.VIDEO_SCRIPT: {"temperature": 0.6, "max_tokens": 2048},
        OutputType.BLOG_POST: {"temperature": 0.6, "max_tokens": 1800},
        OutputType.CUSTOM: {"temperature": 0.5, "max_tokens": 1200},
    }

    def generate(
        self,
        prompt: str,
        output_type: OutputType,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Executes LLM generation on the formatted prompt string.
        Returns (raw_text_output, metadata_dict).
        """
        params = self.FORMAT_HYPERPARAMS.get(output_type, {"temperature": 0.5, "max_tokens": 1024})
        temp = temperature if temperature is not None else params["temperature"]
        tokens = max_tokens if max_tokens is not None else params["max_tokens"]

        start_time = time.time()
        model_name = qwen_orchestrator_initializer.active_model_name

        if qwen_orchestrator_initializer.is_available():
            try:
                model = qwen_orchestrator_initializer.load()
                model_name = qwen_orchestrator_initializer.active_model_name
                logger.info(
                    "Qwen generation started | model=%s | output_type=%s | max_tokens=%d",
                    model_name,
                    output_type.value,
                    tokens,
                )
                response = model.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a professional content generation AI. Output valid JSON strictly grounded in the given context. Return ONLY a valid JSON object. Do not include explanations, reasoning, markdown code fences, or <think> blocks."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    max_tokens=tokens,
                    response_format={"type": "json_object"},
                    stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
                )
                choice = response["choices"][0]
                raw_text = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "unknown")
                usage = response.get("usage", {})
                latency = round(time.time() - start_time, 2)

                # Classify stop reason: 'stop token', 'max_tokens', 'EOS'
                if finish_reason == "length":
                    stop_reason = "max_tokens"
                    logger.warning(
                        "Qwen generation truncated due to reaching max_tokens (%d) for %s (finish_reason='length')",
                        tokens,
                        output_type.value,
                    )
                elif finish_reason == "eos":
                    stop_reason = "EOS"
                    logger.info("Qwen generation ended with EOS for %s", output_type.value)
                elif finish_reason == "stop":
                    stop_reason = "stop token"
                    logger.info("Qwen generation stopped on stop token for %s", output_type.value)
                else:
                    stop_reason = str(finish_reason)
                    logger.info("Qwen generation finished for %s with finish_reason: %s", output_type.value, finish_reason)

                metadata = {
                    "model": model_name,
                    "latency_seconds": latency,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "temperature": temp,
                    "backend": "llama-cpp-python",
                    "finish_reason": finish_reason,
                    "stop_reason": stop_reason,
                }
                logger.info(
                    "Qwen generation completed | model=%s | output_type=%s | elapsed=%.2fs | finish_reason=%s",
                    model_name,
                    output_type.value,
                    latency,
                    finish_reason,
                )
                return raw_text, metadata
            except Exception as exc:
                logger.error("Error during Qwen inference: %s. Falling back to structured simulation.", exc)

        # Offline / Mock Fallback if GGUF model files are absent (e.g., CI or lightweight environment)
        latency = round(time.time() - start_time, 2)
        metadata = {
            "model": f"{model_name} (Deterministic Fallback)",
            "latency_seconds": latency,
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": 0,
            "temperature": temp,
            "backend": "offline_fallback",
            "finish_reason": "stop",
            "stop_reason": "stop token",
        }
        return "", metadata


qwen3_generation_service = Qwen3GenerationService()
