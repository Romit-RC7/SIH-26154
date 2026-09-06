"""Lazy model initializers used by staged recognition services."""

from backend.app.services.model_initializer.qwen_initializers import (
    QwenFusionInitializer,
    QwenModelInitializer,
    QwenVisionInitializer,
    qwen_fusion_initializer,
    qwen_vision_initializer,
)
from backend.app.services.model_initializer.pp_structure_initializer import (
    PPStructureInitializer,
    pp_structure_initializer,
)
from backend.app.services.model_initializer.unichart_initializer import (
    UniChartInitializer,
    unichart_initializer,
)
<<<<<<< HEAD
from backend.app.services.model_initializer.faster_whisper_initializer import (
    FasterWhisperInitializer,
    faster_whisper_initializer,
=======
from backend.app.services.model_initializer.bge_initializer import (
    BGEModelInitializer,
    bge_initializer,
>>>>>>> origin/romit/knowledge_retrival
)

__all__ = [
    "QwenFusionInitializer",
    "QwenModelInitializer",
    "QwenVisionInitializer",
    "qwen_fusion_initializer",
    "qwen_vision_initializer",
    "PPStructureInitializer",
    "pp_structure_initializer",
    "UniChartInitializer",
    "unichart_initializer",
<<<<<<< HEAD
    "FasterWhisperInitializer",
    "faster_whisper_initializer",
=======
    "BGEModelInitializer",
    "bge_initializer",
>>>>>>> origin/romit/knowledge_retrival
]
