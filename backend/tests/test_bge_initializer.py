"""
Unit tests for BGEModelInitializer.
"""

import pytest
from backend.app.services.model_initializer.bge_initializer import BGEModelInitializer, bge_initializer


def test_bge_initializer_availability():
    init = BGEModelInitializer()
    # In test environment with staged model weights directory
    available = init.is_available()
    assert isinstance(available, bool)


def test_bge_initializer_encode_dimension_and_normalization():
    init = BGEModelInitializer()
    texts = [
        "Artificial Intelligence is transforming business operations.",
        "Cloud revenue increased by 35% year over year."
    ]
    embeddings = init.encode(texts, normalize_embeddings=True)

    assert len(embeddings) == 2
    for emb in embeddings:
        assert len(emb) == 384
        # Verify L2 normalization (norm ~ 1.0)
        norm = sum(x * x for x in emb) ** 0.5
        assert abs(norm - 1.0) < 1e-3


def test_bge_initializer_encode_query():
    init = BGEModelInitializer()
    query_vec = init.encode_query("What was the revenue growth in Q3?")
    assert len(query_vec) == 384
    norm = sum(x * x for x in query_vec) ** 0.5
    assert abs(norm - 1.0) < 1e-3


def test_bge_initializer_unload():
    init = BGEModelInitializer()
    init.encode(["Sample text for testing."])
    assert init.model is not None
    init.unload()
    assert init.model is None
