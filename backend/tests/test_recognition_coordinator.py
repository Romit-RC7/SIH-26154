from pathlib import Path

from PIL import Image

from backend.app.processors.base import RawDocumentElement
from backend.app.services.recognition.coordinator import RecognitionCoordinator


class FakeModel:
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def predict(self, input, batch_size):
        assert input is not None
        assert batch_size == 1
        yield self.payload

    def close(self):
        self.closed = True


def _write_model_package(root: Path, name: str, chart: bool = False):
    model_dir = root / name
    model_dir.mkdir()
    (model_dir / "inference.yml").write_text("mode: paddle", encoding="utf-8")
    weight_name = "model_state.pdparams" if chart else "inference.pdiparams"
    (model_dir / weight_name).write_bytes(b"weights")


def test_recognizes_formula_and_chart_sequentially(tmp_path, monkeypatch):
    _write_model_package(tmp_path, "formula")
    _write_model_package(tmp_path, "chart", chart=True)
    models = []

    def fake_create_model(model_name, model_dir):
        model = FakeModel({"model": model_name})
        models.append(model)
        return model

    monkeypatch.setattr(
        "backend.app.services.recognition.coordinator.RecognitionCoordinator._create_model",
        staticmethod(fake_create_model),
    )
    monkeypatch.setattr(
        "backend.app.services.recognition.chart_service.ChartRecognitionService._load_model",
        lambda self: fake_create_model(self.model_name, self.model_dir),
    )

    elements = [
        RawDocumentElement("text", 1, image=Image.new("RGB", (4, 4)), attributes={"recognition_type": "formula"}),
        RawDocumentElement("chart", 1, image=Image.new("RGB", (4, 4))),
    ]
    RecognitionCoordinator(tmp_path).recognize(elements)

    assert elements[0].attributes["formula_recognition_status"] == "completed"
    assert elements[1].attributes["chart_recognition_status"] == "completed"
    assert elements[0].attributes["formula_recognition"]["model"] == "PP-FormulaNet_plus-M"
    assert elements[1].attributes["chart_recognition"]["model"] == "PP-Chart2Table"
    assert all(model.closed is True for model in models)


def test_missing_model_is_reported_without_failure(tmp_path):
    element = RawDocumentElement("formula", 1, image=Image.new("RGB", (4, 4)))

    RecognitionCoordinator(tmp_path).recognize([element])

    assert element.attributes["formula_recognition_status"] == "unavailable"
    assert "Local model package not found" in element.attributes["formula_recognition_error"]
