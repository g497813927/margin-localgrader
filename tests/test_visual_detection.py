"""Normal Word/PDF imports discover visuals without running a model."""
import io

from docx import Document
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from grader.app import app
from grader import vision
from grader.visual_detection import detect_visuals
from grader.workspace import parse_workspace, default_workspace
from tests.test_visual_inputs import docx_bytes, pdf_bytes


def mixed_pdf():
    output = io.BytesIO()
    writer = canvas.Canvas(output)
    writer.drawString(50, 600, "Text-only first page")
    writer.showPage()
    writer.rect(50, 500, 80, 100, fill=1)
    writer.drawString(50, 620, "Chart page")
    writer.showPage()
    writer.save()
    return output.getvalue()


def test_detection_finds_vector_charts_but_not_plain_text_page():
    result = detect_visuals("chart.pdf", mixed_pdf())
    assert result["visuals"] == [{"index": 2, "image_index": 0, "kind": "pdf_vectors", "label": "Page 2 — vector/page layout"}]
    assert result["visual_count"] == 2


def test_detection_word_images_matches_preview_indices():
    result = detect_visuals("mixed.docx", docx_bytes())
    assert [item["index"] for item in result["visuals"]] == [1, 2]
    assert result["visual_count"] == 2


def test_text_only_word_has_no_visuals():
    document = Document()
    document.add_paragraph("Synthetic answer with no images")
    output = io.BytesIO(); document.save(output)
    assert detect_visuals("answer.docx", output.getvalue())["visuals"] == []


def test_failed_detection_explicit_warning():
    result = detect_visuals("bad.pdf", b"SYNTHETIC_CORRUPT_DOCUMENT")
    assert result["visuals"] == [] and result["visual_warnings"]
    assert "SYNTHETIC_CORRUPT_DOCUMENT" not in str(result)


def test_normal_upload_detects_automatically_only_when_requested(monkeypatch):
    def no_inference(*args, **kwargs):
        raise AssertionError("Detection must not run a model.")
    monkeypatch.setattr(vision, "analyze_image", no_inference)
    client = TestClient(app)
    response = client.post("/api/extract", files={"file": ("chart.pdf", mixed_pdf())}, data={"detect_visuals": "true"})
    assert response.status_code == 200
    value = response.json()
    assert "Text-only first page" in value["text"]
    assert value["visuals"] == [{"index": 2, "image_index": 0, "kind": "pdf_vectors", "label": "Page 2 — vector/page layout"}]
    assert any("detected" in note for note in value["warnings"])
    plain = client.post("/api/extract", files={"file": ("chart.pdf", pdf_bytes())}, data={"detect_visuals": "false"})
    assert plain.status_code == 200 and "visuals" not in plain.json()


def test_older_workspace_defaults_detection_on_and_saves_only_settings():
    import json
    original = default_workspace().model_dump()
    original.pop("vision_settings")
    parsed = parse_workspace(json.dumps(original).encode())
    assert parsed.vision_settings.auto_detect is True
    parsed.vision_settings.auto_detect = False
    assert set(parsed.vision_settings.model_dump()) == {"auto_detect", "model", "detailed_crops"}
    assert parse_workspace(parsed.model_dump_json().encode()).vision_settings.auto_detect is False
