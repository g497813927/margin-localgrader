"""Unknown decoder failures must not expose input-derived exception messages."""
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from grader import app as application
from grader import visual_inputs as visual


PRIVATE_MARKER = "SYNTHETIC_PRIVATE_DECODER_DETAIL"


def fail_with_unknown_value_error(*args, **kwargs):
    raise ValueError(PRIVATE_MARKER)


@pytest.mark.parametrize("kind", ["image", "word"])
def test_unknown_decoder_value_errors_are_sanitized(monkeypatch, kind):
    if kind == "image":
        monkeypatch.setattr(visual.Image, "open", fail_with_unknown_value_error)
        filename, data = "test.png", b"synthetic image bytes"
    else:
        monkeypatch.setattr(visual, "_read_xml", fail_with_unknown_value_error)
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", "<document/>")
        filename, data = "test.docx", archive_bytes.getvalue()
    with pytest.raises(ValueError) as error:
        visual.prepare_visual(filename, data)
    assert PRIVATE_MARKER not in str(error.value)


def test_unknown_prepare_value_error_is_not_an_api_message(monkeypatch):
    monkeypatch.setattr(application, "prepare_visual", fail_with_unknown_value_error)
    response = TestClient(application.app).post(
        "/api/vision/prepare", files={"file": ("test.png", b"synthetic image bytes")}
    )
    assert response.status_code == 422
    assert PRIVATE_MARKER not in response.text
    assert "could not be prepared" in response.json()["detail"]
