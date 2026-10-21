"""Synthetic pixels and mocked local inference; no retained submission data."""
import base64
import io
import json

import pytest
from PIL import Image, ImageDraw

from grader import vision
from grader.vision_crops import image_views, combine_views
from grader.visual_inputs import normalize_image_data_url
from tests.test_vision import MODEL, mock_ollama, synthetic_image


def dense_image(size=(2400, 1600)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for y in range(15, size[1], 70):
        for x in range(15, size[0], 110):
            draw.rectangle((x, y, x + 55, y + 35), outline="black", width=2)
            draw.text((x + 4, y + 5), "key=?", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def decode(value):
    return Image.open(io.BytesIO(base64.b64decode(value.split(",")[1])))


def test_source_pixels_survive_until_cropping_and_every_region_is_covered():
    source = normalize_image_data_url(dense_image())
    assert (source["width"], source["height"]) == (2400, 1600)
    views = image_views(source, True)
    assert 1 < len(views) <= 10
    assert max(decode(views[0]["image_data_url"]).size) == 1600
    cover = Image.new("L", (2400, 1600), 0)
    draw = ImageDraw.Draw(cover)
    for view in views[1:]:
        left, top, right, bottom = view["bounds"]
        draw.rectangle((left, top, right - 1, bottom - 1), fill=255)
        with decode(view["image_data_url"]) as detail, decode(source["image_data_url"]) as original:
            assert detail.size == (right - left, bottom - top)
            assert detail.tobytes() == original.crop(view["bounds"]).tobytes()
    assert cover.getextrema() == (255, 255)
    first, second = views[1]["bounds"], views[2]["bounds"]
    assert first[2] > second[0]  # Genuine overlap, not disjoint tiles.


def test_detail_opt_out_and_small_images_do_not_add_calls():
    assert len(image_views(normalize_image_data_url(dense_image()), False)) == 1
    assert len(image_views(normalize_image_data_url(synthetic_image()), True)) == 1


def test_combination_preserves_conflicting_readings_with_provenance():
    views = [{"label": "Whole image", "bounds": [0, 0, 1000, 700],
              "transcription": "league 200", "description": "A table.", "uncertainties": []},
             {"label": "Detail row 1, column 1", "bounds": [0, 0, 560, 400],
              "transcription": "league 1", "description": "A table crop.", "uncertainties": []}]
    result = combine_views(views, MODEL, [])
    assert "league 200" in result["transcription"] and "league 1" in result["transcription"]
    assert result["views"] == views
    assert any("disagree" in note for note in result["uncertainties"])


def test_all_crops_use_verified_local_model_and_one_inference_lock(monkeypatch):
    seen, _ = mock_ollama(monkeypatch)
    result = vision.analyze_image(dense_image((1000, 700)), MODEL, detailed_crops=True)
    chats = [body for path, body in seen if path == "/api/chat"]
    assert result["view_count"] == len(chats) == 5
    assert len([path for path, _ in seen if path == "/api/status"]) == 1
    assert all(chat["keep_alive"] == 0 and chat["truncate"] is False for chat in chats)
    assert "overlapping detail crop" in chats[1]["messages"][1]["content"]
    assert not vision.explanations._LOCK.locked()


def test_vector_page_ocr_is_discarded_even_if_model_ignores_instruction(monkeypatch):
    seen, _ = mock_ollama(monkeypatch)
    result = vision.analyze_image(dense_image((1000, 700)), MODEL,
                                  detailed_crops=True, transcribe=False)
    assert result["transcription"] == ""
    assert all(view["transcription"] == "" for view in result["views"])
    assert all("Native PDF text was extracted separately" in body["messages"][1]["content"]
               for path, body in seen if path == "/api/chat")


@pytest.mark.parametrize("option", ["true", 1, None])
def test_invalid_detail_option_rejected_before_model_access(monkeypatch, option):
    monkeypatch.setattr(vision.httpx, "Client", lambda **kw: pytest.fail("No model call expected"))
    with pytest.raises(vision.VisionInputError):
        vision.analyze_image(synthetic_image(), MODEL, detailed_crops=option)


def mock_view_sequence(monkeypatch, steps):
    """Change verified mock chat envelopes without making any real requests."""
    seen, _ = mock_ollama(monkeypatch)
    request = vision.explanations._json_request
    calls = 0

    def reply(client, method, path, **kwargs):
        nonlocal calls
        output = request(client, method, path, **kwargs)
        if path == "/api/chat":
            step = steps[calls]
            calls += 1
            if isinstance(step, Exception):
                raise step
            output.update(step)
        return output

    monkeypatch.setattr(vision.explanations, "_json_request", reply)
    return seen


@pytest.mark.parametrize("failure", [
    {"done_reason": "length", "eval_count": 2048,
     "message": {"role": "assistant", "content": "REJECTED_PARTIAL_OUTPUT"}},
    {"eval_count": 2048,
     "message": {"role": "assistant", "content": "REJECTED_PARTIAL_OUTPUT"}},
    {"message": {"role": "assistant", "content": "REJECTED_PARTIAL_OUTPUT {"}},
    {"message": {"role": "assistant", "content": json.dumps({
        "transcription": "REJECTED_PARTIAL_OUTPUT", "description": "invalid", "points": 9})}},
])
def test_incomplete_view_keeps_successful_views_but_discards_its_content(monkeypatch, failure):
    seen = mock_view_sequence(monkeypatch, [{}, failure, {}, {}, {}])
    result = vision.analyze_image(dense_image((1000, 700)), MODEL, detailed_crops=True)
    assert len([path for path, _ in seen if path == "/api/chat"]) == 5
    assert result["view_count"] == 5 and result["incomplete_views"] == 1
    assert [view["status"] for view in result["views"]] == [
        "complete", "incomplete", "complete", "complete", "complete"]
    failed = result["views"][1]
    assert failed["label"] == "Detail row 1, column 1"
    assert failed["bounds"] == [0, 0, 560, 392]
    assert failed["transcription"] == failed["description"] == ""
    assert "Recognition failure is not missing student work" in failed["error"]
    assert failed["uncertainties"] == [failed["error"]]
    assert "REJECTED_PARTIAL_OUTPUT" not in json.dumps(result)
    assert all(view["transcription"] == "  if (x > 0) {\n    return x;\n  }\n"
               for view in result["views"] if view["status"] == "complete")
    assert "Whole image" in result["transcription"] and "Detail row 2, column 2" in result["transcription"]
    assert any("1 of 5 views" in note for note in result["uncertainties"])
    assert not vision.explanations._LOCK.locked()


def test_all_incomplete_views_remain_an_error(monkeypatch):
    seen = mock_view_sequence(monkeypatch, [{"done_reason": "length"}] * 5)
    with pytest.raises(vision.VisionUnavailable, match="complete result for any view"):
        vision.analyze_image(dense_image((1000, 700)), MODEL, detailed_crops=True)
    assert len([path for path, _ in seen if path == "/api/chat"]) == 5
    assert not vision.explanations._LOCK.locked()


@pytest.mark.parametrize("fatal", [
    {"model": "different-model", "done_reason": "length"},
    {"done": False}, {"prompt_eval_count": None}, {"truncated": True},
    {"message": {"role": "user", "content": "{}"}},
    {"message": {"role": "assistant", "content": "{}", "tool_calls": [{"name": "fetch"}]}},
    vision.explanations.Unavailable("The local explanation service is unavailable or timed out."),
])
def test_identity_protocol_and_transport_failures_after_success_fail_closed(monkeypatch, fatal):
    seen = mock_view_sequence(monkeypatch, [{}, fatal, {}, {}, {}])
    with pytest.raises(vision.VisionUnavailable):
        vision.analyze_image(dense_image((1000, 700)), MODEL, detailed_crops=True)
    assert len([path for path, _ in seen if path == "/api/chat"]) == 2
    assert not vision.explanations._LOCK.locked()


def test_incomplete_crop_never_restores_native_page_ocr(monkeypatch):
    mock_view_sequence(monkeypatch, [{}, {"done_reason": "length"}, {}, {}, {}])
    result = vision.analyze_image(dense_image((1000, 700)), MODEL,
                                  detailed_crops=True, transcribe=False)
    assert result["incomplete_views"] == 1
    assert result["transcription"] == ""
    assert all(view["transcription"] == "" for view in result["views"])
    assert not vision.explanations._LOCK.locked()


def test_single_view_with_details_enabled_still_rejects_incomplete_output(monkeypatch):
    mock_view_sequence(monkeypatch, [{"done_reason": "length"}])
    with pytest.raises(vision.VisionUnavailable, match="complete result"):
        vision.analyze_image(synthetic_image(), MODEL, detailed_crops=True)
    assert not vision.explanations._LOCK.locked()
