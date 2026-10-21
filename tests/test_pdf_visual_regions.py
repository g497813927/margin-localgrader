"""Generated PDF crops verify native text, placement and source fidelity."""
import io

import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter

from grader.extract import extract_document
from grader.visual_detection import detect_visuals
from grader.visual_inputs import prepare_visual, VisualInputError
from tests.test_visual_inputs import decoded


def sample_pdf(*, rotated=False, nested=False, clipped=False):
    out = io.BytesIO()
    writer = canvas.Canvas(out, pagesize=(400, 300))
    writer.drawString(20, 280, "SELECT ?; league = 1; score >= 0")
    red = Image.new("RGB", (1200, 500), "red")
    if nested:
        writer.beginForm("nested", 0, 0, 200, 100)
        writer.drawImage(ImageReader(red), 0, 0, 200, 100)
        writer.endForm()
        writer.saveState(); writer.translate(30, 110); writer.scale(0.75, 0.75)
        writer.doForm("nested"); writer.restoreState()
    else:
        writer.saveState()
        if clipped:
            path = writer.beginPath(); path.rect(30, 110, 100, 100)
            writer.clipPath(path, stroke=0)
        writer.drawImage(ImageReader(red), 30, 110, 200, 100)
        # Native labels drawn over a screenshot must stay visible.
        writer.setFillColorRGB(0, 0, 0)
        writer.drawString(40, 150, "Overlay")
        writer.restoreState()
    writer.drawImage(ImageReader(Image.new("RGB", (300, 150), "blue")), 250, 30, 120, 60)
    writer.showPage(); writer.save()
    data = out.getvalue()
    if rotated:
        result = io.BytesIO(); pdf = PdfWriter(clone_from=PdfReader(io.BytesIO(data)))
        pdf.pages[0].rotate(90); pdf.write(result); data = result.getvalue()
    return data


def test_pdf_catalog_selects_individual_images_and_preserves_native_punctuation():
    data = sample_pdf()
    found = detect_visuals("test.pdf", data)
    images = [item for item in found["visuals"] if item["kind"] == "pdf_image"]
    assert [item["image_index"] for item in images] == [1, 2]
    first = prepare_visual("test.pdf", data)
    second = prepare_visual("test.pdf", data, image_index=2)
    assert first["kind"] == "pdf_image" and first["image_index"] == 1
    assert first["image_count"] == 2 and first["page"] == 1
    assert "SELECT ?; league = 1; score >= 0" in first["native_text"]
    assert first["native_text"] == extract_document("test.pdf", data)["text"]
    assert first["transcription_source"] == "vision"
    # Crops use the image region, not a shrunk full page with surrounding prose.
    first_image, second_image = decoded(first), decoded(second)
    assert first_image.getpixel((5, 5)) == (255, 0, 0)
    assert second_image.getpixel((5, 5)) == (0, 0, 255)
    assert first_image.width >= 1198
    assert first_image.width / first_image.height == pytest.approx(2, abs=0.02)
    assert not any(pixel == (0, 0, 255) for pixel in first_image.get_flattened_data())
    # The native overlaid text remains visible in the region.
    assert any(pixel == (0, 0, 0) for pixel in first_image.get_flattened_data())


def test_vector_page_uses_exact_native_text_instead_of_an_ocr_substitute():
    data = sample_pdf()
    result = prepare_visual("test.pdf", data, image_index=0)
    assert result["kind"] == "pdf_vectors" and result["image_index"] == 0
    assert result["transcription_source"] == "native_pdf"
    assert result["native_text"] == extract_document("test.pdf", data)["text"]
    assert "?" in result["native_text"]


@pytest.mark.parametrize("rotated,nested", [(True, False), (False, True), (True, True)])
def test_rotated_pages_and_nested_form_image_bounds_are_resolved(rotated, nested):
    result = prepare_visual("test.pdf", sample_pdf(rotated=rotated, nested=nested), image_index=1)
    image = decoded(result)
    assert image.getpixel((image.width // 2, image.height // 2)) == (255, 0, 0)
    if rotated:
        assert image.height > image.width
    else:
        assert image.width > image.height
    assert not any(pixel == (0, 0, 255) for pixel in image.get_flattened_data())


def test_clipped_image_does_not_reveal_hidden_pixels():
    image = decoded(prepare_visual("test.pdf", sample_pdf(clipped=True), image_index=1))
    assert image.getpixel((image.width // 4, 5)) == (255, 0, 0)
    assert image.getpixel((3 * image.width // 4, 5)) == (255, 255, 255)


def test_image_index_out_of_range_and_wrong_type_fail():
    with pytest.raises(VisualInputError, match="outside"):
        prepare_visual("test.pdf", sample_pdf(), image_index=3)
    with pytest.raises(VisualInputError, match="number"):
        prepare_visual("test.pdf", sample_pdf(), image_index=True)


def test_native_text_is_not_replaced_by_a_model_or_added_to_raster_pixels():
    data = sample_pdf()
    text = extract_document("test.pdf", data)["text"]
    for image_index in (1, 2, 0):
        assert prepare_visual("test.pdf", data, image_index=image_index)["native_text"] == text
    assert "league = 1" in text and "SELECT ?" in text


def test_vector_shapes_without_native_text_can_still_request_visual_transcription():
    output = io.BytesIO()
    writer = canvas.Canvas(output, pagesize=(200, 200))
    writer.rect(20, 20, 100, 100)
    writer.showPage(); writer.save()
    result = prepare_visual("shape.pdf", output.getvalue())
    assert result["native_text"] == ""
    assert result["transcription_source"] == "vision"
