"""PDF visual selection inside the isolated worker; no filesystem or network I/O.

Image regions are rendered from the visible PDF page, rather than decoding raw
XObjects. This preserves rotation, clipping, masks and text drawn over an image.
Native text is separate evidence and is never replaced by a model transcript.
"""
from contextlib import closing
import ctypes
import io
import math

import pypdfium2 as pdfium
from pypdf import PdfReader, apply_configuration

from .visual_inputs import _encode, MAX_EDGE, PDF_PAGE_NOTE
from .pdf_worker_errors import PdfPreparationError
from .visual_worker import MAX_OPERATIONS, MAX_DEPTH, MAX_IMAGE_PIXELS


def _dimensions(page):
    width, height = page.get_size()
    if not all(math.isfinite(n) and 1 <= n <= 100_000 for n in (width, height)):
        raise PdfPreparationError("dimensions")
    return width, height


def _visible_bounds(page, obj):
    """Map form-local PDF coordinates to the displayed page, including rotation."""
    bounds = obj.get_bounds()
    container = obj.container
    while container is not None:
        bounds = container.get_matrix().on_rect(*bounds)
        container = container.container
    if not all(math.isfinite(n) for n in bounds):
        raise PdfPreparationError("dimensions")
    width, height = _dimensions(page)
    # PageToDevice accounts for the page cropbox origin and /Rotate. A large,
    # bounded coordinate grid avoids rounding away small image borders.
    grid_w, grid_h = max(1, round(width * 100)), max(1, round(height * 100))
    points = []
    for x, y in ((bounds[0], bounds[1]), (bounds[0], bounds[3]),
                 (bounds[2], bounds[1]), (bounds[2], bounds[3])):
        dx, dy = ctypes.c_int(), ctypes.c_int()
        if not pdfium.raw.FPDF_PageToDevice(page, 0, 0, grid_w, grid_h, 0, x, y,
                                          ctypes.byref(dx), ctypes.byref(dy)):
            raise PdfPreparationError("dimensions")
        points.append((dx.value * width / grid_w, dy.value * height / grid_h))
    left = max(0.0, min(x for x, _ in points))
    top = max(0.0, min(y for _, y in points))
    right = min(width, max(x for x, _ in points))
    bottom = min(height, max(y for _, y in points))
    return (left, top, right, bottom) if right - left >= 0.5 and bottom - top >= 0.5 else None


def _inventory(page):
    images = []
    vectors = False
    for count, obj in enumerate(page.get_objects(max_depth=MAX_DEPTH), 1):
        if count > MAX_OPERATIONS:
            raise PdfPreparationError("resources")
        if obj.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE:
            bounds = _visible_bounds(page, obj)
            if bounds is None:
                continue
            pixels = obj.get_px_size()
            if pixels[0] * pixels[1] > MAX_IMAGE_PIXELS or min(pixels) < 1:
                raise PdfPreparationError("resources")
            images.append((bounds, pixels))
            if len(images) > 300:
                raise PdfPreparationError("resources")
        elif obj.type in (pdfium.raw.FPDF_PAGEOBJ_PATH, pdfium.raw.FPDF_PAGEOBJ_SHADING):
            vectors = True
    return images, vectors


def catalog_pdf_visuals(data, count):
    visuals = []
    with pdfium.PdfDocument(data) as document:
        if len(document) != count:
            raise PdfPreparationError("failed")
        for index in range(1, count + 1):
            with closing(document[index - 1]) as page:
                _dimensions(page)
                images, vectors = _inventory(page)
                visuals.extend({"index": index, "image_index": number, "kind": "pdf_image",
                                "label": f"Page {index} — embedded image {number}"}
                               for number in range(1, len(images) + 1))
                if vectors:
                    visuals.append({"index": index, "image_index": 0, "kind": "pdf_vectors",
                                    "label": f"Page {index} — vector/page layout"})
                if len(visuals) > 300:
                    raise PdfPreparationError("resources")
    notes = []
    if visuals:
        notes.append("Embedded images are listed separately from page layouts. Borders or decorative marks can also be detected; review relevant visuals. Native PDF text stays separate.")
    return {"visuals": visuals, "visual_count": count, "visual_warnings": notes}


def _native_text(data, index):
    # Match the normal import's literal native extraction; no model touches it.
    limit = 16 * 1024 * 1024
    with apply_configuration(maximum_declared_stream_length=limit,
                             array_based_stream_maximum_output_length=limit,
                             zlib_maximum_output_length=limit, lzw_maximum_output_length=limit,
                             run_length_maximum_output_length=limit, brotli_maximum_output_length=limit,
                             stream_decoding_work_maximum_length=2 * limit,
                             page_tree_maximum_entries=3_000, xform_maximum_invocations_per_extraction=500,
                             jbig2dec_binary=None):
        reader = PdfReader(io.BytesIO(data), strict=False)
        text = (reader.pages[index - 1].extract_text() or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()
    if len(text) > 200_000:
        raise PdfPreparationError("resources")
    return text


def prepare_pdf_visual(data, index, image_index, count):
    native = _native_text(data, index)
    with pdfium.PdfDocument(data) as document:
        if len(document) != count:
            raise PdfPreparationError("failed")
        with closing(document[index - 1]) as page:
            width, height = _dimensions(page)
            images, _ = _inventory(page)
            selected = (1 if images else 0) if image_index is None else image_index
            if selected > len(images):
                raise PdfPreparationError("image_index")
            if selected:
                (left, top, right, bottom), pixels = images[selected - 1]
                region_width, region_height = right - left, bottom - top
                # Preserve at least native pixel density without exceeding the
                # source cap. Page rendering respects masks, clipping and overlays.
                density = max(pixels) / max(region_width, region_height)
                scale = min(max(1.0, density), (MAX_EDGE - 2) / max(region_width, region_height))
                crop = (left, height - bottom, width - right, top)
                notes = ["Only the selected image's visible page region is included. PDF clipping, rotation and overlays are preserved; surrounding native text stays separate. Overlapping visuals may share pixels."]
            else:
                scale = min(4.0, (MAX_EDGE - 2) / max(width, height))
                crop = (0, 0, 0, 0)
                notes = [PDF_PAGE_NOTE]
            if max(width, height) * scale > 10_000_000:
                raise PdfPreparationError("dimensions")
            with closing(page.render(scale=scale, crop=crop, draw_annots=False, may_draw_forms=False)) as bitmap:
                result = _encode(bitmap.to_pil(), notes)
            return {"image_data_url": result["image_data_url"], "warnings": result["warnings"],
                    "count": count, "page": index, "image_index": selected, "image_count": len(images),
                    "kind": "pdf_image" if selected else "pdf_vectors", "native_text": native,
                    "transcription_source": "native_pdf" if not selected and native else "vision"}
