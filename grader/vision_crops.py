"""Bounded overlapping views built from source pixels, before model resizing.

No OCR, files, or model calls occur here. An overview retains global connections;
independent detail views improve the size of small labels without inventing pixels.
"""
from __future__ import annotations

import base64
import io
import math

from PIL import Image, ImageStat

from .visual_inputs import _encode

MODEL_EDGE = 1600
MAX_DETAIL_VIEWS = 9


def image_views(prepared: dict, detailed: bool) -> list[dict]:
    data = base64.b64decode(prepared["image_data_url"].split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(data)) as source:
        source.load()
        width, height = source.size
        overview = {"label": "Whole image", "bounds": [0, 0, width, height],
                    **_encode(source.copy(), [], max_edge=MODEL_EDGE)}
        views = [overview]
        if not detailed or width * height < 250_000:
            return views
        columns = min(3, max(1, math.ceil(width / 700)))
        rows = min(3, max(1, math.ceil(height / 550)))
        if columns * rows == 1:
            return views
        for row in range(rows):
            for column in range(columns):
                # Extend each cell 12% on both sides: adjacent crops overlap,
                # helping avoid splitting table headers and diagram nodes.
                margin_x = width / columns * 0.12 if columns > 1 else 0
                margin_y = height / rows * 0.12 if rows > 1 else 0
                bounds = [max(0, math.floor(column * width / columns - margin_x)),
                          max(0, math.floor(row * height / rows - margin_y)),
                          min(width, math.ceil((column + 1) * width / columns + margin_x)),
                          min(height, math.ceil((row + 1) * height / rows + margin_y))]
                with source.crop(bounds) as crop:
                    # Uniform white margins (or uniform dark backgrounds) add
                    # no evidence. Do not use this as a semantic classifier.
                    if max(ImageStat.Stat(crop.resize((64, 64))).stddev) < 1:
                        continue
                    views.append({"label": f"Detail row {row + 1}, column {column + 1}",
                                  "bounds": bounds, **_encode(crop, [], max_edge=MODEL_EDGE)})
        assert len(views) <= MAX_DETAIL_VIEWS + 1
        return views


def combine_views(results: list[dict], model: str, warnings: list[str]) -> dict:
    """Keep provenance and differing readings; never ask a model to merge them."""
    incomplete = sum(result.get("status") == "incomplete" for result in results)
    if len(results) == 1:
        result = results[0]
        return {"model": model, "transcription": result["transcription"],
                "description": result["description"],
                "uncertainties": warnings + result["uncertainties"],
                "views": results, "view_count": 1, "incomplete_views": incomplete}
    notes = [*warnings,
             "Overlapping views can repeat text or disagree. Compare the labelled readings with the image and reconcile them before adding evidence. A crop cannot establish connections beyond its edges."]
    if incomplete:
        notes.insert(0, f"Analysis is incomplete: {incomplete} of {len(results)} views could not be read. Recognition failure is not missing student work; check the unread regions in the original image.")
    for result in results:
        notes.extend(f'{result["label"]}: {item}' for item in result["uncertainties"])
    def sections(field):
        return "\n\n".join(f'{result["label"]} — pixels {result["bounds"]}:\n{result[field]}'
                            for result in results if result[field].strip())
    return {"model": model, "transcription": sections("transcription"),
            "description": sections("description"), "uncertainties": notes,
            "views": results, "view_count": len(results), "incomplete_views": incomplete}
