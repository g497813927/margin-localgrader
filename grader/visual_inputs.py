"""Portable, bounded visual inputs. No files are written or external links read.

PDF images are rendered as visible page crops; native text is returned separately. Word imports expose
only referenced embedded raster images; Word layout needs an exported PDF.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import subprocess
import sys
import threading
import warnings
import zipfile

from PIL import Image, ImageOps

from .extract import (MAX_PDF_PAGES, MAX_UPLOAD_BYTES, MAX_ZIP_ENTRIES,
                      MAX_ZIP_EXPANDED_BYTES, PACKAGE_R, R, _read_xml)

class VisualInputError(ValueError):
    """Only explicit, static visual-input errors may reach the browser."""


PDF_WORKER_TIMEOUT = 30
MAX_WORKER_OUTPUT_BYTES = 11_000_128
PDF_ERRORS = {
    "failed": "The PDF page could not be rendered safely. Use an unencrypted, valid PDF or export the page as PNG.",
    "encrypted": "The PDF is password-protected. Save an unencrypted copy before importing.",
    "pages": "Use a PDF with 1 to 300 pages, or split it into smaller documents.",
    "index": "The requested page is outside this PDF.",
    "dimensions": "This PDF page has unsupported dimensions. Export the page as PNG.",
    "image_index": "The requested embedded image is outside this PDF page.",
    "resources": "The PDF page exceeds safe image or drawing resource limits. Export a smaller page image or crop.",
}
PDF_PAGE_NOTE = ("Native PDF text is preserved separately. Page labels remain visible for diagram context; "
                 "use native text instead of model transcription. Annotations and interactive forms are not included.")


MAX_PIXELS = 20_000_000
MAX_EDGE = 4096
MAX_DATA_URL_CHARS = 11_000_000
_PDF_LOCK = threading.Lock()  # PDFium does not permit simultaneous calls.
_FORMATS = {"PNG", "JPEG", "WEBP"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
V = "{urn:schemas-microsoft-com:vml}"


def _encode(image: Image.Image, notes: list[str], max_edge: int = MAX_EDGE) -> dict:
    original = image.size
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    if original != image.size:
        notes.append(f"Source reduced to at most {max_edge:,} pixels per side. Dense diagrams can be analyzed in overlapping crops.")
    # A fresh RGB image drops EXIF, comments, profiles and other metadata. White
    # matches the page background for transparent diagrams and screenshots.
    rgba = image.convert("RGBA")
    clean = Image.new("RGB", image.size, "white")
    clean.paste(rgba, mask=rgba.getchannel("A"))
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG")
    encoded = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    if len(encoded) > MAX_DATA_URL_CHARS:
        raise VisualInputError("The prepared image is too large. Crop it into smaller images.")
    return {"image_data_url": encoded, "width": clean.width, "height": clean.height, "warnings": notes}


def _normalize(data: bytes) -> dict:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise VisualInputError("Choose a nonempty image no larger than 15 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=list(_FORMATS)) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise VisualInputError("Image exceeds 20 million pixels. Crop or resize it before importing.")
                if getattr(image, "n_frames", 1) != 1:
                    raise VisualInputError("Animated or multi-frame images are not supported. Export a single frame.")
                image.load()
                return _encode(ImageOps.exif_transpose(image), [])
    except VisualInputError:
        raise
    except Exception:
        raise VisualInputError("The image could not be decoded safely. Use a valid PNG, JPEG, or WebP image.") from None


def normalize_image_data_url(value: str) -> dict:
    if not isinstance(value, str) or len(value) > MAX_DATA_URL_CHARS:
        raise VisualInputError("The prepared image exceeds the local size limit.")
    prefix, separator, encoded = value.partition(",")
    if not separator or prefix not in {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"}:
        raise VisualInputError("Only an inline PNG, JPEG, or WebP image is accepted. Image URLs are never fetched.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise VisualInputError("The image encoding is invalid. Prepare the image again.") from None
    return _normalize(data)


def _pdf_worker(data: bytes, arguments: list[str]) -> dict:
    if not _PDF_LOCK.acquire(blocking=False):
        raise VisualInputError("Another PDF is being prepared. Wait and try again.")
    process = None
    try:
        root = Path(__file__).resolve().parents[1]
        environment = {key: os.environ[key] for key in
                       ("PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "LANG", "LC_ALL") if key in os.environ}
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(root / "grader/visual_worker.py"), *arguments],
            cwd=root, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        output, _ = process.communicate(input=data, timeout=PDF_WORKER_TIMEOUT)
        if process.returncode != 0 or len(output) > MAX_WORKER_OUTPUT_BYTES:
            raise VisualInputError(PDF_ERRORS["failed"])
        value = json.loads(output)
        if not isinstance(value, dict):
            raise VisualInputError(PDF_ERRORS["failed"])
        if set(value) == {"error"}:
            code = value["error"]
            raise VisualInputError(PDF_ERRORS.get(code, PDF_ERRORS["failed"]) if isinstance(code, str) else PDF_ERRORS["failed"])
        return value
    except subprocess.TimeoutExpired:
        raise VisualInputError("PDF preparation exceeded 30 seconds. Export the selected visual as PNG or use a simpler PDF.") from None
    except VisualInputError:
        raise
    except Exception:
        raise VisualInputError(PDF_ERRORS["failed"]) from None
    finally:
        try:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
                process.wait(timeout=5)
        except Exception:
            raise VisualInputError(PDF_ERRORS["failed"]) from None
        finally:
            _PDF_LOCK.release()


def pdf_visual_catalog(data: bytes) -> dict:
    value = _pdf_worker(data, ["catalog"])
    if (set(value) != {"visuals", "visual_count", "visual_warnings"}
            or type(value["visual_count"]) is not int or not 1 <= value["visual_count"] <= MAX_PDF_PAGES
            or not isinstance(value["visuals"], list) or len(value["visuals"]) > 300
            or not isinstance(value["visual_warnings"], list)
            or any(not isinstance(note, str) or len(note) > 500 for note in value["visual_warnings"])):
        raise VisualInputError(PDF_ERRORS["failed"])
    for item in value["visuals"]:
        if (not isinstance(item, dict) or set(item) != {"index", "image_index", "kind", "label"}
                or type(item["index"]) is not int or not 1 <= item["index"] <= value["visual_count"]
                or type(item["image_index"]) is not int or not 0 <= item["image_index"] <= 300
                or item["kind"] not in {"pdf_image", "pdf_vectors"}
                or not isinstance(item["label"], str) or len(item["label"]) > 120):
            raise VisualInputError(PDF_ERRORS["failed"])
    return value


def _pdf_page(data: bytes, index: int, image_index: int | None) -> tuple[dict, int]:
    value = _pdf_worker(data, [str(index), "auto" if image_index is None else str(image_index)])
    fields = {"image_data_url", "count", "page", "image_index", "image_count", "kind", "native_text", "transcription_source", "warnings"}
    if (set(value) != fields or type(value["count"]) is not int or not index <= value["count"] <= MAX_PDF_PAGES
            or type(value["page"]) is not int or value["page"] != index or type(value["image_count"]) is not int or not 0 <= value["image_count"] <= 300
            or type(value["image_index"]) is not int or not 0 <= value["image_index"] <= value["image_count"]
            or value["kind"] not in {"pdf_image", "pdf_vectors"}
            or value["transcription_source"] not in {"native_pdf", "vision"}
            or not isinstance(value["native_text"], str) or len(value["native_text"]) > 200_000
            or not isinstance(value["warnings"], list)
            or any(not isinstance(note, str) or len(note) > 500 for note in value["warnings"])):
        raise VisualInputError(PDF_ERRORS["failed"])
    result = normalize_image_data_url(value.pop("image_data_url"))
    result["warnings"].extend(value.pop("warnings"))
    count = value.pop("count")
    return {**result, **value}, count


def _read_visual_xml(archive, name):
    try:
        return _read_xml(archive, name)
    except Exception:
        raise VisualInputError("The Word XML could not be read safely. XML entity declarations and unsupported document parts are not accepted.") from None


def _docx_image(data: bytes, index: int) -> tuple[dict, int]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            names = {item.filename for item in entries}
            if (len(entries) > MAX_ZIP_ENTRIES or len(names) != len(entries)
                    or sum(item.file_size for item in entries) > MAX_ZIP_EXPANDED_BYTES
                    or any(item.flag_bits & 1 for item in entries)):
                raise VisualInputError("The Word document exceeds safe archive limits or contains ambiguous or encrypted parts.")
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise VisualInputError("Use a valid .docx document.")
            document = _read_visual_xml(archive, "word/document.xml")
            relation_file = "word/_rels/document.xml.rels"
            relations = {}
            if relation_file in names:
                relations = {node.get("Id"): node for node in _read_visual_xml(archive, relation_file).findall(PACKAGE_R + "Relationship")}
            targets = []
            skipped = False
            for node in document.iter():
                if node.tag not in {A + "blip", V + "imagedata"}:
                    continue
                relation = relations.get(node.get(R + "embed") or node.get(R + "id"))
                if relation is None or relation.get("TargetMode", "").lower() == "external":
                    skipped = True
                    continue
                target = relation.get("Target", "")
                path = posixpath.normpath(target.lstrip("/") if target.startswith("/") else posixpath.join("word", target))
                if (not relation.get("Type", "").endswith("/image") or not path.startswith("word/media/")
                        or path not in names or "\\" in target or PurePosixPath(path).suffix.lower() not in _IMAGE_SUFFIXES):
                    skipped = True
                    continue
                targets.append(path)
                if len(targets) > MAX_PDF_PAGES:
                    raise VisualInputError("The Word document contains more than 300 image references. Split it into smaller documents.")
            if not targets:
                raise VisualInputError("No supported embedded images were found in the Word document. Export the document as PDF to include charts, shapes, and layout.")
            if index > len(targets):
                raise VisualInputError("The requested image is outside this Word document.")
            item = archive.getinfo(targets[index - 1])
            if item.file_size > MAX_UPLOAD_BYTES:
                raise VisualInputError("This embedded image exceeds 15 MiB. Export or resize it separately.")
            result = _normalize(archive.read(item))
            result["warnings"].append("Only this embedded image from the document body is included. Word charts, shapes, headers, cropping, placement, and layout are not rendered. Export as PDF for those visuals.")
            if skipped:
                result["warnings"].append("Some external or unsupported image references were skipped. No external resource was fetched.")
            return result, len(targets)
    except VisualInputError:
        raise
    except Exception:
        raise VisualInputError("The Word images could not be read safely. Export the relevant page as PDF or PNG.") from None


def prepare_visual(filename: str, data: bytes, index: int = 1, image_index: int | None = None) -> dict:
    if type(index) is not int or not 1 <= index <= MAX_PDF_PAGES:
        raise VisualInputError("Choose an image or page number from 1 to 300.")
    if image_index is not None and (type(image_index) is not int or not 0 <= image_index <= 300):
        raise VisualInputError("Choose an embedded image number from 1 to 300, or 0 for the page layout.")
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise VisualInputError("Choose a nonempty file no larger than 15 MiB.")
    safe_name = PurePosixPath(str(filename).replace("\\", "/")).name[:250]
    extension = PurePosixPath(safe_name).suffix.lower()
    if extension == ".pdf":
        result, count = _pdf_page(data, index, image_index)
        suffix = f" — embedded image {result['image_index']}" if result["image_index"] else " — page layout"
        label = f"{safe_name} — page {index}{suffix}"
    elif extension == ".docx":
        result, count = _docx_image(data, index)
        label = f"{safe_name} — embedded image {index}"
    elif extension in _IMAGE_SUFFIXES:
        if index != 1:
            raise VisualInputError("A standalone image has only one page.")
        result, count, label = _normalize(data), 1, safe_name
    else:
        raise VisualInputError("Use PNG, JPEG, WebP, PDF, or Word (.docx) for visual input.")
    return {**result, "index": index, "count": count, "source_label": label}
