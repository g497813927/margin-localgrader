"""Bounded, local text extraction. Images are never decoded or returned.

This module intentionally does not perform OCR, fetch links, execute macros, or
load remote resources. PDF text follows the reading order available in the PDF;
Word paragraphs and tables follow document order.
"""

from __future__ import annotations

import codecs
import io
import posixpath
import zipfile
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET

from pypdf import PdfReader, apply_configuration

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_TEXT_CHARS = 200_000
MAX_PDF_PAGES = 300
MAX_PDF_STREAM_BYTES = 16 * 1024 * 1024
MAX_ZIP_ENTRIES = 2_000
MAX_ZIP_EXPANDED_BYTES = 64 * 1024 * 1024
MAX_XML_BYTES = 8 * 1024 * 1024

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PACKAGE_R = "{http://schemas.openxmlformats.org/package/2006/relationships}"
IGNORED_WORD_TAGS = {W + name for name in ("drawing", "pict", "object", "txbxContent", "del", "pPr", "rPr")}


def _clean_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


def _bounded_join(parts: list[str], separator: str = "\n\n") -> str:
    if sum(map(len, parts)) + max(0, len(parts) - 1) * len(separator) > MAX_TEXT_CHARS:
        raise ValueError(f"Extracted text exceeds the {MAX_TEXT_CHARS:,}-character limit. Split the document into smaller files.")
    return separator.join(parts)


def extract_document(filename: str, data: bytes) -> dict:
    """Return filename, text, warnings, and (for PDFs) pages.

    Unsupported, corrupt, encrypted, or oversized files raise ``ValueError``.
    A readable file with no text returns an empty string and a warning, allowing
    the caller to explain why an image-only submission cannot be graded.
    """
    safe_name = PurePosixPath(str(filename).replace("\\", "/")).name
    extension = PurePosixPath(safe_name).suffix.lower()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("File exceeds the 15 MiB upload limit. Split it into smaller files.")
    if not data:
        raise ValueError("The uploaded file is empty.")
    if extension == ".doc":
        raise ValueError("Legacy .doc files are not supported. Open the file in Word or LibreOffice and save it as .docx or .txt.")
    if extension not in {".txt", ".pdf", ".docx"}:
        raise ValueError("Use a .txt, .pdf, or .docx file. Images and other file types are not supported.")

    if extension == ".txt":
        result = _extract_txt(data)
    elif extension == ".pdf":
        # Context-local limits also bound compressed text streams before the
        # extracted-character limit can be checked. No image decoder is used.
        with apply_configuration(
            maximum_declared_stream_length=MAX_PDF_STREAM_BYTES,
            array_based_stream_maximum_output_length=MAX_PDF_STREAM_BYTES,
            zlib_maximum_output_length=MAX_PDF_STREAM_BYTES,
            lzw_maximum_output_length=MAX_PDF_STREAM_BYTES,
            run_length_maximum_output_length=MAX_PDF_STREAM_BYTES,
            brotli_maximum_output_length=MAX_PDF_STREAM_BYTES,
            stream_decoding_work_maximum_length=2 * MAX_PDF_STREAM_BYTES,
            page_tree_maximum_entries=3_000,
            xform_maximum_invocations_per_extraction=500,
            jbig2dec_binary=None,
        ):
            result = _extract_pdf(data)
    else:
        result = _extract_docx(data)

    result["text"] = _clean_text(result["text"])
    if len(result["text"]) > MAX_TEXT_CHARS:
        raise ValueError(f"Extracted text exceeds the {MAX_TEXT_CHARS:,}-character limit. Split the document into smaller files.")
    if not result["text"]:
        result["warnings"].append("No extractable text was found. Image-only or scanned content is ignored; paste the text or provide a text-based document.")
    result["filename"] = safe_name
    return result


def _extract_txt(data: bytes) -> dict:
    if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        raise ValueError("UTF-32 text is not supported. Save the file as UTF-8 or UTF-16 with a byte-order mark.")
    encoding = "utf-16" if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig"
    try:
        text = data.decode(encoding, errors="strict")
    except UnicodeError as exc:
        raise ValueError("The text file could not be decoded. Save it as UTF-8 or UTF-16 with a byte-order mark.") from exc
    if "\x00" in text or any(ord(c) < 32 and c not in "\t\n\r\f" for c in text):
        raise ValueError("The file contains binary or unsupported control characters. Save a plain-text copy as UTF-8.")
    return {"text": text, "warnings": []}


def _extract_pdf(data: bytes) -> dict:
    warnings = ["Only extractable PDF text is included. Images, scanned text, annotations, and page layout are ignored."]
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except Exception as exc:
        raise ValueError("The PDF could not be read. It may be damaged or not a valid PDF; export a new text-based PDF.") from exc
    if reader.is_encrypted:
        raise ValueError("The PDF is encrypted or password-protected. Save an unencrypted copy before uploading.")
    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise ValueError("The PDF page structure could not be read. Export a new text-based PDF.") from exc
    if page_count > MAX_PDF_PAGES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit. Split it into smaller files.")

    parts: list[str] = []
    empty_pages: list[int] = []
    failed_pages: list[int] = []
    length = 0
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = _clean_text(page.extract_text() or "")
        except Exception:
            failed_pages.append(number)
            continue
        if not text:
            empty_pages.append(number)
            continue
        length += len(text) + (2 if parts else 0)
        if length > MAX_TEXT_CHARS:
            raise ValueError(f"Extracted text exceeds the {MAX_TEXT_CHARS:,}-character limit. Split the document into smaller files.")
        parts.append(text)
    if empty_pages:
        warnings.append(f"No extractable text on PDF page(s): {_page_list(empty_pages)}. These pages were skipped.")
    if failed_pages:
        raise ValueError(f"Text extraction failed on PDF page(s): {_page_list(failed_pages)}. The partial document was not accepted; export a new text-based PDF or paste its text.")
    return {"text": "\n\n".join(parts), "warnings": warnings, "pages": page_count}


def _page_list(pages: list[int]) -> str:
    return ", ".join(map(str, pages[:20])) + (f" (and {len(pages) - 20} more)" if len(pages) > 20 else "")


def _read_xml(archive: zipfile.ZipFile, name: str) -> ET.Element:
    info = archive.getinfo(name)
    if info.file_size > MAX_XML_BYTES:
        raise ValueError("A Word XML part is too large to safely read. Split the document into smaller files.")
    with archive.open(info) as source:
        data = source.read(MAX_XML_BYTES + 1)
    if len(data) > MAX_XML_BYTES:
        raise ValueError("A Word XML part exceeds the safe extraction limit.")
    declaration_check = data.replace(b"\x00", b"").upper()
    if b"<!DOCTYPE" in declaration_check or b"<!ENTITY" in declaration_check:
        raise ValueError("Word documents containing XML entity declarations are not supported.")
    return ET.fromstring(data)


def _paragraph_text(element: ET.Element) -> str:
    pieces: list[str] = []

    def visit(node: ET.Element) -> None:
        if node.tag in IGNORED_WORD_TAGS:
            return
        if node.tag == W + "t":
            pieces.append(node.text or "")
        elif node.tag == W + "tab":
            pieces.append("\t")
        elif node.tag in {W + "br", W + "cr"}:
            pieces.append("\n")
        else:
            for child in node:
                visit(child)

    visit(element)
    return "".join(pieces).strip()


def _word_blocks(element: ET.Element) -> list[str]:
    """Walk paragraphs/tables once, including content-control wrappers."""
    parts: list[str] = []
    used_chars = 0

    def append(text: str) -> None:
        nonlocal used_chars
        used_chars += len(text) + (1 if parts else 0)
        if used_chars > MAX_TEXT_CHARS:
            raise ValueError(f"Extracted text exceeds the {MAX_TEXT_CHARS:,}-character limit. Split the document into smaller files.")
        parts.append(text)

    for child in element:
        if child.tag in IGNORED_WORD_TAGS:
            continue
        if child.tag == W + "p":
            text = _paragraph_text(child)
            if text:
                append(text)
        elif child.tag == W + "tbl":
            for row in child.findall(W + "tr"):
                cells = ["\n".join(_word_blocks(cell)) for cell in row.findall(W + "tc")]
                if any(cells):
                    append(_bounded_join(cells, "\t"))
        elif child.tag not in {W + "sectPr", W + "tblPr", W + "tcPr", W + "trPr"}:
            for text in _word_blocks(child):
                append(text)
    return parts


def _extract_docx(data: bytes) -> dict:
    warnings = ["Only Word document text is included. Images, drawings, text boxes, comments, and embedded objects are ignored."]
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ZIP_ENTRIES or sum(item.file_size for item in entries) > MAX_ZIP_EXPANDED_BYTES:
                raise ValueError("The Word archive exceeds safe expansion limits. Remove large attachments or split the document.")
            names = [item.filename for item in entries]
            if len(set(names)) != len(names):
                raise ValueError("The Word archive contains duplicate parts and cannot be read safely.")
            if any(item.flag_bits & 1 for item in entries):
                raise ValueError("The Word archive is encrypted. Save an unencrypted .docx copy.")
            if "word/document.xml" not in names or "[Content_Types].xml" not in names:
                raise ValueError("This is not a supported .docx document. Save a new .docx copy from Word or LibreOffice.")
            document = _read_xml(archive, "word/document.xml")
            body = document.find(W + "body")
            if body is None:
                raise ValueError("The Word document has no readable document body.")
            parts = _word_blocks(body)

            # Use only referenced local header/footer parts. External relationships
            # are never opened, and no ZIP entry is extracted to the filesystem.
            relation_path = "word/_rels/document.xml.rels"
            relations = {}
            if relation_path in names:
                for relation in _read_xml(archive, relation_path).findall(PACKAGE_R + "Relationship"):
                    if relation.get("TargetMode") != "External":
                        relations[relation.get("Id")] = relation
            headers: list[str] = []
            footers: list[str] = []
            seen: set[str] = set()
            for reference in document.iter():
                if reference.tag not in {W + "headerReference", W + "footerReference"}:
                    continue
                relation = relations.get(reference.get(R + "id"))
                if relation is None:
                    continue
                kind = "header" if reference.tag == W + "headerReference" else "footer"
                if not (relation.get("Type") or "").endswith("/" + kind):
                    continue
                target = relation.get("Target", "")
                target_path = posixpath.normpath(target.lstrip("/") if target.startswith("/") else posixpath.join("word", target))
                if not target_path.startswith("word/") or target_path not in names or target_path in seen:
                    continue
                seen.add(target_path)
                texts = _word_blocks(_read_xml(archive, target_path))
                (headers if kind == "header" else footers).extend(texts)
            if headers or footers:
                warnings.append("Referenced headers and footers are included once; page layout and repeated page headers are not reproduced.")
            text = _bounded_join(headers + parts + footers, "\n")
    except ValueError:
        raise
    except (zipfile.BadZipFile, KeyError, ET.ParseError, RuntimeError, OSError, NotImplementedError, RecursionError) as exc:
        raise ValueError("The Word document could not be read. It may be damaged or encrypted; save a new .docx copy.") from exc
    return {"text": text, "warnings": warnings}
