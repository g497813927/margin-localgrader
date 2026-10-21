"""Find potential visuals during the normal document import, without inference."""
import io
from pathlib import PurePosixPath
import posixpath
import zipfile


from .extract import (MAX_PDF_PAGES, MAX_ZIP_ENTRIES,
                      MAX_ZIP_EXPANDED_BYTES, PACKAGE_R, R, _read_xml)

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
V = "{urn:schemas-microsoft-com:vml}"


def detect_visuals(filename: str, data: bytes) -> dict:
    extension = PurePosixPath(filename).suffix.lower()
    empty = {"visuals": [], "visual_count": 0, "visual_warnings": []}
    try:
        if extension == ".pdf":
            from .visual_inputs import pdf_visual_catalog
            return pdf_visual_catalog(data)
        if extension == ".docx":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = archive.infolist()
                if (len(entries) > MAX_ZIP_ENTRIES or sum(item.file_size for item in entries) > MAX_ZIP_EXPANDED_BYTES
                        or any(item.flag_bits & 1 for item in entries)):
                    raise ValueError
                names = {item.filename for item in entries}
                if len(names) != len(entries):
                    raise ValueError
                document = _read_xml(archive, "word/document.xml")
                relation_path = "word/_rels/document.xml.rels"
                relations = {node.get("Id"): node for node in _read_xml(archive, relation_path).findall(PACKAGE_R + "Relationship")} if relation_path in names else {}
                count, skipped = 0, False
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
                            or path not in names or "\\" in target or PurePosixPath(path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}):
                        skipped = True
                        continue
                    count += 1
                    if count > 300:
                        raise ValueError
                notes = []
                if skipped or any(name.startswith(("word/charts/", "word/diagrams/")) for name in names):
                    notes.append("Some Word charts, drawings, or image references need a PDF export to preserve their visual layout. External resources were not fetched.")
                return {"visuals": [{"index": i, "label": f"Embedded image {i}"} for i in range(1, count + 1)],
                        "visual_count": count, "visual_warnings": notes}
    except Exception:
        return {**empty, "visual_warnings": ["Automatic visual detection could not complete safely. Text extraction is separate; check the document and use an exported PDF page or image for the visuals."]}
    return empty
