"""One-shot PDF renderer: bounded pipes, no temporary documents or network.

Invoked only by visual_inputs through the current Python interpreter. Native
decoders run here so a crash or timeout cannot take down the web application.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys


MAX_INPUT_BYTES = 15 * 1024 * 1024
MAX_OUTPUT_BYTES = 11_000_128
MAX_IMAGE_PIXELS = 20_000_000
MAX_TOTAL_PIXELS = 40_000_000
MAX_STREAM_BYTES = 2 * 1024 * 1024
MAX_TOTAL_STREAM_BYTES = 8 * 1024 * 1024
MAX_OPERATIONS = 20_000
MAX_RESOURCES = 256
MAX_DEPTH = 16


# Keep the same error class when executed as a script or imported in tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grader.pdf_worker_errors import PdfPreparationError


def _unix_limits():
    if os.name == "nt":
        return
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 21))
    # macOS reserves a very large virtual address range for system libraries;
    # limiting address space there prevents Python from importing normally.
    memory_limit = 1024 * 1024 * 1024
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    else:
        # Darwin also rejects useful RLIMIT_DATA values on some releases. A
        # worker-local RSS watchdog complements the strict PDF resource budget.
        import threading
        import time
        def monitor_memory():
            while True:
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > memory_limit:
                    os._exit(1)
                time.sleep(0.1)
        threading.Thread(target=monitor_memory, daemon=True).start()


def preflight_pdf(data: bytes, index: int, reader=None) -> int:
    """Reject excessive image/stream resources before invoking native PDFium."""
    from pypdf import PdfReader, apply_configuration
    from pypdf.generic import ContentStream, DictionaryObject, StreamObject

    with apply_configuration(
        maximum_declared_stream_length=MAX_STREAM_BYTES,
        array_based_stream_maximum_output_length=MAX_STREAM_BYTES,
        zlib_maximum_output_length=MAX_STREAM_BYTES,
        lzw_maximum_output_length=MAX_STREAM_BYTES,
        run_length_maximum_output_length=MAX_STREAM_BYTES,
        brotli_maximum_output_length=MAX_STREAM_BYTES,
        stream_decoding_work_maximum_length=2 * MAX_STREAM_BYTES,
        page_tree_maximum_entries=3_000,
        jbig2dec_binary=None,
    ):
        reader = reader if reader is not None else PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise PdfPreparationError("encrypted")
        count = len(reader.pages)
        if not 1 <= count <= 300:
            raise PdfPreparationError("pages")
        if not 1 <= index <= count:
            raise PdfPreparationError("index")
        page = reader.pages[index - 1]
        seen = set()
        totals = {"pixels": 0, "streams": 0, "operations": 0, "resources": 0}

        def image_budget(value):
            width, height = value.get("/Width", value.get("/W")), value.get("/Height", value.get("/H"))
            if (not isinstance(width, int) or isinstance(width, bool)
                    or not isinstance(height, int) or isinstance(height, bool)
                    or width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS):
                raise PdfPreparationError("resources")
            totals["pixels"] += width * height
            if totals["pixels"] > MAX_TOTAL_PIXELS:
                raise PdfPreparationError("resources")

        def content_budget(stream):
            if stream is None:
                return
            content = ContentStream(stream, reader)
            totals["streams"] += len(content.get_data())
            if totals["streams"] > MAX_TOTAL_STREAM_BYTES:
                raise PdfPreparationError("resources")
            operations = content.operations
            totals["operations"] += len(operations)
            if totals["operations"] > MAX_OPERATIONS:
                raise PdfPreparationError("resources")
            for operands, operator in operations:
                if operator == b"INLINE IMAGE":
                    image_budget(operands["settings"])

        def resource_object(reference, depth):
            if depth > MAX_DEPTH:
                raise PdfPreparationError("resources")
            value = reference.get_object()
            if not isinstance(value, DictionaryObject):
                return
            if isinstance(value, StreamObject) and any(key in value for key in ("/F", "/FFilter", "/FDecodeParms")):
                # PDF external stream references are never opened, even if a
                # renderer or future parser would otherwise support them.
                raise PdfPreparationError("resources")
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            totals["resources"] += 1
            if totals["resources"] > MAX_RESOURCES:
                raise PdfPreparationError("resources")
            if value.get("/Subtype") == "/Image":
                image_budget(value)
                for key in ("/SMask", "/Mask"):
                    mask = value.get(key)
                    if mask is not None:
                        resource_object(mask, depth + 1)
                return
            if isinstance(value, StreamObject):
                content_budget(value)
            resources = value.get("/Resources")
            if resources is not None:
                walk_resources(resources, depth + 1)

        def walk_resources(reference, depth):
            if depth > MAX_DEPTH:
                raise PdfPreparationError("resources")
            resources = reference.get_object()
            if not isinstance(resources, DictionaryObject):
                raise PdfPreparationError("resources")
            for category in ("/XObject", "/Pattern"):
                mapping = resources.get(category)
                if mapping is not None:
                    mapping = mapping.get_object()
                    if not isinstance(mapping, DictionaryObject) or len(mapping) > MAX_RESOURCES:
                        raise PdfPreparationError("resources")
                    for reference in mapping.values():
                        resource_object(reference, depth + 1)
            states = resources.get("/ExtGState")
            if states is not None:
                states = states.get_object()
                if not isinstance(states, DictionaryObject) or len(states) > MAX_RESOURCES:
                    raise PdfPreparationError("resources")
                for reference in states.values():
                    state = reference.get_object()
                    mask = state.get("/SMask")
                    if mask is not None:
                        mask = mask.get_object()
                        if isinstance(mask, DictionaryObject) and mask.get("/G") is not None:
                            resource_object(mask["/G"], depth + 1)

        content_budget(page.get_contents())
        if page.get("/Resources") is not None:
            walk_resources(page["/Resources"], 0)
        return count


def render_pdf(data: bytes, index: int, image_index: int | None = None) -> dict:
    count = preflight_pdf(data, index)
    from grader.pdf_visuals import prepare_pdf_visual
    return prepare_pdf_visual(data, index, image_index, count)


def catalog_pdf(data: bytes) -> dict:
    from pypdf import PdfReader
    from grader.pdf_visuals import catalog_pdf_visuals
    reader = PdfReader(io.BytesIO(data), strict=False)
    count = preflight_pdf(data, 1, reader)
    for index in range(2, count + 1):
        preflight_pdf(data, index, reader)
    return catalog_pdf_visuals(data, count)


def main():
    # Keep only the result pipe. Native decoder diagnostics must never reach
    # either the browser or an inherited log, even on malformed inputs.
    result_fd = os.dup(sys.stdout.fileno())
    with open(os.devnull, "wb") as null:
        os.dup2(null.fileno(), sys.stdout.fileno())
        os.dup2(null.fileno(), sys.stderr.fileno())
    payload = {"error": "failed"}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from grader.privacy import install_network_guard
        install_network_guard()
        from grader.worker_limits import apply_worker_limits
        apply_worker_limits()
        _unix_limits()
        data = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if not data or len(data) > MAX_INPUT_BYTES:
            raise PdfPreparationError("resources")
        if sys.argv[1:] == ["catalog"]:
            payload = catalog_pdf(data)
        elif len(sys.argv) == 3:
            index = int(sys.argv[1])
            image_index = None if sys.argv[2] == "auto" else int(sys.argv[2])
            if not 1 <= index <= 300 or (image_index is not None and not 0 <= image_index <= 300):
                raise PdfPreparationError("index")
            payload = render_pdf(data, index, image_index)
        else:
            raise PdfPreparationError("index")
    except PdfPreparationError as exc:
        payload = {"error": exc.code}
    except BaseException:
        payload = {"error": "failed"}
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    if len(encoded) > MAX_OUTPUT_BYTES:
        encoded = b'{"error":"resources"}'
    with os.fdopen(result_fd, "wb") as result:
        result.write(encoded)


if __name__ == "__main__":
    main()
