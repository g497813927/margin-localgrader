"""Local HTTP boundary. No documents or grades are persisted by the server."""
import os
import logging
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

# No access, request, extraction, inference, or exception logging.
logging.disable(logging.CRITICAL)

# Set before importing any model implementation; downloads belong to setup only.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .privacy import install_network_guard
install_network_guard()

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool

from .defaults import DEFAULT_GRADING_GUIDANCE, DEFAULT_RUBRIC
from .extract import extract_document
from . import providers
from .schemas import CompareRequest, ExplanationRequest, ExplanationSettings, GradeRequest, ProbeRequest, Settings, VisionRequest, VisionSettings
from . import explanations
from . import vision
from .visual_inputs import VisualInputError, prepare_visual
from .visual_detection import detect_visuals as find_visuals
from .scoring import score_response
from .harness import HARNESS_VERSION, predict_rubric
from .upload_boundary import BodySizeLimitMiddleware
from .workspace_api import make_router, runtime_manager


@asynccontextmanager
async def lifespan(_app):
    # Server startup is explicit in the workspace. Shutdown only cleans up
    # processes that this application's runtime manager actually started.
    try:
        yield
    finally:
        await run_in_threadpool(runtime_manager.shutdown)


app = FastAPI(title="Margin · Local rubric grader", docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
app.add_middleware(BodySizeLimitMiddleware)
STATIC = Path(__file__).parent / "static"
MODEL_LOCK = threading.Lock()
MAX_UPLOAD = 15 * 1024 * 1024
app.include_router(make_router(MODEL_LOCK))


def require_stable_decision_runtimes(settings):
    """Inspect cached ownership state while holding MODEL_LOCK; no health calls."""
    bundled_ports = {1235: "clef", 1236: "startlux", 1237: "startlux", 1238: "d3"}
    for model in settings:
        if model.provider != "systemone":
            continue
        endpoint = urlsplit(model.endpoint)
        runtime_id = bundled_ports.get(endpoint.port) if endpoint.hostname == "127.0.0.1" else None
        if runtime_id and runtime_manager.is_transitioning(runtime_id):
            raise HTTPException(409, "A selected local server is starting or stopping. Wait until it is ready, then try again.")


def require_stable_explanation_runtime():
    # explanations.explain invokes this only after acquiring its inference lock,
    # which the bundled Ollama start/stop routes also acquire.
    if runtime_manager.is_transitioning("ollama"):
        raise explanations.Unavailable("The local explanation server is starting or stopping. Wait until it is ready, then try again.")


@app.middleware("http")
async def local_boundary(request: Request, call_next):
    origin = request.headers.get("origin")
    expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
    if origin and origin != expected:
        return JSONResponse({"detail": "Cross-origin requests are disabled."}, status_code=403)
    if request.headers.get("sec-fetch-site") == "cross-site":
        return JSONResponse({"detail": "Cross-site requests are disabled."}, status_code=403)
    # A page already open during an update may still use the old JavaScript,
    # which coerces null/withheld scores to zero. Never send it the new contract.
    if (request.method == "POST" and request.url.path in {"/api/compare", "/api/grade"}
            and (origin or request.headers.get("sec-fetch-site"))
            and request.headers.get("x-margin-scoring-version") != "3"):
        return JSONResponse({"detail": (
            "This page uses an older scoring display that can show unassigned scores as zero. "
            "Keep any work you need, then open Margin in a new tab to use the updated display. "
            "Refreshing clears unsaved work. No new grade was assigned."
        )}, status_code=409)
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError:
        return JSONResponse({"detail": "Invalid request size."}, status_code=400)
    if length > MAX_UPLOAD + 1024 * 1024:
        return JSONResponse({"detail": "The upload is too large (15 MiB maximum)."}, status_code=413)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Pydantic's raw errors contain input values; do not echo submissions into errors.
    errors = [f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()]
    return JSONResponse({"detail": "; ".join(errors)}, status_code=422)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "mode": "local-only", "document_storage": "none", "harness": HARNESS_VERSION}


@app.get("/api/defaults")
def defaults():
    return {"rubric": DEFAULT_RUBRIC, "grading_guidance": DEFAULT_GRADING_GUIDANCE,
            "explanation_settings": ExplanationSettings().model_dump(),
            "explanation_endpoint": explanations.OLLAMA_ENDPOINT,
            "vision_settings": VisionSettings().model_dump(),
            "settings": Settings().model_dump(),
            "comparison_models": [
                Settings().model_dump(),
                Settings(provider="systemone", model="clef-flash", endpoint="http://127.0.0.1:1235/v1/systemone").model_dump(),
                Settings(provider="systemone", model="StartLux-Decision-v1-4B-Q8_0", endpoint="http://127.0.0.1:1236/v1/systemone").model_dump(),
                Settings(provider="systemone", model="d3-mini", endpoint="http://127.0.0.1:1238/v1/systemone").model_dump(),
            ]}


@app.post("/api/extract")
async def extract(file: UploadFile = File(...), detect_visuals: bool = Form(default=False)):
    try:
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "The upload is too large (15 MiB maximum).")
        result = await run_in_threadpool(extract_document, file.filename or "document", data)
        if detect_visuals:
            detected = await run_in_threadpool(find_visuals, file.filename or "document", data)
            result.update(detected)
            # Text remains literal. The optional visual review is a separate
            # draft, never an automatic addition to the student's answer.
            if detected["visuals"]:
                result["warnings"] = [warning for warning in result["warnings"]
                                      if not warning.startswith(("Only extractable PDF text", "Only Word document text", "No extractable text was found"))]
                result["warnings"].append("Visual content was detected and opened for local review. Analyze and review the relevant images or pages before adding them to the submission.")
            result["warnings"].extend(detected["visual_warnings"])
        return result
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    finally:
        await file.close()


@app.post("/api/probe")
def probe(request: ProbeRequest):
    if not MODEL_LOCK.acquire(blocking=False):
        raise HTTPException(409, "The local model is busy. Wait for the current request.")
    try:
        require_stable_decision_runtimes([request.settings])
        return providers.probe(request.settings.model_dump())
    except (ValueError, RuntimeError) as exc:
        return {"ok": False, "message": str(exc)}
    finally:
        MODEL_LOCK.release()


@app.post("/api/vision/prepare")
async def prepare_image(file: UploadFile = File(...), index: int = Form(default=1, ge=1, le=300),
                        image_index: int | None = Form(default=None, ge=0, le=300)):
    try:
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "The upload is too large (15 MiB maximum).")
        return await run_in_threadpool(prepare_visual, file.filename or "image", data, index,
                                      image_index=image_index)
    except VisualInputError as exc:
        raise HTTPException(422, str(exc)) from None
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422, "The visual input could not be prepared. Use a valid image or export the page as PNG.") from None
    finally:
        await file.close()


@app.post("/api/vision/analyze")
def analyze_image(request: VisionRequest):
    # Shares the explanation lock (including Ollama start/stop). Neither this
    # route nor preparation receives a rubric or changes any grading result.
    try:
        return vision.analyze_image(request.image_data_url, request.model,
                                    runtime_guard=require_stable_explanation_runtime,
                                    detailed_crops=request.detailed_crops,
                                    transcribe=request.transcribe)
    except vision.VisionInputError as exc:
        raise HTTPException(422, str(exc)) from None
    except vision.VisionUnavailable as exc:
        raise HTTPException(503, str(exc)) from None


@app.post("/api/grade")
def grade(request: GradeRequest):
    if len(request.text) > request.settings.max_input_chars:
        raise HTTPException(422, "Submission exceeds the configured character limit. Shorten it or increase the limit; text is never silently truncated.")
    if not MODEL_LOCK.acquire(blocking=False):
        raise HTTPException(409, "The local model is busy. Wait for the current request.")
    try:
        require_stable_decision_runtimes([request.settings])
        prediction = predict_rubric(request)
        return score_response(request, prediction)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from None
    except Exception:
        raise HTTPException(503, "Local inference failed unexpectedly; no grade was assigned.") from None
    finally:
        MODEL_LOCK.release()


@app.post("/api/compare")
def compare(request: CompareRequest):
    if not MODEL_LOCK.acquire(blocking=False):
        raise HTTPException(409, "The local models are busy. Wait for the current assessment.")

    def run_one(index, settings):
        model_name = settings.display_name or ("Laya" if settings.provider == "laya" else settings.model)
        label = f"Model {chr(ord('A') + index)} · {model_name}"
        try:
            if len(request.text) > settings.max_input_chars:
                raise ValueError("Submission exceeds this model's character limit. No text was truncated.")
            grade_request = GradeRequest(rubric=request.rubric, settings=settings,
                                         text=request.text, submission_name=request.submission_name,
                                         assignment_details=request.assignment_details,
                                         grading_guidance=request.grading_guidance)
            result = score_response(grade_request, predict_rubric(grade_request))
            return {"label": label, "ok": True, "result": result}
        except (ValueError, RuntimeError) as exc:
            return {"label": label, "ok": False, "error": str(exc)}
        except Exception:
            # A third-party runtime must not echo the input through its traceback.
            return {"label": label, "ok": False, "error": "Local inference failed unexpectedly. Check the model installation; no grade was assigned."}

    try:
        require_stable_decision_runtimes(request.models)
        # At most four runtimes assess concurrently, including larger workspaces.
        # Each worker assesses criteria serially; results are never averaged.
        with ThreadPoolExecutor(max_workers=min(len(request.models), 4)) as pool:
            futures = [pool.submit(run_one, index, settings) for index, settings in enumerate(request.models)]
            results = [future.result() for future in futures]
        comparisons = []
        for criterion in request.rubric.criteria:
            rows = [next((row for row in entry["result"]["criteria"] if row["id"] == criterion.id), None)
                    if entry["ok"] else None for entry in results]
            points = [row["points"] if row else None for row in rows]
            level_ids = [row["level_id"] if row else None for row in rows]
            available = [row for row in rows if row is not None and row["points"] is not None]
            comparisons.append({"id": criterion.id, "name": criterion.name,
                                "model_points": points, "model_level_ids": level_ids,
                                "available_models": len(available),
                                "comparison_receipt": explanations.comparison_receipt(request, criterion.id, results),
                                "model_a_points": points[0], "model_b_points": points[1] if len(points) > 1 else None,
                                "disagrees": len({row["level_id"] for row in available}) > 1
                                if len(available) >= 2 else None})
        return {"results": results, "comparison": comparisons,
                "warnings": ["Comparing models can reveal disagreement; it does not eliminate bias or establish correctness. Review the evidence and choose the final grade yourself."]}
    finally:
        MODEL_LOCK.release()


@app.post("/api/explain")
def explain(request: ExplanationRequest):
    # Explanation inference has its own bounded lock and never blocks grading.
    return explanations.explain(request, runtime_guard=require_stable_explanation_runtime)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
