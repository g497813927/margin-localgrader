"""Local workspace and bundled-process controls. No inference or downloads here."""
from contextlib import ExitStack

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from . import explanations
from .runtime_manager import RuntimeManager
from .workspace import MAX_WORKSPACE_BYTES, ROOT, WorkspaceStore, model_catalog, parse_workspace

workspace_store = WorkspaceStore()
runtime_manager = RuntimeManager()
RUNTIME_NAMES = {"clef": "Clef-Flash", "startlux": "StartLux Decision 4B",
                 "d3": "Decision 3.0 Mini", "ollama": "Local explanation and vision service"}


def runtime_view(value):
    return {**value, "status": value["state"], "name": RUNTIME_NAMES[value["id"]]}


async def read_workspace(request):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_WORKSPACE_BYTES:
            raise HTTPException(413, "Workspace settings are too large (64 KiB maximum).")
    try:
        return parse_workspace(bytes(data))
    except (ValueError, TypeError, RecursionError):
        # Do not include submitted values, unknown keys, or parser error text.
        raise HTTPException(422, "Invalid workspace. Use settings-only workspace JSON with up to eight distinct enabled local models.") from None


def make_router(model_lock):
    router = APIRouter()

    @router.get("/setup")
    def setup_guide():
        return FileResponse(ROOT / "SETUP.html")

    @router.get("/api/model-catalog")
    def catalog():
        return model_catalog()

    @router.get("/api/workspace")
    def load_workspace():
        try:
            workspace, saved = workspace_store.load()
        except (OSError, ValueError, TypeError, RecursionError):
            raise HTTPException(409, "Saved workspace could not be read safely. Defaults are available in memory; import or explicitly save valid settings to recover.") from None
        return {"workspace": workspace.model_dump(), "saved": saved}

    @router.post("/api/workspace/validate")
    async def validate_workspace(request: Request):
        workspace = await read_workspace(request)
        return {"workspace": workspace.model_dump()}

    @router.put("/api/workspace")
    async def save_workspace(request: Request):
        workspace = await read_workspace(request)
        try:
            checked = await run_in_threadpool(workspace_store.save, workspace)
        except (OSError, ValueError, TypeError):
            raise HTTPException(409, "Workspace settings could not be saved safely. Check the project's config folder permissions.") from None
        return {"workspace": checked.model_dump(), "saved": True}

    @router.get("/api/runtimes")
    def runtimes():
        return {"services": [runtime_view(value) for value in runtime_manager.statuses().values()]}

    def runtime_action(service_id, action):
        if service_id not in RUNTIME_NAMES:
            raise HTTPException(404, "Unknown bundled server.")
        with ExitStack() as release:
            locks = [model_lock] + ([explanations._LOCK] if service_id == "ollama" else [])
            for lock in locks:
                if not lock.acquire(blocking=False):
                    raise HTTPException(409, "A local assessment, explanation, or image analysis is busy. Wait before changing servers.")
                release.callback(lock.release)
            try:
                value = getattr(runtime_manager, action)(service_id)
            except (ValueError, RuntimeError, OSError):
                raise HTTPException(409, "The bundled server action could not be completed. Refresh server status and check the setup guide.") from None
            return {"service": runtime_view(value)}

    @router.post("/api/runtimes/{service_id}/start")
    def start_runtime(service_id: str):
        return runtime_action(service_id, "start")

    @router.post("/api/runtimes/{service_id}/stop")
    def stop_runtime(service_id: str):
        return runtime_action(service_id, "stop")

    return router
