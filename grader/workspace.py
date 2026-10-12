"""Explicit settings-only persistence. Student work never enters this schema."""
import json
import os
from pathlib import Path
import secrets
import stat
import threading
from typing import Literal

from pydantic import Field, model_validator

from .schemas import ExplanationSettings, Level, Settings, StrictModel, model_identity
from pydantic import field_validator

MAX_WORKSPACE_BYTES = 65536
ROOT = Path(__file__).resolve().parents[1]


class WorkspaceEntry(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    enabled: bool = Field(default=True, strict=True)
    preset_id: Literal["laya", "clef", "startlux", "d3"] | None = None
    settings: Settings

    _valid_id = field_validator("id")(Level.valid_id.__func__)


class Workspace(StrictModel):
    schema_version: int = Field(default=1, strict=True, ge=1, le=1)
    models: list[WorkspaceEntry] = Field(max_length=8)
    explanation_settings: ExplanationSettings = Field(default_factory=ExplanationSettings)

    @model_validator(mode="after")
    def unique_models(self):
        if len({entry.id for entry in self.models}) != len(self.models):
            raise ValueError("Workspace row IDs must be unique.")
        enabled = [model_identity(entry.settings) for entry in self.models if entry.enabled]
        if len(set(enabled)) != len(enabled):
            raise ValueError("Enabled model configurations must be different.")
        return self


def model_catalog():
    presets = [
        ("laya", "Laya", "Embedded local decision model; runs inside Margin.",
         Settings(), None),
        ("clef", "Clef-Flash", "Local GGUF decision model on port 1235.",
         Settings(provider="systemone", model="clef-flash", endpoint="http://127.0.0.1:1235/v1/systemone"), "clef"),
        ("startlux", "StartLux Decision 4B", "Local GGUF decision model on port 1236.",
         Settings(provider="systemone", model="StartLux-Decision-v1-4B-Q8_0", endpoint="http://127.0.0.1:1236/v1/systemone"), "startlux"),
        ("d3", "Decision 3.0 Mini", "Local PyTorch decision model on port 1238.",
         Settings(provider="systemone", model="d3-mini", endpoint="http://127.0.0.1:1238/v1/systemone"), "d3"),
    ]
    return {"models": [{"id": key, "name": name, "description": description,
                         "settings": settings.model_dump(), "service_id": service,
                         "guide_anchor": key}
                        for key, name, description, settings, service in presets],
            "max_models": 8, "providers": ["laya", "systemone"]}


def default_workspace():
    return Workspace(models=[WorkspaceEntry(id=item["id"], preset_id=item["id"],
                                             settings=Settings(**item["settings"]))
                             for item in model_catalog()["models"]])


def parse_workspace(data: bytes):
    if len(data) > MAX_WORKSPACE_BYTES:
        raise ValueError("Workspace settings are too large.")
    # Reject duplicate JSON keys instead of silently retaining the final value.
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field.")
            result[key] = value
        return result
    return Workspace.model_validate(json.loads(data, object_pairs_hook=unique_object))


class WorkspaceStore:
    """Fixed file, private permissions, atomic replacement, no symlink following."""
    def __init__(self, root=ROOT):
        self.directory = Path(root) / "config"
        self.filename = "model-workspace.json"
        self.lock = threading.Lock()

    def _open_directory(self, create=False):
        if create:
            self.directory.mkdir(mode=0o700, exist_ok=True)
        return os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    @staticmethod
    def _check_file(directory, filename):
        try:
            info = os.stat(filename, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Unsafe workspace file.")
        return True

    def load(self):
        with self.lock:
            try:
                directory = self._open_directory()
            except FileNotFoundError:
                return default_workspace(), False
            try:
                if not self._check_file(directory, self.filename):
                    return default_workspace(), False
                descriptor = os.open(self.filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
                with os.fdopen(descriptor, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise ValueError("Unsafe workspace file.")
                    data = stream.read(MAX_WORKSPACE_BYTES + 1)
                return parse_workspace(data), True
            finally:
                os.close(directory)

    def save(self, workspace):
        # Revalidate, even for internal callers; never serialize an arbitrary object.
        checked = Workspace.model_validate(workspace.model_dump())
        data = (json.dumps(checked.model_dump(), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if len(data) > MAX_WORKSPACE_BYTES:
            raise ValueError("Workspace settings are too large.")
        with self.lock:
            directory = self._open_directory(create=True)
            temporary = ".workspace-" + secrets.token_hex(12)
            created = False
            try:
                self._check_file(directory, self.filename)
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     0o600, dir_fd=directory)
                created = True
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.filename, src_dir_fd=directory, dst_dir_fd=directory)
                created = False
                os.fsync(directory)
            finally:
                if created:
                    os.unlink(temporary, dir_fd=directory)
                os.close(directory)
        return checked
