"""Run real Win32 settings operations on Windows CI, never simulated student data."""
import os

import pytest

from grader.workspace import WorkspaceStore, default_workspace, Workspace

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Exercises native Win32 handles")


def test_native_windows_workspace_roundtrip(tmp_path):
    store = WorkspaceStore(tmp_path)
    assert store.load()[1] is False
    original = default_workspace()
    original.vision_settings.model = "qwen3.5:2b-q4_K_M"
    store.save(original)
    assert store.load() == (original, True)
    store.save(Workspace(models=[]))
    assert store.load()[0].models == []
    assert len(list(store.directory.iterdir())) == 1


def test_windows_hardlink_not_overwritten(tmp_path):
    store = WorkspaceStore(tmp_path)
    store.directory.mkdir()
    original = tmp_path / "original.json"
    original.write_text("private-settings-placeholder")
    os.link(original, store.directory / store.filename)
    with pytest.raises(ValueError):
        store.save(default_workspace())
    assert original.read_text() == "private-settings-placeholder"


def test_windows_directory_pin_detects_rename(tmp_path):
    from grader.windows_files import checked_handle
    directory = tmp_path / "config"
    directory.mkdir()
    with pytest.raises(OSError):
        with checked_handle(directory, directory=True):
            directory.rename(tmp_path / "moved")
            directory.mkdir()  # A replacement path must not pass the identity check.
