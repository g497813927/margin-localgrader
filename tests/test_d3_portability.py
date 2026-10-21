"""Exercise platform choices without downloading or loading a model."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts import run_d3


@pytest.mark.parametrize("platform,library", [("win32", "ucrtbase.dll"), ("linux", None), ("darwin", None)])
def test_native_output_flush_uses_python_platform_c_runtime(monkeypatch, platform, library):
    calls = []
    class Flush:
        def __call__(self, stream):
            assert stream is None
            return 0
    flush = Flush()
    def load(name):
        calls.append(name)
        return SimpleNamespace(fflush=flush)
    monkeypatch.setattr(run_d3.sys, "platform", platform)
    monkeypatch.setattr(run_d3.ctypes, "CDLL", load)
    run_d3.flush_native_output()
    assert calls == [library]
    assert flush.argtypes == [run_d3.ctypes.c_void_p]
    assert flush.restype is run_d3.ctypes.c_int


@pytest.mark.parametrize("platform,device", [("win32", "cpu"), ("linux", "cpu"), ("darwin", "mps")])
def test_cli_device_default_matches_available_platform_family(monkeypatch, platform, device):
    monkeypatch.setattr(run_d3.sys, "platform", platform)
    monkeypatch.setattr(run_d3, "configure_offline", lambda: None)
    monkeypatch.setattr(run_d3, "silence_process", nullcontext)
    calls = []
    def load(path, chosen_device, max_context):
        calls.append(chosen_device)
        raise KeyboardInterrupt
    monkeypatch.setattr(run_d3, "load_runtime", load)
    assert run_d3.main([]) == 0
    assert calls == [device]


def test_current_platform_native_flush_is_available():
    # Exercise the actual exported symbol on each native CI runner.
    run_d3.flush_native_output()
