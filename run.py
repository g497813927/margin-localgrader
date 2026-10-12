"""Start the local UI. Never installs packages or downloads model files."""
import os
import logging
from pathlib import Path
import sys

os.chdir(Path(__file__).resolve().parent)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["OTEL_SDK_DISABLED"] = "true"
logging.disable(logging.CRITICAL)

if __name__ == "__main__":
    # Some native numerical libraries write directly to stdout/stderr. The
    # application is explicitly no-log; all actionable errors are shown in UI.
    null_output = open(os.devnull, "w")
    os.dup2(null_output.fileno(), sys.stdout.fileno())
    os.dup2(null_output.fileno(), sys.stderr.fileno())
    import uvicorn

    port = int(os.environ.get("MARGIN_PORT", "8765"))
    uvicorn.run("grader.app:app", host="127.0.0.1", port=port,
                access_log=False, log_config=None, log_level="critical")
