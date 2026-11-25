#!/usr/bin/env python3
"""Backend launcher for the CNN Classifier API.

Starts uvicorn with the configured host/port and watches stdin for parent
process termination (exits when stdin closes). Handles SIGTERM/SIGINT
gracefully.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def _resolve_backend_src() -> Path:
    """Resolve the backend/src directory for PYTHONPATH."""
    # This script is at backend/scripts/run_backend.py
    # backend/src is one level up from scripts/
    return Path(__file__).resolve().parent.parent / "src"


def _watch_stdin() -> None:
    """Watch stdin for EOF (parent process exit) and terminate the child."""
    try:
        sys.stdin.read()
    except Exception:
        pass
    # stdin closed -> parent died -> exit
    os._exit(0)


def main() -> int:
    """Launch the backend server."""
    # Ensure PYTHONPATH includes backend/src
    src_path = _resolve_backend_src()
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(src_path)

    # Default to 127.0.0.1:8000; allow override via CLI args or env
    host = os.getenv("CNN_HOST", "127.0.0.1")
    port = os.getenv("CNN_PORT", "8000")

    # Pass through any additional CLI arguments to uvicorn
    uvicorn_args = sys.argv[1:]

    # Build uvicorn command
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "cnn_classifier.api.main:app",
        "--host",
        host,
        "--port",
        port,
        *uvicorn_args,
    ]

    logger_name = "cnn_classifier.launcher"
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    launcher_logger = logging.getLogger(logger_name)

    launcher_logger.info("Starting backend: %s", " ".join(cmd))
    launcher_logger.info("PYTHONPATH=%s", env["PYTHONPATH"])

    # Start uvicorn as a child process
    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE)

    # Watch stdin in a separate thread: when parent closes stdin, we terminate the child
    stdin_watcher = threading.Thread(target=_watch_stdin, daemon=True)
    stdin_watcher.start()

    # Signal handlers for graceful shutdown
    def _signal_handler(signum, _frame):
        launcher_logger.info("Received signal %s, shutting down backend...", signum)
        if proc.poll() is None:
            proc.terminate()
            # Wait up to 5 seconds for graceful shutdown
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                launcher_logger.warning("Backend did not terminate gracefully, killing...")
                proc.kill()
                proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Wait for the child process
    try:
        return_code = proc.wait()
        launcher_logger.info("Backend exited with code %d", return_code)
        return return_code
    except KeyboardInterrupt:
        launcher_logger.info("Interrupted, shutting down...")
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        return 130


if __name__ == "__main__":
    sys.exit(main())