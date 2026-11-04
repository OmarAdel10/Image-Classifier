"""Regression tests for the CIFAR-10 download script used by CI.

The download logic used to live inline in two workflow files, where a
regression - or a drift between the two copies - could only be caught by a
failing CI run. It now lives in ``backend/scripts/fetch_cifar10.sh`` and is
exercised here with a stubbed ``curl`` so the behaviour that broke CI (a
corrupt cache poisoning itself, a partial download being spliced with the
bytes of another source) is pinned by tests.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "backend" / "scripts" / "fetch_cifar10.sh"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
WORKFLOW_FILES = ("backend-ci.yml", "image-classifier-ci.yml")

PAYLOAD = b"cifar-10-python-archive-bytes" * 64
CORRUPT_PAYLOAD = b"corrupted" * 8
ARCHIVE_NAME = "cifar-10-python.tar.gz"

_STUB_CURL = """#!/usr/bin/env python3
import os
import sys

args = sys.argv[1:]
target = None
url = args[-1]
for index, arg in enumerate(args):
    if arg == "-o":
        target = args[index + 1]

size = os.path.getsize(target) if target and os.path.exists(target) else 0
with open(os.environ["CURL_LOG"], "a") as log:
    log.write(f"{url}|{size}\\n")

if "fail" in url:
    sys.exit(22)

if "corrupt" in url:
    with open(os.environ["CURL_CORRUPT_FILE"], "rb") as handle:
        data = handle.read()
else:
    with open(os.environ["CURL_PAYLOAD_FILE"], "rb") as handle:
        data = handle.read()

with open(target, "wb") as handle:
    handle.write(data)
"""


def _md5(data: bytes) -> str:
    try:
        return hashlib.md5(data, usedforsecurity=False).hexdigest()
    except TypeError:  # pragma: no cover - Python < 3.9
        return hashlib.md5(data).hexdigest()


def _write_stub_curl(bin_dir: Path) -> None:
    stub = bin_dir / "curl"
    stub.write_text(_STUB_CURL)
    stub.chmod(0o755)


def _run_script(
    tmp_path: Path,
    urls: list[str],
    existing_archive: bytes | None = None,
):
    """Runs the download script with a stubbed curl.

    Returns the completed process and the recorded curl calls as
    ``(url, size_of_target_before_the_call)`` pairs.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_stub_curl(bin_dir)

    log_path = tmp_path / "curl.log"
    payload_file = tmp_path / "payload.bin"
    payload_file.write_bytes(PAYLOAD)
    corrupt_file = tmp_path / "corrupt.bin"
    corrupt_file.write_bytes(CORRUPT_PAYLOAD)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    if existing_archive is not None:
        (cache_dir / ARCHIVE_NAME).write_bytes(existing_archive)

    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "CURL_LOG": str(log_path),
            "CURL_PAYLOAD_FILE": str(payload_file),
            "CURL_CORRUPT_FILE": str(corrupt_file),
            "CIFAR10_MD5": _md5(PAYLOAD),
            "CIFAR10_SHA256": hashlib.sha256(PAYLOAD).hexdigest(),
            "CIFAR10_URLS": "\n".join(urls),
            "CIFAR10_ATTEMPTS": "2",
            "CIFAR10_TIMEOUT": "30",
        }
    )

    process = subprocess.run(
        ["bash", str(SCRIPT), str(cache_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    calls: list[tuple[str, int]] = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line:
                continue
            url, _, size = line.partition("|")
            calls.append((url, int(size)))
    return process, calls


def test_script_exists() -> None:
    assert SCRIPT.is_file(), f"missing CIFAR-10 download script: {SCRIPT}"


def test_verified_cache_is_reused_without_downloading(tmp_path: Path) -> None:
    process, calls = _run_script(tmp_path, ["https://example.test/good"], PAYLOAD)

    assert process.returncode == 0, process.stdout + process.stderr
    assert calls == []
    assert "verified from the restored cache" in process.stdout


def test_corrupt_cache_is_purged_and_downloaded_again(tmp_path: Path) -> None:
    process, calls = _run_script(tmp_path, ["https://example.test/good"], b"truncated-archive")

    assert process.returncode == 0, process.stdout + process.stderr
    # The corrupt archive was removed before the download, so the download
    # started from byte 0 instead of resuming on top of the broken file.
    assert calls == [("https://example.test/good", 0)]
    assert "OK" in process.stdout


def test_failed_download_reports_an_error(tmp_path: Path) -> None:
    process, calls = _run_script(tmp_path, ["https://example.test/fail"])

    assert process.returncode == 1
    assert "::error::CIFAR-10 archive could not be downloaded and verified." in (process.stdout)
    assert len(calls) == 2


def test_partial_download_is_kept_between_attempts_of_the_same_source(
    tmp_path: Path,
) -> None:
    process, calls = _run_script(tmp_path, ["https://example.test/corrupt"])

    assert process.returncode == 1
    assert len(calls) == 2
    # The second attempt sees the partial file, so 'curl -C -' can resume it.
    assert calls[1][1] == len(CORRUPT_PAYLOAD)


def test_partial_download_is_dropped_when_the_source_changes(tmp_path: Path) -> None:
    process, calls = _run_script(
        tmp_path,
        ["https://example.test/corrupt", "https://example.test/good"],
    )

    assert process.returncode == 0, process.stdout + process.stderr
    # The second source starts from byte 0: resuming on top of the first
    # source's bytes would splice two different archives together.
    good_calls = [call for call in calls if call[0] == "https://example.test/good"]
    assert good_calls == [("https://example.test/good", 0)]


def test_workflows_share_the_script_and_one_cache_key() -> None:
    for name in WORKFLOW_FILES:
        text = (WORKFLOW_DIR / name).read_text()
        assert "scripts/fetch_cifar10.sh" in text, f"{name} does not use the script"
        assert "cifar10-python-md5-c58f3010-v2" in text, f"{name} uses another key"
        # The download logic must not be duplicated inline any more.
        assert "md5sum -c --status" not in text, f"{name} inlines the download logic"
