"""Regression tests for the CIFAR-10 download script used by CI.

The download logic used to live inline in two workflow files, where a
regression - or a drift between the two copies - could only be caught by a
failing CI run. It now lives in ``backend/scripts/fetch_cifar10.sh`` and is
exercised here with a stubbed ``curl`` so the behaviour that broke CI (a
corrupt cache poisoning itself, a partial download being spliced with the
bytes of another source) is pinned by tests.

The stub records the full argument vector of every ``curl`` call, so the
tests can assert on the flags the script passes and not only on the files
that appear afterwards.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "backend" / "scripts" / "fetch_cifar10.sh"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
WORKFLOW_FILES = ("backend-ci.yml", "image-classifier-ci.yml")

PAYLOAD = b"cifar-10-python-archive-bytes" * 64
CORRUPT_PAYLOAD = b"corrupted" * 8
OTHER_PAYLOAD = b"a-different-archive" * 64
ARCHIVE_NAME = "cifar-10-python.tar.gz"

# The digests of the upstream archive, as pinned by the download script and
# by the cache key of the workflows.
PINNED_MD5 = "c58f30108f718f92721af3b95e74349a"
PINNED_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
CACHE_KEY = f"cifar10-python-md5-{PINNED_MD5[:8]}-v2"
CACHE_PATH = "/home/runner/.cache/cnn-cifar10"
MIRROR_URL = (
    "https://web.archive.org/web/2020id_/" "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
)
ORIGIN_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

_STUB_CURL = """#!/usr/bin/env python3
import json
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
    log.write(json.dumps({"url": url, "size": size, "argv": args}) + "\\n")

fail_substring = os.environ.get("CURL_FAIL_SUBSTRING", "")
if fail_substring and fail_substring in url:
    sys.exit(22)

if "fail" in url:
    sys.exit(22)

if "corrupt" in url:
    with open(os.environ["CURL_CORRUPT_FILE"], "rb") as handle:
        data = handle.read()
else:
    with open(os.environ["CURL_PAYLOAD_FILE"], "rb") as handle:
        data = handle.read()

# 'curl -C -' appends to the existing file, a plain download truncates it.
resume = "-C" in args and target and os.path.exists(target)
with open(target, "ab" if resume else "wb") as handle:
    handle.write(data)
"""


class CurlCall(NamedTuple):
    """One recorded ``curl`` invocation."""

    url: str
    size: int
    argv: list[str]


def _md5(data: bytes) -> str:
    try:
        return hashlib.md5(data, usedforsecurity=False).hexdigest()
    except TypeError:  # pragma: no cover - Python < 3.9
        return hashlib.md5(data).hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_stub_curl(bin_dir: Path) -> None:
    stub = bin_dir / "curl"
    stub.write_text(_STUB_CURL)
    stub.chmod(0o755)


def _run_script(
    tmp_path: Path,
    urls: list[str] | None = None,
    existing_archive: bytes | None = None,
    fail_substring: str | None = None,
    md5: str | None = None,
    sha256: str | None = None,
):
    """Runs the download script with a stubbed curl.

    ``urls=None`` keeps the production source list of the script, so the
    default mirror order can be observed. ``md5``/``sha256`` override the
    pinned digests the script verifies against.

    Returns the completed process and the recorded curl calls.
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
            "CIFAR10_MD5": md5 or _md5(PAYLOAD),
            "CIFAR10_SHA256": sha256 or _sha256(PAYLOAD),
            "CIFAR10_ATTEMPTS": "2",
            "CIFAR10_TIMEOUT": "30",
        }
    )
    if urls is not None:
        env["CIFAR10_URLS"] = "\n".join(urls)
    if fail_substring is not None:
        env["CURL_FAIL_SUBSTRING"] = fail_substring

    process = subprocess.run(
        ["bash", str(SCRIPT), str(cache_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    calls: list[CurlCall] = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line:
                continue
            calls.append(CurlCall(**json.loads(line)))
    return process, calls


def _script_default(env_name: str, length: int) -> str:
    """Returns the default digest the script pins for ``CIFAR10_<env_name>``."""
    match = re.search(
        rf"CIFAR10_{env_name}:-([0-9a-f]{{{length}}})",
        SCRIPT.read_text(),
    )
    assert match, f"the script does not pin a default for CIFAR10_{env_name}"
    return match.group(1)


def _cifar_cache_block(text: str, name: str) -> str:
    """Returns the workflow section that restores, downloads and saves CIFAR-10.

    The block ends before the torch weight cache, because that cache
    legitimately uses the plain ``actions/cache`` action.
    """
    start_marker = "Restore CIFAR-10 dataset cache"
    end_marker = "Cache pretrained model weights"
    assert start_marker in text, f"{name} has no CIFAR-10 restore step"
    assert end_marker in text, f"{name} has no torch weight cache step"
    start = text.index(start_marker)
    end = text.index(end_marker)
    assert start < end, f"{name} saves the torch cache before the CIFAR-10 cache"
    return text[start:end]


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
    assert calls == [CurlCall("https://example.test/good", 0, calls[0].argv)]
    # Both digests are verified: md5sum and sha256sum each print one OK line.
    assert process.stdout.count(f"{ARCHIVE_NAME}: OK") == 2


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
    # The second attempt sees the partial file and resumes it.
    assert calls[1].size == len(CORRUPT_PAYLOAD)
    # 'curl -C -' is what turns the partial file into a resumed download.
    assert "-C" in calls[1].argv
    assert "-" in calls[1].argv


def test_partial_download_is_dropped_when_the_source_changes(tmp_path: Path) -> None:
    process, calls = _run_script(
        tmp_path,
        ["https://example.test/corrupt", "https://example.test/good"],
    )

    assert process.returncode == 0, process.stdout + process.stderr
    # The second source starts from byte 0: resuming on top of the first
    # source's bytes would splice two different archives together.
    good_calls = [call for call in calls if call.url == "https://example.test/good"]
    assert good_calls == [CurlCall("https://example.test/good", 0, good_calls[0].argv)]


def test_a_matching_md5_does_not_excuse_a_wrong_sha256(tmp_path: Path) -> None:
    """The sha256 pin must be enforced on its own.

    With only the md5 checked, an archive that matches the md5 but not the
    sha256 would be accepted, so the sha256 line must decide the result.
    """
    process, calls = _run_script(
        tmp_path,
        ["https://example.test/good"],
        md5=_md5(PAYLOAD),
        sha256=_sha256(OTHER_PAYLOAD),
    )

    assert process.returncode == 1
    assert "::error::CIFAR-10 archive could not be downloaded and verified." in (process.stdout)
    assert len(calls) == 2


def test_the_script_pins_the_upstream_digests() -> None:
    assert _script_default("MD5", 32) == PINNED_MD5
    assert _script_default("SHA256", 64) == PINNED_SHA256


def test_the_default_sources_prefer_the_archive_mirror(tmp_path: Path) -> None:
    """The mirror is tried first, the throttled origin is the fallback.

    The cs.toronto.edu origin serves the archive at ~80 KB/s, which does not
    fit into the CI budget, so the mirror has to come first.
    """
    process, calls = _run_script(
        tmp_path,
        urls=None,
        fail_substring="web.archive.org",
    )

    assert process.returncode == 0, process.stdout + process.stderr
    assert calls[0].url == MIRROR_URL
    assert calls[-1].url == ORIGIN_URL
    # The mirror is tried before the origin.
    assert calls.index(next(c for c in calls if c.url == ORIGIN_URL)) > calls.index(
        next(c for c in calls if c.url == MIRROR_URL)
    )


def test_workflows_share_the_script_and_one_cache_key() -> None:
    for name in WORKFLOW_FILES:
        text = (WORKFLOW_DIR / name).read_text()
        assert "scripts/fetch_cifar10.sh" in text, f"{name} does not use the script"
        assert CACHE_KEY in text, f"{name} uses another key"
        # The download logic must not be duplicated inline any more.
        assert "md5sum -c --status" not in text, f"{name} inlines the download logic"


def test_workflows_save_the_cifar_cache_only_after_it_was_verified() -> None:
    """Pins the fix that stopped the cache from poisoning itself.

    A plain ``actions/cache`` saves the cache even when the download failed,
    so a truncated archive was stored under the immutable key and every later
    run restored the broken file. The cache must therefore be restored
    explicitly, verified by the download script, and saved explicitly only
    when the restore did not hit.
    """
    for name in WORKFLOW_FILES:
        text = (WORKFLOW_DIR / name).read_text()
        block = _cifar_cache_block(text, name)

        assert "actions/cache/restore@" in block, f"{name} does not restore explicitly"
        assert "actions/cache/save@" in block, f"{name} does not save explicitly"
        # The plain action saves unconditionally, which is what poisoned the
        # cache. The torch cache uses it legitimately, but it lives outside
        # this block.
        assert "uses: actions/cache@" not in block, f"{name} uses the unsafe plain cache"
        assert (
            "if: steps.cifar-cache.outputs.cache-hit != 'true'" in block
        ), f"{name} saves the cache even on a cache hit"
        # The archive must be downloaded and verified before it can be saved.
        assert block.index("fetch_cifar10.sh") < block.index(
            "actions/cache/save@"
        ), f"{name} saves the cache before the archive is verified"
        assert CACHE_PATH in block, f"{name} caches another path"


def test_workflow_restore_and_save_agree_on_key_and_path() -> None:
    """The restore and the save step must describe the same cache.

    A drifting key would restore one entry and save another, and a drifting
    path would save an archive the job never verified.
    """
    for name in WORKFLOW_FILES:
        text = (WORKFLOW_DIR / name).read_text()
        block = _cifar_cache_block(text, name)

        keys = set(re.findall(r"(?m)^\s*key:\s*(\S+)\s*$", block))
        assert keys == {CACHE_KEY}, f"{name} uses more than one cache key: {keys}"

        paths = set(re.findall(r"(?m)^\s*path:\s*(\S+)\s*$", block))
        assert paths == {CACHE_PATH}, f"{name} restores and saves other paths: {paths}"

        restore_keys = re.findall(r"(?m)^\s*restore-keys:\s*(.+)$", block)
        assert restore_keys, f"{name} has no restore-keys"
        for value in restore_keys:
            for entry in value.split():
                assert entry == CACHE_KEY, f"{name} can restore a stale key: {entry}"

        # The script must be invoked by a run step, not only mentioned in a
        # comment.
        run_lines = re.findall(r"(?m)^\s*run:\s*(.*fetch_cifar10\.sh.*)$", block)
        assert len(run_lines) == 1, f"{name} does not run the script exactly once"
        assert f"scripts/fetch_cifar10.sh {CACHE_PATH}" in run_lines[0]

        # The download logic must not be inlined in the workflow again.
        assert not re.search(
            r"(?m)^\s*(run:.*)?\b(md5sum|sha256sum|wget|curl)\b",
            block,
        ), f"{name} inlines the download logic"
