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
import time
from pathlib import Path
from typing import NamedTuple

import yaml

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
import time

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

if "slow" in url:
    # A stalled transfer: the script has to abort it with its own timeout
    # instead of holding the job for the whole budget.
    time.sleep(30)

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
    cache_dir: Path | None = None,
    timeout: int = 30,
):
    """Runs the download script with a stubbed curl.

    ``urls=None`` keeps the production source list of the script, so the
    default mirror order can be observed. ``md5``/``sha256`` override the
    pinned digests the script verifies against. ``cache_dir`` overrides the
    cache directory, so a test can point the script at a directory that does
    not exist yet. ``timeout`` is the per-attempt budget in seconds.

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

    if cache_dir is None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(exist_ok=True)
    elif existing_archive is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
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
            "CIFAR10_TIMEOUT": str(timeout),
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
    assert [call.url for call in calls] == ["https://example.test/good"]
    assert [call.size for call in calls] == [0]
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
    assert [call.size for call in good_calls] == [0]


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


def test_the_first_verified_source_stops_the_download(tmp_path: Path) -> None:
    """A verified archive must end the download loop.

    Without the break, the next source would resume on top of the complete
    archive, which produces a corrupt file and fails the job.
    """
    process, calls = _run_script(
        tmp_path,
        ["https://example.test/first", "https://example.test/second"],
    )

    assert process.returncode == 0, process.stdout + process.stderr
    assert len(calls) == 1
    assert calls[0].url == "https://example.test/first"


def test_a_matching_sha256_does_not_excuse_a_wrong_md5(tmp_path: Path) -> None:
    """The md5 pin must be enforced on its own.

    The archive matches the sha256 but not the md5, so the md5 line has to
    decide the result.
    """
    process, calls = _run_script(
        tmp_path,
        ["https://example.test/good"],
        md5=_md5(OTHER_PAYLOAD),
        sha256=_sha256(PAYLOAD),
    )

    assert process.returncode == 1
    assert "::error::CIFAR-10 archive could not be downloaded and verified." in (process.stdout)
    assert len(calls) == 2


def test_the_script_fails_fast_and_rejects_error_pages() -> None:
    """The script must stop on the first error and never keep an error page.

    'set -euo pipefail' makes an unexpected command failure end the script,
    and 'curl -f' turns an HTTP error page into a failed download instead of
    a file that is verified as broken.
    """
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "curl -fL" in text


def _load_workflow(name: str) -> dict:
    """Parses a workflow file, so a broken YAML fails the suite."""
    return yaml.safe_load((WORKFLOW_DIR / name).read_text())


def _backend_job(name: str) -> dict:
    workflow = _load_workflow(name)
    jobs = workflow["jobs"]
    assert "python-backend" in jobs, f"{name} has no python-backend job"
    return jobs["python-backend"]


def _step_by_name(job: dict, name: str) -> tuple[int, dict]:
    """Finds a step by its exact name.

    The exact name is pinned: a renamed step must fail the suite instead of
    silently changing which step the assertions cover.
    """
    for index, step in enumerate(job["steps"]):
        if step.get("name") == name:
            return index, step
    raise AssertionError(f"no step named {name!r}")


def test_the_backend_job_has_a_120_minute_timeout() -> None:
    """The job must be allowed the budget the download script can use.

    The archive is ~170 MB and the throttled origin served it at ~80 KB/s,
    so the old 45 minute budget could not finish the download. The timeout
    is part of the fix, so it needs a regression test.
    """
    for name in WORKFLOW_FILES:
        job = _backend_job(name)
        assert job.get("timeout-minutes") == 120, f"{name} has another job timeout"


def test_the_workflow_is_valid_yaml_with_the_cache_steps_in_order() -> None:
    """The cache contract is asserted on the parsed workflow, not on text.

    A regex over the raw file cannot tell a broken YAML file from a working
    one, and it cannot see that the save step lost its gate.
    """
    for name in WORKFLOW_FILES:
        job = _backend_job(name)

        restore_index, restore = _step_by_name(job, "Restore CIFAR-10 dataset cache")
        download_index, download = _step_by_name(job, "Download and verify CIFAR-10")
        save_index, save = _step_by_name(job, "Save verified CIFAR-10 dataset cache")

        assert (
            restore_index < download_index < save_index
        ), f"{name} runs the cache steps out of order"

        assert restore["uses"].startswith("actions/cache/restore@")
        assert restore["id"] == "cifar-cache"
        assert restore["with"]["path"] == CACHE_PATH
        assert restore["with"]["key"] == CACHE_KEY

        # The script must be run through bash: the file is not executable, so
        # './scripts/fetch_cifar10.sh' would fail in CI.
        assert download["working-directory"] == "backend"
        assert download["run"].strip() == f"bash scripts/fetch_cifar10.sh {CACHE_PATH}"

        assert save["uses"].startswith("actions/cache/save@")
        # The gate is explicit: the cache is saved only when the restore did
        # not hit and the job itself succeeded.
        assert save["if"].strip() == "success() && steps.cifar-cache.outputs.cache-hit != 'true'"
        assert save["with"]["path"] == CACHE_PATH
        assert save["with"]["key"] == CACHE_KEY


def test_the_download_step_always_runs_and_always_fails_the_job() -> None:
    """The download step must run on every job and must fail the job.

    A conditional download would skip the verification of a restored
    archive, which is the path that let a poisoned cache through. A
    tolerated download failure would let the save step store an unverified
    archive under the immutable cache key, which is how the cache poisoned
    itself in the first place.
    """
    for name in WORKFLOW_FILES:
        job = _backend_job(name)
        _, download = _step_by_name(job, "Download and verify CIFAR-10")

        assert "if" not in download, f"{name} can skip the CIFAR-10 download"
        assert "continue-on-error" not in download, f"{name} ignores a failed CIFAR-10 download"


def test_the_per_attempt_timeout_stops_a_stalled_download(tmp_path: Path) -> None:
    """A stalled transfer must be aborted by the per-attempt timeout.

    The throttled origin once kept a job running for 37 minutes without
    delivering the archive, so the timeout that wraps the download has to
    be applied - a pinned default value is not enough.
    """
    started = time.monotonic()
    process, calls = _run_script(
        tmp_path,
        ["https://example.test/slow"],
        timeout=1,
    )
    elapsed = time.monotonic() - started

    assert process.returncode == 1
    # Two attempts, each aborted after one second. Without the timeout
    # wrapper the stalled transfer would hold the script for a minute.
    assert elapsed < 15, f"the stalled download was not aborted ({elapsed:.1f} s)"
    assert len(calls) == 2


def test_the_script_pins_the_download_budget() -> None:
    """The attempt count and the per-attempt timeout are part of the fix.

    They decide how long a broken source can hold the job, so their
    production defaults must not change unnoticed.
    """
    text = SCRIPT.read_text()
    assert re.search(r"CIFAR10_ATTEMPTS:-(\d+)", text).group(1) == "2"
    assert re.search(r"CIFAR10_TIMEOUT:-(\d+)", text).group(1) == "1200"


def test_the_download_budget_fits_inside_the_job_timeout() -> None:
    """The worst case has to end before the job is killed.

    Two sources with two attempts each and the pinned per-attempt timeout
    decide how long a broken mirror can hold the job. When that exceeds the
    job timeout, the job is killed without the script's own error message.
    """
    text = SCRIPT.read_text()
    attempts = int(re.search(r"CIFAR10_ATTEMPTS:-(\d+)", text).group(1))
    per_attempt = int(re.search(r"CIFAR10_TIMEOUT:-(\d+)", text).group(1))
    default_sources = re.search(r"CIFAR10_URLS:-(.*?)\}", text, re.S)
    assert default_sources, "the script does not pin a default source list"
    sources = len([line for line in default_sources.group(1).splitlines() if line.strip()])

    for name in WORKFLOW_FILES:
        job = _backend_job(name)
        job_timeout = int(job["timeout-minutes"]) * 60
        worst_case = sources * attempts * per_attempt
        assert worst_case < job_timeout, (
            f"{name} can spend {worst_case} s downloading, which does not fit "
            f"inside its {job_timeout} s job timeout"
        )


def test_the_download_aborts_a_stalled_transfer(tmp_path: Path) -> None:
    """A stalled transfer must not hold the job for the whole budget.

    The throttled origin once kept the job running for 37 minutes without
    delivering the archive, so the speed guard has to stay.
    """
    process, calls = _run_script(tmp_path, ["https://example.test/good"])

    assert process.returncode == 0, process.stdout + process.stderr
    argv = calls[0].argv
    for flag, value in (
        ("--speed-limit", "1024"),
        ("--speed-time", "120"),
        ("--connect-timeout", "30"),
        ("--retry", "3"),
    ):
        assert flag in argv, f"{flag} is missing from the curl call"
        assert value in argv, f"{flag} {value} is missing from the curl call"


def test_the_script_creates_a_missing_cache_directory(tmp_path: Path) -> None:
    """The script has to create its cache directory itself.

    A fresh runner has no cache directory, and the workflow does not create
    one before calling the script.
    """
    cache_dir = tmp_path / "missing" / "cache"
    process, _calls = _run_script(
        tmp_path,
        ["https://example.test/good"],
        cache_dir=cache_dir,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    assert (cache_dir / ARCHIVE_NAME).is_file()


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
            "if: success() && steps.cifar-cache.outputs.cache-hit != 'true'" in block
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
