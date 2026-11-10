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
    "https://web.archive.org/web/20201130004040id_/"
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
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

if "errorpage" in url:
    # An HTTP error page: the server answers with an error body instead of
    # the archive. The body must never survive as the cached archive.
    with open(target, "wb") as handle:
        handle.write(b"<html>500 Internal Server Error</html>")
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

# 'curl -C -' resumes the transfer at the current size of the file, so the
# stub serves the payload from that offset instead of appending all of it.
offset = 0
if "-C" in args:
    index = args.index("-C")
    value = args[index + 1] if index + 1 < len(args) else "-"
    if value == "-":
        offset = size
    elif value.isdigit():
        offset = int(value)

# 'curl -C -' appends to the existing file, a plain download truncates it.
resume = "-C" in args and target and os.path.exists(target)
mode = "ab" if resume else "wb"

if "partial" in url and offset == 0:
    # A transfer that drops half way through: the first attempt delivers
    # half of the archive and fails, the way a lost connection does.
    with open(target, mode) as handle:
        handle.write(data[: len(data) // 2])
    sys.exit(18)

with open(target, mode) as handle:
    handle.write(data[offset:])
"""


# A shell stub for the stalled-transfer test. It records whether the transfer
# was killed or ran to completion, and then blocks on a read from a FIFO that
# never receives data. The shell waits on the read itself, so the TERM signal
# of the per-attempt timeout is handled right away. A foreground child (a
# `sleep`) would defer the trap until that child exits, which made the test
# depend on how the signal was delivered.
_STUB_CURL_SHELL = """#!/bin/sh
url=""
out=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "-o" ]; then
    out="$arg"
  fi
  prev="$arg"
  url="$arg"
done
size=0
if [ -n "$out" ] && [ -f "$out" ]; then
  size=$(wc -c < "$out")
fi
trap 'printf "killed\\n" >> "$CURL_KILLED_LOG"; exit 143' TERM
printf '{"url": "%s", "size": %s, "argv": []}\\n' "$url" "$size" >> "$CURL_LOG"
exec 3< "$CURL_STALL_FIFO"
while :; do
  read -r _line <&3 || true
done
"""


# A stub for the `timeout` command. It records the argument vector the script
# passes to `timeout` and then runs the wrapped command without waiting, so the
# test can assert on the per-attempt budget the script configured.
_STUB_TIMEOUT = """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["TIMEOUT_ARGV_LOG"], "a") as log:
    log.write(json.dumps({"argv": args}) + "\\n")

index = 0
while index < len(args) and args[index].startswith("-"):
    index += 1
command = args[index + 1 :]
os.execvp(command[0], command)
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


def _write_stub_curl(bin_dir: Path, body: str = _STUB_CURL) -> None:
    stub = bin_dir / "curl"
    stub.write_text(body)
    stub.chmod(0o755)


def _default_source_count() -> int:
    """Returns the number of download sources the script uses by default."""
    match = re.search(r'DEFAULT_URLS="(.*?)"', SCRIPT.read_text(), re.S)
    assert match, "the script does not pin a default source list"
    return len([line for line in match.group(1).splitlines() if line.strip()])


def _run_script(
    tmp_path: Path,
    urls: list[str] | None = None,
    existing_archive: bytes | None = None,
    fail_substring: str | None = None,
    md5: str | None = None,
    sha256: str | None = None,
    cache_dir: Path | None = None,
    timeout: int | None = 30,
    attempts: int = 2,
    stub: str = _STUB_CURL,
    timeout_stub: str | None = None,
):
    """Runs the download script with a stubbed curl.

    ``urls=None`` keeps the production source list of the script, so the
    default mirror order can be observed. ``md5``/``sha256`` override the
    pinned digests the script verifies against. ``cache_dir`` overrides the
    cache directory, so a test can point the script at a directory that does
    not exist yet. ``timeout`` is the per-attempt budget in seconds; ``None``
    keeps the default the script pins. ``stub`` replaces the stubbed curl, so
    a test can use a stub that blocks instead of delivering the archive.
    ``timeout_stub`` replaces the ``timeout`` command, so a test can observe
    the budget the script configured.

    Returns the completed process and the recorded curl calls.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_stub_curl(bin_dir, stub)
    if timeout_stub is not None:
        timeout_command = bin_dir / "timeout"
        timeout_command.write_text(timeout_stub)
        timeout_command.chmod(0o755)

    log_path = tmp_path / "curl.log"
    killed_path = tmp_path / "curl.killed.log"
    timeout_argv_path = tmp_path / "timeout.argv.log"
    payload_file = tmp_path / "payload.bin"
    payload_file.write_bytes(PAYLOAD)
    corrupt_file = tmp_path / "corrupt.bin"
    corrupt_file.write_bytes(CORRUPT_PAYLOAD)
    # The stalled stub blocks on this FIFO. It is opened read-write, so the
    # stub never waits for a writer and the read stays blocked until the
    # timeout signals the stub.
    stall_fifo = tmp_path / "stall.fifo"
    if not stall_fifo.exists():
        os.mkfifo(stall_fifo)
    stall_fd = os.open(stall_fifo, os.O_RDWR)

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
            "CURL_KILLED_LOG": str(killed_path),
            "TIMEOUT_ARGV_LOG": str(timeout_argv_path),
            "CURL_PAYLOAD_FILE": str(payload_file),
            "CURL_CORRUPT_FILE": str(corrupt_file),
            "CURL_STALL_FIFO": str(stall_fifo),
            "CIFAR10_MD5": md5 or _md5(PAYLOAD),
            "CIFAR10_SHA256": sha256 or _sha256(PAYLOAD),
            "CIFAR10_ATTEMPTS": str(attempts),
        }
    )
    if timeout is not None:
        env["CIFAR10_TIMEOUT"] = str(timeout)
    if urls is not None:
        env["CIFAR10_URLS"] = "\n".join(urls)
    if fail_substring is not None:
        env["CURL_FAIL_SUBSTRING"] = fail_substring

    source_count = (
        len([url for url in urls if url.strip()]) if urls is not None else _default_source_count()
    )
    # The harness has to outlast the budget the script may spend: every
    # source may use every attempt. A fixed timeout cut the script off
    # while it was still inside its own budget.
    per_attempt = timeout if timeout is not None else _script_default_timeout()
    harness_timeout = max(120, per_attempt * attempts * source_count + 60)
    try:
        process = subprocess.run(
            ["bash", str(SCRIPT), str(cache_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=harness_timeout,
            check=False,
        )
    finally:
        os.close(stall_fd)

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


def _script_default_timeout() -> int:
    """Returns the per-attempt budget the script pins by default."""
    match = re.search(r"CIFAR10_TIMEOUT:-(\d+)", SCRIPT.read_text())
    assert match, "the script does not pin a default timeout"
    return int(match.group(1))


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
    # 'curl -C -' is what turns the partial file into a resumed download: the
    # offset has to be the next byte, not a separate flag.
    assert any(pair == ("-C", "-") for pair in zip(calls[1].argv, calls[1].argv[1:]))


def test_a_resumed_download_completes_the_archive(tmp_path: Path) -> None:
    """A transfer that drops half way through is resumed to a valid archive.

    The stub serves the payload from the requested byte offset, so the second
    attempt of the same source completes the file and both pinned digests
    match. A resume that cannot succeed can only be asserted as a failure,
    which would leave the resume path untested.
    """
    process, calls = _run_script(tmp_path, ["https://example.test/partial"])

    assert process.returncode == 0, process.stdout + process.stderr
    # The first attempt delivered half of the archive and failed, the second
    # attempt resumed at that offset.
    assert len(calls) == 2
    assert calls[0].size == 0
    assert calls[1].size == len(PAYLOAD) // 2
    assert any(pair == ("-C", "-") for pair in zip(calls[1].argv, calls[1].argv[1:]))
    # The completed archive matches both pinned digests.
    assert process.stdout.count(f"{ARCHIVE_NAME}: OK") == 2


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


def test_the_attempt_count_is_honoured(tmp_path: Path) -> None:
    """The per-source attempt count comes from the environment.

    The default is only pinned by a regex over the script source, so the
    loop itself has to be observed: a source that keeps failing must be
    tried the configured number of times.
    """
    process, calls = _run_script(
        tmp_path,
        ["https://example.test/corrupt"],
        attempts=3,
    )

    assert process.returncode == 1
    assert len(calls) == 3


def test_one_attempt_is_enough_when_the_source_works(tmp_path: Path) -> None:
    """A working source must not be retried."""
    process, calls = _run_script(
        tmp_path,
        ["https://example.test/good"],
        attempts=1,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    assert len(calls) == 1


def test_the_partial_download_is_purged_at_every_attempt_count(tmp_path: Path) -> None:
    """The purge must use the configured attempt count, not a hard-coded two.

    The partial download of an exhausted source is dropped so the next source
    starts at byte 0. If the purge fired at a fixed count instead of the
    configured one, the second source could resume on the bytes of the first
    source and the archive would be spliced together.
    """
    for attempts in (1, 3):
        work = tmp_path / f"attempts_{attempts}"
        work.mkdir()
        process, calls = _run_script(
            work,
            ["https://example.test/corrupt", "https://example.test/good"],
            attempts=attempts,
        )

        assert process.returncode == 0, process.stdout + process.stderr
        good_calls = [call for call in calls if call.url == "https://example.test/good"]
        assert good_calls
        # The second source starts at byte 0, whatever the attempt count is.
        assert [call.size for call in good_calls] == [0]


def test_the_pytest_step_reads_the_dataset_from_the_cache_directory() -> None:
    """The tests must use the directory the download script filled.

    If the two paths drift apart, the dataset is downloaded again into a
    different directory: the exact failure this branch fixes, but silent.
    """
    for name in WORKFLOW_FILES:
        job = _backend_job(name)
        _, download = _step_by_name(job, "Download and verify CIFAR-10")
        _, pytest_step = _step_by_name(job, "Run tests (pytest)")
        assert CACHE_PATH in download["run"], f"{name} downloads elsewhere"
        assert (
            pytest_step["env"]["CNN_TEST_DATA_DIR"] == CACHE_PATH
        ), f"{name} reads the dataset from another directory"


def test_the_logged_source_is_verbatim_and_cannot_start_a_workflow_command(
    tmp_path: Path,
) -> None:
    """The source is written to the CI log as it is.

    GitHub Actions reads a line that starts with '::' as a workflow command.
    The log line starts with a fixed text, so the source can never begin a
    line, and the source itself stays readable instead of being rewritten.
    """
    source = "https://example.test/::error::injected"
    process, _ = _run_script(
        tmp_path,
        [source],
        fail_substring="example.test",
    )

    logged = [
        line for line in process.stdout.splitlines() if line.startswith("Downloading CIFAR-10 from")
    ]
    assert logged, "the source was not logged"
    for line in logged:
        assert (
            line == f"Downloading CIFAR-10 from {source}"
        ), f"the logged source was rewritten: {line}"
        assert not line.startswith("::"), f"the logged line starts a workflow command: {line}"


def test_a_rejected_digest_cannot_forge_a_workflow_annotation(tmp_path: Path) -> None:
    """A rejected digest is reported on a single line.

    The rejected value is written into the error annotation, so a value that
    carries a newline could add a second line that starts a workflow command.
    """
    process, _ = _run_script(
        tmp_path,
        ["https://example.test/archive.tar.gz"],
        md5="not-a-digest\n::error::injected",
    )

    assert process.returncode == 1, "the rejected digest was accepted"
    annotations = [line for line in process.stdout.splitlines() if line.startswith("::error::")]
    assert len(annotations) == 1, f"the rejection spans several lines: {process.stdout}"
    assert "not-a-digest" in annotations[0], f"the rejected value is missing: {annotations[0]}"
    for line in process.stdout.splitlines():
        assert not line.startswith("::error::injected"), f"a forged annotation was logged: {line}"


def test_the_failure_annotation_names_the_source_and_the_reason(
    tmp_path: Path,
) -> None:
    """A failed download reports which source failed and why.

    The annotation is the only thing a triage sees, so it has to name the
    source and the exit status of the transfer.
    """
    source = "https://example.test/archive.tar.gz"

    process, _ = _run_script(
        tmp_path,
        [source],
        fail_substring="example.test",
    )

    assert process.returncode == 1, "the failed download was accepted"
    annotations = [line for line in process.stdout.splitlines() if line.startswith("::error::")]
    assert len(annotations) == 1, f"the failure spans several lines: {process.stdout}"
    assert source in annotations[0], f"the source is missing: {annotations[0]}"
    assert "curl exit 22" in annotations[0], f"the reason is missing: {annotations[0]}"


def test_a_rejected_digest_cannot_carry_control_characters_into_the_log(
    tmp_path: Path,
) -> None:
    """A rejected digest is reported without control characters.

    The rejected value is written into the error annotation, so a value that
    carries a control character could rewrite the line in the CI log.
    """
    process, _ = _run_script(
        tmp_path,
        ["https://example.test/archive.tar.gz"],
        md5="not-a-digest\x1b[31m",
    )

    assert process.returncode == 1, "the rejected digest was accepted"
    for line in process.stdout.splitlines():
        assert "\x1b" not in line, f"a control character reached the log: {line!r}"


def test_the_workflow_pins_the_expected_digests() -> None:
    """The digests CI accepts are pinned in the workflow as well.

    The script reads the expected digests from the environment, so a workflow
    that sets them explicitly keeps the accepted archive from drifting with
    the script defaults. The values have to match the script defaults, or the
    workflow would verify something the script does not expect.
    """
    for name in WORKFLOW_FILES:
        job = _backend_job(name)
        _, download = _step_by_name(job, "Download and verify CIFAR-10")
        env = download.get("env", {})

        assert env.get("CIFAR10_MD5") == _script_default(
            "MD5", 32
        ), f"{name} does not pin the expected md5 to the script default"
        assert env.get("CIFAR10_SHA256") == _script_default(
            "SHA256", 64
        ), f"{name} does not pin the expected sha256 to the script default"


def test_the_cache_actions_are_pinned_to_a_commit_sha() -> None:
    """The cache actions must be pinned, and both must use the same commit."""
    for name in WORKFLOW_FILES:
        job = _backend_job(name)
        _, restore = _step_by_name(job, "Restore CIFAR-10 dataset cache")
        _, save = _step_by_name(job, "Save verified CIFAR-10 dataset cache")

        restore_ref = restore["uses"]
        save_ref = save["uses"]
        for ref in (restore_ref, save_ref):
            assert re.fullmatch(
                r"actions/cache/(restore|save)@[0-9a-f]{40}", ref
            ), f"{name} does not pin {ref} to a commit SHA"

        assert (
            restore_ref.split("@", 1)[1] == save_ref.split("@", 1)[1]
        ), f"{name} restores and saves with different commits"


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
    process, calls = _run_script(
        tmp_path,
        ["https://example.test/slow"],
        # The budget is much larger than the start-up cost of the stub, so
        # the stub is always running when the timeout signals it.
        timeout=5,
        attempts=1,
        stub=_STUB_CURL_SHELL,
    )

    assert process.returncode == 1
    assert "::error::CIFAR-10 archive could not be downloaded and verified." in (process.stdout)
    # The stalled transfer was aborted: the script reported its own error
    # instead of waiting for the transfer to finish. Its own budget is the
    # attempt timeout plus the kill grace (5 s + 30 s), and the harness
    # timeout stays the hard limit: a script that hangs is reported as a
    # timeout by the harness instead of failing an assertion on the clock.
    killed_path = tmp_path / "curl.killed.log"
    killed = killed_path.read_text() if killed_path.exists() else ""
    # The stub was signalled and never completed the transfer: the timeout
    # interrupted it instead of the transfer running to completion.
    assert killed.count("killed") >= 1
    # The attempt really ran: the stub was started before it was killed.
    assert len(calls) == 1


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
    default_sources = re.search(r'DEFAULT_URLS="(.*?)"', text, re.S)
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
    ):
        assert flag in argv, f"{flag} is missing from the curl call"
        assert value in argv, f"{flag} {value} is missing from the curl call"
    # An HTTP error has to fail the transfer, and the transfer has to resume
    # at the current size of the file.
    assert "-fL" in argv
    assert any(
        pair == ("-C", "-") for pair in zip(argv, argv[1:])
    ), f"the transfer does not resume at the current file size: {argv}"
    # An automatic retry of a resumed transfer can re-send bytes and splice
    # the file, so the script retries through its attempt loop instead.
    assert "--retry" not in argv
    assert "--retry-all-errors" not in argv


def test_the_transfer_only_follows_https_redirects(tmp_path: Path) -> None:
    """A redirect must not be able to move the download off HTTPS.

    Both pinned sources are HTTPS, and the archive is verified against the
    pinned digests, so a redirect cannot change the bytes. Restricting the
    protocol keeps the transfer from following a redirect into a scheme the
    curl build accepts by default.
    """
    process, calls = _run_script(tmp_path, ["https://example.test/good"])

    assert process.returncode == 0, process.stdout + process.stderr
    argv = calls[0].argv
    for flag in ("--proto", "--proto-redir"):
        assert flag in argv, f"{flag} is missing from the curl call: {argv}"
        value = argv[argv.index(flag) + 1]
        assert value == "=https", f"{flag} must allow HTTPS only, got {value}"


def test_the_attempt_is_bounded_and_kills_a_stubborn_child(tmp_path: Path) -> None:
    """The per-attempt budget has to be enforced by `timeout`.

    The budget is only real when the transfer is wrapped in `timeout`, and a
    child that ignores the TERM signal has to be killed after the grace
    period. Both are part of the contract of the script.
    """
    process, _calls = _run_script(
        tmp_path,
        ["https://example.test/good"],
        timeout=30,
        timeout_stub=_STUB_TIMEOUT,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    argv_log = tmp_path / "timeout.argv.log"
    assert argv_log.is_file(), "the script did not run curl through timeout"
    records = [json.loads(line) for line in argv_log.read_text().splitlines() if line.strip()]
    assert len(records) == 1
    argv = records[0]["argv"]
    assert "--kill-after=30" in argv, f"the attempt does not kill a stubborn child: {argv}"
    # The budget follows the grace period, so the configured value is the
    # argument that follows the flag.
    assert (
        argv[argv.index("--kill-after=30") + 1] == "30"
    ), f"the attempt does not use the configured budget: {argv}"
    # The wrapped command is curl, so the budget applies to the transfer.
    assert "curl" in argv


def test_the_default_attempt_budget_is_twenty_minutes(tmp_path: Path) -> None:
    """The pinned default budget is what a cold run has to live with."""
    process, _calls = _run_script(
        tmp_path,
        ["https://example.test/good"],
        timeout=None,
        timeout_stub=_STUB_TIMEOUT,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    argv_log = tmp_path / "timeout.argv.log"
    records = [json.loads(line) for line in argv_log.read_text().splitlines() if line.strip()]
    assert len(records) == 1
    argv = records[0]["argv"]
    assert "--kill-after=30" in argv
    assert argv[argv.index("--kill-after=30") + 1] == str(_script_default_timeout())


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


def test_the_script_reports_its_usage_without_a_cache_directory() -> None:
    """A call without a cache directory has to fail with a usage message.

    The workflow calls the script with one argument. A call without it must
    stop before anything is downloaded.
    """
    for arguments in ([], [""]):
        process = subprocess.run(
            ["bash", str(SCRIPT), *arguments],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert process.returncode != 0, "the script accepted a call without a cache directory"
        assert "usage: fetch_cifar10.sh <cache-dir>" in process.stderr


def test_a_blank_source_is_skipped(tmp_path: Path) -> None:
    """A blank line in the source list must not start a download.

    The source list is a newline separated environment variable, so it can
    carry an empty line. That line is not a source and must be skipped.
    """
    process, calls = _run_script(
        tmp_path,
        ["https://example.test/good", ""],
    )

    assert process.returncode == 0, process.stdout + process.stderr
    assert len(calls) == 1
    assert calls[0].url == "https://example.test/good"


def test_a_whitespace_only_source_is_skipped(tmp_path: Path) -> None:
    """A line that only holds spaces must not start a download.

    A blank line is skipped, but a line that only holds whitespace is just as
    empty: it is not a source, and spending an attempt on it would waste the
    budget of the sources that follow.
    """
    process, calls = _run_script(
        tmp_path,
        ["   ", "https://example.test/good"],
    )

    assert process.returncode == 0, process.stdout + process.stderr
    assert [call.url for call in calls] == ["https://example.test/good"]


def test_an_empty_source_list_falls_back_to_the_production_sources(
    tmp_path: Path,
) -> None:
    """An empty source list must not leave the script without a source.

    The source list is optional, so an empty value falls back to the pinned
    production sources. The first of them is the archive.org capture.
    """
    process, calls = _run_script(tmp_path, [])

    assert process.returncode == 0, process.stdout + process.stderr
    assert calls[0].url == MIRROR_URL


def test_a_blank_source_list_falls_back_to_the_production_sources(
    tmp_path: Path,
) -> None:
    """A source list that holds only whitespace is empty.

    A list of blanks names no source at all, so it falls back to the pinned
    production sources the way an empty list does. Failing instead would
    report a download error for a list that never named a source.
    """
    for index, urls in enumerate((["   "], ["\n", "  \n"])):
        # Every case gets its own work directory: a shared one keeps the
        # verified archive of the first run, so the second case would reuse
        # the cache and never call the mirror at all.
        work = tmp_path / f"case_{index}"
        work.mkdir()
        process, calls = _run_script(work, urls)

        assert process.returncode == 0, process.stdout + process.stderr
        assert calls, "the fallback sources were not tried"
        assert calls[0].url == MIRROR_URL


def test_an_http_error_page_is_not_kept_as_the_archive(tmp_path: Path) -> None:
    """An error page must never end up in the cache directory.

    A mirror that answers with an error body would leave a file that looks
    like the archive. The pinned digests reject it and the script drops it.
    """
    cache_dir = tmp_path / "cache"
    process, _calls = _run_script(
        tmp_path,
        ["https://example.test/errorpage"],
        cache_dir=cache_dir,
    )

    assert process.returncode == 1
    assert "::error::CIFAR-10 archive could not be downloaded and verified." in (process.stdout)
    assert not (cache_dir / ARCHIVE_NAME).exists()


def test_the_script_aborts_when_the_cache_directory_cannot_be_created(
    tmp_path: Path,
) -> None:
    """A cache directory that cannot be created has to stop the script.

    The script must not download into a directory it was unable to create.
    """
    blocked = tmp_path / "blocked"
    blocked.write_bytes(b"not a directory")
    process, calls = _run_script(
        tmp_path,
        ["https://example.test/good"],
        cache_dir=blocked,
    )

    assert process.returncode != 0
    assert calls == [], "the script downloaded into a cache directory it could not create"


def test_invalid_budget_overrides_are_rejected(tmp_path: Path) -> None:
    """A broken budget override has to be reported by the script itself.

    An attempt count or a timeout that is not a number would otherwise fail
    inside `seq` or `timeout`, with an error that does not name the
    override that caused it.
    """
    for name, value in (
        ("CIFAR10_ATTEMPTS", "abc"),
        ("CIFAR10_ATTEMPTS", "0"),
        ("CIFAR10_TIMEOUT", "abc"),
        ("CIFAR10_TIMEOUT", "0"),
    ):
        env = dict(os.environ)
        env[name] = value
        env["CIFAR10_URLS"] = "https://example.test/good"
        process = subprocess.run(
            ["bash", str(SCRIPT), str(tmp_path / "cache")],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert process.returncode == 1, f"{name}={value} was accepted"
        assert f"::error::{name} must be a positive integer" in (process.stdout)
        assert "Downloading CIFAR-10 from" not in (process.stdout)


def test_invalid_digest_overrides_are_rejected(tmp_path: Path) -> None:
    """A broken digest override has to be reported by the script itself.

    The digest is written into the checksum file that ``md5sum -c`` reads, so
    a value that is not a plain hex digest could name a second file and let
    that file satisfy the check.
    """
    for name, value in (
        ("CIFAR10_MD5", "not-a-digest"),
        ("CIFAR10_MD5", "c58f30108f718f92721af3b95e74349a  other-file"),
        ("CIFAR10_MD5", "C58F30108F718F92721AF3B95E74349A"),
        ("CIFAR10_SHA256", "not-a-digest"),
        (
            "CIFAR10_SHA256",
            "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce  other-file",
        ),
        (
            "CIFAR10_SHA256",
            "6D958BE074577803D12ECDEFD02955F39262C83C16FE9348329D7FE0B5C001CE",
        ),
    ):
        env = dict(os.environ)
        env[name] = value
        env["CIFAR10_URLS"] = "https://example.test/good"
        process = subprocess.run(
            ["bash", str(SCRIPT), str(tmp_path / "cache")],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert process.returncode == 1, f"{name}={value} was accepted"
        assert f"::error::{name} must be a" in (process.stdout)
        assert "Downloading CIFAR-10 from" not in (process.stdout)


def test_the_cache_save_step_cannot_fail_the_job_on_a_reservation_conflict() -> None:
    """A cache-reservation conflict must not turn a green job red.

    Both workflows save the verified archive under the same immutable key.
    When they run at the same time on a cold cache, the second save cannot
    reserve that key and the action fails the step. The cache is an
    optimisation, so the save step must not be allowed to fail the job.
    """
    for name in WORKFLOW_FILES:
        job = _backend_job(name)
        _, save = _step_by_name(job, "Save verified CIFAR-10 dataset cache")

        assert (
            save.get("continue-on-error") is True
        ), f"{name} can fail the job when the cache key is already reserved"


def test_the_download_is_capped_in_size(tmp_path: Path) -> None:
    """The transfer must not be able to fill the runner disk.

    The digest check runs after the transfer, so a broken or hostile mirror
    could serve an arbitrarily large body before it is rejected. The curl
    call therefore carries a size cap that stays above the real archive.
    """
    _, calls = _run_script(tmp_path, [MIRROR_URL])

    assert calls, "the script did not call curl"
    argv = calls[0].argv
    assert "--max-filesize" in argv, f"the download is not capped: {argv}"
    index = argv.index("--max-filesize")
    # The real archive is 170498071 bytes; the cap has to stay above it.
    assert int(argv[index + 1]) > 170498071, f"the cap is below the archive: {argv}"


def test_the_logged_source_cannot_carry_control_characters(tmp_path: Path) -> None:
    """A control character in a source must not reach the CI log.

    The source list is read line by line, so a source cannot contain a line
    break, but it can contain other control characters. They are stripped
    before the source is written to the log.
    """
    source = "https://example.test/archive\x01\x1b"
    process, _ = _run_script(
        tmp_path,
        [source],
        fail_substring="example.test",
    )

    logged = [
        line for line in process.stdout.splitlines() if line.startswith("Downloading CIFAR-10 from")
    ]
    assert logged, "the source was not logged"
    for line in logged:
        assert "\x01" not in line, f"a control character reached the log: {line!r}"
        assert "\x1b" not in line, f"a control character reached the log: {line!r}"


def test_the_failure_annotation_cannot_carry_control_characters(tmp_path: Path) -> None:
    """The failure annotation must not carry a control character into the log.

    The annotation names the source that failed, so a source with a control
    character would put it into the CI log as well.
    """
    source = "https://example.test/archive\x01\x1b"
    process, _ = _run_script(
        tmp_path,
        [source],
        fail_substring="example.test",
    )

    annotations = [line for line in process.stdout.splitlines() if "Last failure:" in line]
    assert annotations, "the failure was not annotated"
    for line in annotations:
        assert "\x01" not in line, f"a control character reached the log: {line!r}"
        assert "\x1b" not in line, f"a control character reached the log: {line!r}"


def test_every_failure_annotation_names_the_failure(tmp_path: Path) -> None:
    """Every error annotation has to name the source that failed.

    The script records the source and the curl exit status of every failed
    attempt, so an annotation without that detail would leave the CI log
    without the reason of the failure.
    """
    source = "https://example.test/archive"
    process, _ = _run_script(
        tmp_path,
        [source],
        fail_substring="example.test",
    )

    annotations = [line for line in process.stdout.splitlines() if line.startswith("::error::")]
    assert annotations, "the failure was not annotated"
    for line in annotations:
        assert "Last failure:" in line, f"the annotation does not name the failure: {line!r}"


def test_the_workflow_does_not_override_the_source_list_or_the_budget() -> None:
    """The workflow may only pin the digests.

    The script owns the source list, the attempt count and the download
    budget. A workflow that overrides them would silently change where the
    archive comes from and how long a download may take, so only the digests
    may be pinned.
    """
    for name in WORKFLOW_FILES:
        job = _backend_job(name)
        _, download = _step_by_name(job, "Download and verify CIFAR-10")
        env = download.get("env", {})

        assert set(env) <= {
            "CIFAR10_MD5",
            "CIFAR10_SHA256",
        }, f"{name} overrides more than the digests: {sorted(env)}"
        for variable in ("CIFAR10_URLS", "CIFAR10_ATTEMPTS", "CIFAR10_TIMEOUT"):
            assert variable not in env, f"{name} overrides {variable}"


def test_the_retired_cache_key_is_not_used_anywhere() -> None:
    """The poisoned cache key must not come back.

    The first version of the cache key stored a truncated archive under an
    immutable key, so the failure poisoned every later run. The live key
    carries the -v2 suffix; the retired one must not appear in a workflow.
    """
    retired = "cifar10-python-md5-c58f3010"
    for name in WORKFLOW_FILES:
        text = (WORKFLOW_DIR / name).read_text()

        assert not re.search(
            rf"{re.escape(retired)}(?!-v2)", text
        ), f"{name} still refers to the retired cache key"
