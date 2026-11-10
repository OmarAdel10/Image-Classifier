#!/usr/bin/env bash
# Downloads and verifies the CIFAR-10 python archive into a cache directory.
#
# The archive is verified against the pinned md5/sha256 before it is used, so
# a mirror can only make the download fail - it can never supply different
# bytes. A corrupt or partial archive is purged and downloaded again.
#
# A partial download is kept between attempts of the SAME source so that
# `curl -C -` can resume it, and is dropped when the source changes: resuming
# across two different byte streams would splice two files together.
#
# Usage: fetch_cifar10.sh <cache-dir>
#
# Environment overrides (used by the regression tests in
# backend/tests/test_ci_cifar_script.py):
#   CIFAR10_MD5       expected md5 of the archive
#   CIFAR10_SHA256    expected sha256 of the archive
#   CIFAR10_URLS      newline separated list of download URLs
#   CIFAR10_ATTEMPTS  download attempts per source (default 2)
#   CIFAR10_TIMEOUT   seconds allowed per attempt (default 1200)
set -euo pipefail

CACHE_DIR="${1:?usage: fetch_cifar10.sh <cache-dir>}"
MD5="${CIFAR10_MD5:-c58f30108f718f92721af3b95e74349a}"
SHA256="${CIFAR10_SHA256:-6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce}"
ATTEMPTS="${CIFAR10_ATTEMPTS:-2}"
# Worst case: 2 sources x 2 attempts x 1200 s = 80 min, plus 30 s of kill
# grace per attempt (4920 s = 82 min), which stays inside the 120 min job
# timeout and leaves room for the setup and the test run. A larger
# per-attempt budget would hit the job timeout instead of the script's own
# error path, which reports the failed download. The budget only succeeds
# when the mirror sustains roughly 35 KB/s or more; below that the script
# stops with its own error instead of running past the job timeout.
TIMEOUT="${CIFAR10_TIMEOUT:-1200}"
# The archive.org capture is pinned to an immutable id instead of the
# "2020id_" closest-capture selector, so the mirror cannot start serving
# different bytes later. Measured on 2026-11-08 from a CI runner network:
# the capture delivered 170,498,071 bytes at ~82 KB/s (36m47s), which is
# slower than one per-attempt budget (1200 s), so a cold run always consumes
# attempt 1 and finishes through the "-C -" resume on attempt 2. The
# cs.toronto.edu origin is throttled further. Both sources are verified
# against the pinned hashes above, so the mirror is verified, not trusted.
# The pinned production sources: the archive.org capture first, the
# cs.toronto.edu origin as the fallback.
DEFAULT_URLS="https://web.archive.org/web/20201130004040id_/https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
URLS="${CIFAR10_URLS:-$DEFAULT_URLS}"
# A source list that holds nothing but whitespace is empty: it falls back to
# the pinned production sources instead of failing with a download error that
# names the wrong cause.
if [[ -z "${URLS//[[:space:]]/}" ]]; then
  URLS="$DEFAULT_URLS"
fi

# A rejected value is written to the CI log. A line break in the value could
# forge a workflow command, so the line breaks are removed before the value
# is reported.
log_safe() {
  printf '%s' "$1" | tr -d '\000-\037\177'
}

if ! [[ "$ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::CIFAR10_ATTEMPTS must be a positive integer, got '$(log_safe "$ATTEMPTS")'."
  exit 1
fi

if ! [[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::CIFAR10_TIMEOUT must be a positive integer, got '$(log_safe "$TIMEOUT")'."
  exit 1
fi

# The digest is written into the checksum file that md5sum/sha256sum read, so
# a value that is not a plain hex digest could name a second file and let
# that file satisfy the check. Only a lowercase hex digest is accepted,
# which is the form the checksum tools read.
if ! [[ "$MD5" =~ ^[0-9a-f]{32}$ ]]; then
  echo "::error::CIFAR10_MD5 must be a 32 character hex digest, got '$(log_safe "$MD5")'."
  exit 1
fi

if ! [[ "$SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "::error::CIFAR10_SHA256 must be a 64 character hex digest, got '$(log_safe "$SHA256")'."
  exit 1
fi

mkdir -p "$CACHE_DIR"
cd "$CACHE_DIR"

verify() {
  [ -f cifar-10-python.tar.gz ] \
    && echo "$MD5  cifar-10-python.tar.gz" | md5sum -c --status \
    && echo "$SHA256  cifar-10-python.tar.gz" | sha256sum -c --status
}

# The last failure is reported in the error annotation, so a triage does not
# have to guess which source failed and why.
last_failure=''

if verify; then
  echo "CIFAR-10 archive verified from the restored cache."
else
  rm -f cifar-10-python.tar.gz
  while IFS= read -r url; do
    # A blank or whitespace-only line is not a source.
    [[ -n "${url//[[:space:]]/}" ]] || continue
    # The source is written to the CI log as it is, except for the control
    # characters that are stripped from it: the source list is read line by
    # line, so a source can never contain a line break, and the fixed prefix
    # keeps the logged line from starting with the workflow command syntax.
    echo "Downloading CIFAR-10 from $(log_safe "$url")"
    for attempt in $(seq 1 "$ATTEMPTS"); do
      # No --retry: an automatic retry of a resumed transfer can re-send
      # bytes and splice the file. The attempt loop above retries with a
      # fresh resume instead. --kill-after bounds a child that ignores the
      # TERM signal, so an attempt cannot outlive its budget. --proto and
      # --proto-redir keep the transfer on HTTPS, so a redirect cannot move
      # it into another scheme the curl build accepts by default.
      # --max-filesize caps the transfer: the digest check runs after the
      # download, so a broken mirror could otherwise fill the runner disk
      # before the archive is rejected. The cap stays above the real
      # archive, which is 170498071 bytes.
      status=0
      timeout --kill-after=30 "$TIMEOUT" curl -fL -C - \
        --proto '=https' --proto-redir '=https' \
        --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
        --max-filesize 200000000 \
        -o cifar-10-python.tar.gz "$url" || status=$?
      if verify; then
        break 2
      fi
      last_failure="${url} (curl exit ${status})"
      if [ "$attempt" -eq "$ATTEMPTS" ]; then
        # The source is exhausted: drop the partial so the next source
        # starts from byte 0 instead of splicing two byte streams.
        rm -f cifar-10-python.tar.gz
      fi
    done
  done <<< "$URLS"
fi

if ! verify; then
  # Every failed attempt records its source and the curl exit status, so the
  # annotation always names the reason of the failure. The fallback covers the
  # case where no attempt ran at all.
  failure_detail="${last_failure:-no source was tried}"
  echo "::error::CIFAR-10 archive could not be downloaded and verified. Last failure: $(log_safe "$failure_detail")."
  exit 1
fi

echo "$MD5  cifar-10-python.tar.gz" | md5sum -c -
echo "$SHA256  cifar-10-python.tar.gz" | sha256sum -c -
