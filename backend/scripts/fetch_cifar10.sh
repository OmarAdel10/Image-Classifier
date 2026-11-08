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
# error path, which reports the failed download.
TIMEOUT="${CIFAR10_TIMEOUT:-1200}"
# The archive.org capture is pinned to an immutable id instead of the
# "2020id_" closest-capture selector, so the mirror cannot start serving
# different bytes later. Measured on 2026-11-08 from a CI runner network:
# the capture delivered 170,498,071 bytes at ~82 KB/s (36m47s), which is
# slower than one per-attempt budget (1200 s), so a cold run always consumes
# attempt 1 and finishes through the "-C -" resume on attempt 2. The
# cs.toronto.edu origin is throttled further. Both sources are verified
# against the pinned hashes above, so the mirror is verified, not trusted.
URLS="${CIFAR10_URLS:-https://web.archive.org/web/20201130004040id_/https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz}"

if ! [[ "$ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::CIFAR10_ATTEMPTS must be a positive integer, got '$ATTEMPTS'."
  exit 1
fi

if ! [[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::CIFAR10_TIMEOUT must be a positive integer, got '$TIMEOUT'."
  exit 1
fi

mkdir -p "$CACHE_DIR"
cd "$CACHE_DIR"

verify() {
  [ -f cifar-10-python.tar.gz ] \
    && echo "$MD5  cifar-10-python.tar.gz" | md5sum -c --status \
    && echo "$SHA256  cifar-10-python.tar.gz" | sha256sum -c --status
}

if verify; then
  echo "CIFAR-10 archive verified from the restored cache."
else
  rm -f cifar-10-python.tar.gz
  while IFS= read -r url; do
    # A blank or whitespace-only line is not a source.
    [[ -n "${url//[[:space:]]/}" ]] || continue
    echo "Downloading CIFAR-10 from $url"
    for attempt in $(seq 1 "$ATTEMPTS"); do
      # No --retry: an automatic retry of a resumed transfer can re-send
      # bytes and splice the file. The attempt loop above retries with a
      # fresh resume instead. --kill-after bounds a child that ignores the
      # TERM signal, so an attempt cannot outlive its budget.
      timeout --kill-after=30 "$TIMEOUT" curl -fL -C - \
        --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
        -o cifar-10-python.tar.gz "$url" || true
      if verify; then
        break 2
      fi
      if [ "$attempt" -eq "$ATTEMPTS" ]; then
        # The source is exhausted: drop the partial so the next source
        # starts from byte 0 instead of splicing two byte streams.
        rm -f cifar-10-python.tar.gz
      fi
    done
  done <<< "$URLS"
fi

if ! verify; then
  echo "::error::CIFAR-10 archive could not be downloaded and verified."
  exit 1
fi

echo "$MD5  cifar-10-python.tar.gz" | md5sum -c -
echo "$SHA256  cifar-10-python.tar.gz" | sha256sum -c -
