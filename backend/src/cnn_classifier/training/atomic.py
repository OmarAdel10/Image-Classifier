"""Atomic filesystem writes for training artifacts.

Writes go to a unique temporary sibling (``tempfile.mkstemp`` guarantees
an O_EXCL create, so concurrent runs and pre-planted symlinks cannot
collide on a predictable name) and are moved into place with
``os.replace``. A ``finally`` unlink removes the temporary file when the
writer or the replace step fails partway, so crashed runs never leave
``*.tmp`` residue next to the real artifact.
"""

import os
import tempfile
from collections.abc import Callable
from pathlib import Path


def atomic_write(path: str | Path, writer: Callable[[Path], None]) -> None:
    """Atomically write a file via a unique temp sibling + os.replace.

    Args:
        path: Destination file (parent directories are created).
        writer: Callable receiving the temp file path; it fills the file.

    Raises:
        OSError: Propagated from the filesystem. The temp file is removed
            and any previous destination content is left untouched.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=destination.parent, suffix=destination.suffix + ".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
