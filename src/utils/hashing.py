from __future__ import annotations

import hashlib

from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def sha256_file_set(
    entries: Iterable[tuple[str, str | Path]],
) -> str:
    """Hash labeled files without depending on absolute paths."""
    hasher = hashlib.sha256()

    for label, path in sorted(
        entries,
        key=lambda item: item[0],
    ):
        hasher.update(label.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(sha256_file(path).encode("ascii"))
        hasher.update(b"\0")

    return hasher.hexdigest()
