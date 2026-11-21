"""Content-based file identity.

Videos in a hoard get renamed and moved constantly. Keying the cache by path
would re-hash everything on every reorganise. Instead we key by *content*:

    file_id = sha256( size || first 1 MiB || last 1 MiB )

This is fast (no full read of multi-GB files) and collision-safe in practice for
distinguishing distinct files. A side benefit: two files with the same file_id
are byte-identical, so exact duplicates fall out for free before any perceptual
work.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1024 * 1024  # 1 MiB


def file_id(path: str | Path) -> str:
    """Return a stable content id for *path*.

    Hashes the file size plus the first and last 1 MiB. For files smaller than
    2 MiB the whole file is covered (the two chunks overlap, which is harmless).
    """
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())

    with p.open("rb") as f:
        head = f.read(_CHUNK)
        h.update(head)
        if size > _CHUNK:
            # Seek to the last chunk (may overlap head for small files; fine).
            f.seek(max(0, size - _CHUNK))
            h.update(f.read(_CHUNK))

    return h.hexdigest()
