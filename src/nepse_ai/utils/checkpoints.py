"""Atomic PyTorch checkpoint writing."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .hashing import sha256_file


def save_local_checkpoint(
    payload: dict[str, object], destination: Path
) -> str:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary = Path(temporary_file.name)

    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(destination)
