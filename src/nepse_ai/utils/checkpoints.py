"""Atomic PyTorch checkpoint writing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def save_local_checkpoint(
    payload: dict[str, object], destination: Path
) -> str:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()
