"""Mirror console output to a line-buffered experiment log."""

from __future__ import annotations

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator, TextIO


class TeeStream:
    """Write every message to both the terminal and a log stream."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, message: str) -> int:
        terminal_result = self.terminal.write(message)
        self.log.write(message)
        self.flush()
        return terminal_result

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    @property
    def encoding(self) -> str | None:
        return self.terminal.encoding


@contextmanager
def mirrored_console(log_path: Path) -> Iterator[None]:
    """Display stdout/stderr live while recording both in one file."""
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        stdout = TeeStream(sys.stdout, log)
        stderr = TeeStream(sys.stderr, log)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield
