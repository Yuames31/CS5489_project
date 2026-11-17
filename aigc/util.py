"""Small helpers shared by the entry points."""
from __future__ import annotations

import sys


def fix_console():
    """Never crash on a non-UTF-8 (e.g. cp936) stdout when output is redirected."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def human_time(sec: float) -> str:
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
