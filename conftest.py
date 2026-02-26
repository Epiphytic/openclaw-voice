"""Pytest configuration — ensure the local src/ takes priority over any
editable install from another workspace."""

from __future__ import annotations

import sys
from pathlib import Path

# Insert this repo's src/ at the front of sys.path so that imports resolve
# to the local package, not an editable install from a different working copy.
_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
