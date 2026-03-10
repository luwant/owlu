from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LTCE_SRC = ROOT / "Label-gen" / "src"
if LTCE_SRC.exists() and str(LTCE_SRC) not in sys.path:
    sys.path.insert(0, str(LTCE_SRC))
