import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

exec((SRC / "shiftproof" / "dashboard.py").read_text(encoding="utf-8"))