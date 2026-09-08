"""Make the repository importable from the tests, however pytest was started.

`from app.translate import ...` needs the repository root on sys.path, and
pytest puts the tests directory there, not its parent.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
