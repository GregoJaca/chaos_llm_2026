import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from chaos_llm.analyze import main  # noqa: E402


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", SRC)
    main()
