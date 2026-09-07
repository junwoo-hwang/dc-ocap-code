"""tests/ 에서 app.py 를 import 할 수 있게 저장소 루트를 경로에 넣는다."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
