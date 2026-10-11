"""让 tools/ 下的脚本找到研究引擎、实盘库与运维助手。被 import 时生效。"""
import os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
STUDIES = _HERE.parent / 'studies'
LIVE = Path(os.environ.get('LIVE_SRC_ROOT', _HERE.parents[1] / 'noise-momentum-live'))
for p in (STUDIES, LIVE / 'src', LIVE / 'ops'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
