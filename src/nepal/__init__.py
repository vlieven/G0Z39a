import os
from pathlib import Path
from typing import Final

SRC_ROOT: Final[Path] = Path(os.path.dirname(os.path.abspath(__file__))).parent
PROJECT_ROOT: Final[Path] = SRC_ROOT.parent
