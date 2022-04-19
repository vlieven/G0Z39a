import os
from pathlib import Path

SRC_ROOT: Path = Path(os.path.dirname(os.path.abspath(__file__))).parent
PROJECT_ROOT: Path = SRC_ROOT.parent
