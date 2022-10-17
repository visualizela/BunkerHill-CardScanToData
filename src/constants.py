import os
from pathlib import Path

# Files and Paths
SRC_PATH = Path(os.path.dirname(__file__))
PROJECT_ROOT_PATH = Path(os.path.abspath(SRC_PATH / ".."))
IMAGES_DIR = Path(os.path.abspath(PROJECT_ROOT_PATH / "images"))
DATA_DIR = Path(os.path.abspath(PROJECT_ROOT_PATH / "data"))
BOXED_PATH = Path(os.path.abspath(DATA_DIR / "boxes"))
SLICED_CARDS = Path(os.path.abspath(DATA_DIR / "sliced_cards"))

# Analysis
DEFAULT_VERTEX_OFFSET = 5
DEFAULT_STROKE_SIZE = 2
