import os
from pathlib import Path

# Files and Paths
SRC_PATH = Path(os.path.dirname(__file__))
PROJECT_ROOT_PATH = Path(os.path.abspath(SRC_PATH / ".."))
IMAGES_DIR = Path(os.path.abspath(PROJECT_ROOT_PATH / "images"))
PROCESSED_CARDS = Path(os.path.abspath(PROJECT_ROOT_PATH / "processed"))

# Analysis
DEFAULT_VERTEX_OFFSET = 5
DEFAULT_STROKE_SIZE = 2
