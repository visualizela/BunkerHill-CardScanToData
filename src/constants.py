import os
from pathlib import Path

# Files and Paths
SRC_PATH = Path(os.path.dirname(__file__))
PROJECT_ROOT_PATH = Path(os.path.abspath(SRC_PATH / ".."))
IMAGES_DIR = Path(os.path.abspath(PROJECT_ROOT_PATH / "images"))

# Analysis
VERTEX_OFFSET = 5
