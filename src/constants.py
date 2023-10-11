import os
from pathlib import Path

# Files and Paths
SRC_PATH = Path(os.path.dirname(__file__))
PROJECT_ROOT_PATH = Path(os.path.abspath(SRC_PATH / ".."))
IMAGES_DIR = Path(os.path.abspath(PROJECT_ROOT_PATH / "images"))
DATA_DIR = Path(os.path.abspath(PROJECT_ROOT_PATH / "data"))
BOXED_PATH = Path(os.path.abspath(DATA_DIR / "boxes"))
SLICED_CARDS = Path(os.path.abspath(DATA_DIR / "sliced_cards"))

# BOX SELECTION DEFAULTS
DEFAULT_VERTEX_OFFSET = 9  # how large the selection area for vertex check should be
DEFAULT_STROKE_SIZE = 2  # stroke size for shape borders
DEFAULT_DISPLAY_STATE = 0  # 0 = show all boxes, 1 = show current only, 2 = hide all
DEFAULT_BLANK_BOX_COLOR = (0, 0, 0)  # Default color for blank box (used in display mode 1)
WINDOW_TITLE = "Main View"
DRAW_TEXT_BACKGROUND_PADDING = 4

# Drawing bounding box around mouse
MOUSE_BOX_BUFFER_SIZE = 45  # how many frames to track mouse movement
MOUSE_BOX_SWITCH_TO_CURSOR_SPEED = 25  # How fast the mouse can move before switching from box -> windows cursor
MOUSE_BOX_SWITCH_TO_BOX_SPEED = 2  # How fast the mouse can move before switching from windows cursor -> box
MOUSE_BOX_FLICKER_REDUCTION = 35  # Number of frames to wait before switching from cursor to bounding box
MOUSE_BOX_COLOR = (0, 0, 255)  # (blue, green, red)

# Vertex selection
VERTEX_WEIGHT_ON_CENTER = 0.15  # Percent weighting towards finding vertexes close to the center of bounding box
VERTEX_SIZE = 3  # How large to draw the vertex circle radius

# Image mode
DEFAULT_SHIFT_SIZE = 5  # Default number of pixels to shift image by
DEFAULT_BORDER_COLOR = (0, 0, 0)  # Default border color for image mode
