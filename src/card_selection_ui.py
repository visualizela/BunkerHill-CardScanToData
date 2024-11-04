import json
import os
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from typing import Tuple
import concurrent.futures

import cv2
import numpy as np
from tqdm import tqdm
from wand.exceptions import ImageError
from wand.image import Image

from src.constants import (
    BOXED_PATH,
    DEFAULT_BLANK_BOX_COLOR,
    DEFAULT_BORDER_COLOR,
    DEFAULT_SHIFT_SIZE,
    DEFAULT_STROKE_SIZE,
    DEFAULT_VERTEX_OFFSET,
    DRAW_TEXT_BACKGROUND_PADDING,
    IMAGES_DIR,
    MOUSE_BOX_BUFFER_SIZE,
    MOUSE_BOX_COLOR,
    MOUSE_BOX_FLICKER_REDUCTION,
    MOUSE_BOX_SWITCH_TO_BOX_SPEED,
    MOUSE_BOX_SWITCH_TO_CURSOR_SPEED,
    SLICED_CARDS,
    SRC_PATH,
    VERTEX_SIZE,
    VERTEX_WEIGHT_ON_CENTER,
    WINDOW_TITLE,
)


class BunkerHillCard:
    """
    Class responsible for BunkerHill card UI. This UI allows the user to select each region of a card and will generate
    a sub-image of each region for the analyzer to run.

    Raises:
        RuntimeError: If image cannot be found.
    """

    def __init__(self, path: str, no_find_vertex=False, save_dir=None) -> None:
        """
        Initiate class variables

        Args:
            path (str): path to image directory
        """
        # Last undo variables saved for redo command.
        self.last_undo = None

        # Image variables.
        self.unmodified_current = None
        self.image_names = []
        self.image_paths = []

        # current selection variables.
        self.selections = []
        self.selection_vertexes = []

        # Box variables.
        self.boxes = []
        self.curr_box = {}
        self.box_json = {}
        self.vertex_offset = DEFAULT_VERTEX_OFFSET
        self.stroke_size = DEFAULT_STROKE_SIZE

        self.bounding_box_templates = []

        # Cursor variables.
        self.mouse_locations = [[0, 0] for i in range(MOUSE_BOX_BUFFER_SIZE)]  # Track position of last n mouse locations
        self.frames_since_cursor_transition = 99  # used to track frames since last cursor swap
        self.cursor_is_custom = False  # used to track which mode the cursor is in

        # State flags.
        self.current_mode = 0         # Mode of the application: 0=box_mode, 1=text_mode, 2=image_mode
        self.display_state = 0        # How to draw the page 1=show boxes, 2=show only current box, 3=hide all
        self.last_button_q = False    # Flag to track if the last button press was 'q'
        self.last_button_ret = False  # Flag to track if the last button press was 'return'
        self.current_image = 0        # Index of current image to show
        self.show_preview_box = True  # If the preview box window should be displayed
        self.selected_box_index = -1  # Index of the current selected box

        # Text mode variables
        self.word = ""               # Current content of typed word for text mode
        self.cursor_index = 0        # Position of the cursor in the current word
        self.started_typing = False  # If user has started typing (used to delete default word)

        # Image mode variables
        self.shift_size = DEFAULT_SHIFT_SIZE  # How far to shift border on arrow click
        self.shift_image = None               # Displayed image for shift mode before it is saved
        self.image_mode_last_quit = False     # Confirm quit for image mode

        self.no_find_vertex = False


        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.image_names = [f for f in os.listdir(path) if f.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))]
        if len(self.image_names) > 0:
            for n in self.image_names:
                self.image_paths.append(os.path.join(path, n))

            # Setup initial image to display
            self.image_dir = path
            self.unmodified_current = cv2.imread(self.image_paths[0])
            self.shift_image = self.unmodified_current.copy()

            # Setup metadata
            metadata = {}
            metadata["path"] = os.path.abspath(path)
            self.box_json["metadata"] = metadata

            self.no_find_vertex = no_find_vertex
            if self.no_find_vertex:
                self.vertex_offset = 1
                self.stroke_size = 1

            # initiate empty selection box
            self._initiate_box()

            if os.path.exists(os.path.join(save_dir, "boxes.json")):
                self._load_outline(path=save_dir)
                if self.boxes:
                    self._update_vertex_for_image(self.boxes[self.selected_box_index], self.current_image)
        else:
            raise RuntimeError(f"Error: no images found at {IMAGES_DIR}")

    def _initiate_box(self) -> None:
        """
        Initiate an empty box json
        """

        self.curr_box = {
            "name": None,
            "top_left_bb": (0, 0),
            "top_right_bb": (0, 0),
            "bottom_left_bb": (0, 0),
            "bottom_right_bb": (0, 0),
            "stroke_size": self.stroke_size,
            "vertex_offset": self.vertex_offset,
            "top_left_ref_img_path": None,
            "bottom_right_ref_img_path": None,
            "color": (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)),
        }

        # Initiate an empty vertex for each image in the image path dir
        for file in self.image_names:
            self.curr_box[file] = {
                "top_left_vertex": (0, 0),
                "top_right_vertex": (0, 0),
                "bottom_left_vertex": (0, 0),
                "bottom_right_vertex": (0, 0),
            }

    def _draw_blank_over_box(self, box: dict, image: np.ndarray) -> np.ndarray:
        """
        draw a black box over previous selections

        Args:
            box (dict): box to blank out
            image (np.ndarray): image to draw blank on

        Returns:
            np.ndarray: image with blank box drawn
        """
        margin_size = box["vertex_offset"]
        top_left = box["top_left_bb"]
        bottom_right = box["bottom_right_bb"]

        cv2.rectangle(
            image,
            (top_left[0] + margin_size, top_left[1] + margin_size),
            (bottom_right[0] - margin_size, bottom_right[1] - margin_size),
            DEFAULT_BLANK_BOX_COLOR,
            -1,
        )
        return image

    # pylint: disable=too-many-locals
    def _draw_box(self, box: dict, image: np.ndarray) -> np.ndarray:
        """
        Draw a box on the given image. Returns an image with the box drawn.

        Args:
            box (dict): Box to draw
            image (np.ndarray): image to draw the box on

        Returns:
            np.ndarray: image with box drawn
        """

        vertex_offset = box["vertex_offset"]
        ss = box["stroke_size"]
        selected_stroke = 0

        c = box["color"]
        tlbb = box["top_left_bb"]
        trbb = box["top_right_bb"]
        blbb = box["bottom_left_bb"]
        brbb = box["bottom_right_bb"]

        curr_image = box[self.image_names[self.current_image]]
        tlv = curr_image["top_left_vertex"]
        trv = curr_image["top_right_vertex"]
        blv = curr_image["bottom_left_vertex"]
        brv = curr_image["bottom_right_vertex"]

        if self.boxes[self.selected_box_index] == box and self.display_state % 3 == 0:
            # make box lines a little thicker
            selected_stroke = 2

            # draw indicator lines

            # top middle
            cv2.line(
                image,
                (tlbb[0] + int(abs(tlbb[0] - trbb[0]) / 2), tlbb[1] + int(2 * vertex_offset / 3)),
                (trbb[0] - int(abs(tlbb[0] - trbb[0]) / 2), trbb[1] - int(2 * vertex_offset / 3)),
                c,
                ss + selected_stroke,
            )
            # bot middle
            cv2.line(
                image,
                (blbb[0] + int(abs(blbb[0] - brbb[0]) / 2), blbb[1] + int(2 * vertex_offset / 3)),
                (brbb[0] - int(abs(blbb[0] - brbb[0]) / 2), brbb[1] - int(2 * vertex_offset / 3)),
                c,
                ss + selected_stroke,
            )
            # left middle
            cv2.line(
                image,
                (tlbb[0] + int(2 * vertex_offset / 3), tlbb[1] + int(abs(tlbb[1] - brbb[1]) / 2)),
                (tlbb[0] - int(2 * vertex_offset / 3), blbb[1] - int(abs(tlbb[1] - brbb[1]) / 2)),
                c,
                ss + selected_stroke,
            )
            # right middle
            cv2.line(
                image,
                (trbb[0] + int(2 * vertex_offset / 3), trbb[1] + int(abs(tlbb[1] - brbb[1]) / 2)),
                (trbb[0] - int(2 * vertex_offset / 3), brbb[1] - int(abs(tlbb[1] - brbb[1]) / 2)),
                c,
                ss + selected_stroke,
            )

        # Draw vertex at 4 detected points
        cv2.circle(image, (tlv[0], tlv[1]), radius=3, color=c, thickness=2)
        cv2.circle(image, (trv[0], trv[1]), radius=3, color=c, thickness=2)
        cv2.circle(image, (blv[0], blv[1]), radius=3, color=c, thickness=2)
        cv2.circle(image, (brv[0], brv[1]), radius=3, color=c, thickness=2)

        # Draw lines around 4 points
        cv2.line(image, (tlbb[0] + vertex_offset, tlbb[1]), (trbb[0] - vertex_offset, trbb[1]), c, ss + selected_stroke)
        cv2.line(image, (tlbb[0], tlbb[1] + vertex_offset), (blbb[0], blbb[1] - vertex_offset), c, ss + selected_stroke)
        cv2.line(image, (blbb[0] + vertex_offset, blbb[1]), (brbb[0] - vertex_offset, brbb[1]), c, ss + selected_stroke)
        cv2.line(image, (brbb[0], brbb[1] - vertex_offset), (trbb[0], trbb[1] + vertex_offset), c, ss + selected_stroke)

        # Draw search zones around 4 points
        tlz_tl = (tlbb[0] - vertex_offset, tlbb[1] + vertex_offset)
        tlz_br = (tlbb[0] + vertex_offset, tlbb[1] - vertex_offset)
        cv2.rectangle(image, tlz_tl, tlz_br, c, ss)

        trz_tl = (trbb[0] - vertex_offset, trbb[1] + vertex_offset)
        trz_br = (trbb[0] + vertex_offset, trbb[1] - vertex_offset)
        cv2.rectangle(image, trz_tl, trz_br, c, ss)

        blz_tl = (blbb[0] - vertex_offset, blbb[1] + vertex_offset)
        blz_br = (blbb[0] + vertex_offset, blbb[1] - vertex_offset)
        cv2.rectangle(image, blz_tl, blz_br, c, ss)

        brz_tl = (brbb[0] - vertex_offset, brbb[1] + vertex_offset)
        brz_br = (brbb[0] + vertex_offset, brbb[1] - vertex_offset)
        cv2.rectangle(image, brz_tl, brz_br, c, ss)

        # cv2.putText(image, box["name"], (tlbb[0]+20, tlbb[1]+30), font, 1, c, ss, cv2.LINE_AA)
        if self.current_mode == 1 and box == self.boxes[self.selected_box_index]:
            invert = (255 - c[0], 255 - c[1], 255 - c[2])
            self._draw_box_text(image, box, tlbb, (10, 10), 1, ss, c, text_color_bg=invert)
        else:
            self._draw_box_text(image, box, tlbb, (10, 10), 1, ss, c)

        return image

    def _draw_selection(self, selection: Tuple[int,int], selection_vertex: Tuple[int,int], image: np.ndarray) -> np.ndarray:
        """
        Draws the current selection on the given image.

        Args:
            selection (Tuple[int,int]): x,y vector for template
            selection_vertex (Tuple[int,int]): x,y for particular
            image (np.ndarray): image to draw the selection on.

        Returns:
            np.ndarray: Image with the selection drawn on it.
        """
        vertex_offset = self.vertex_offset
        ss = self.stroke_size

        box_draw_tl = (selection[0] - vertex_offset, selection[1] + vertex_offset)
        box_draw_br = (selection[0] + vertex_offset, selection[1] - vertex_offset)
        cv2.rectangle(image, box_draw_tl, box_draw_br, self.curr_box["color"], ss)

        # draw vertex
        cv2.circle(
            image,
            (selection_vertex[0], selection_vertex[1]),
            radius=VERTEX_SIZE,
            color=self.curr_box["color"],
            thickness=2,
        )

        return image

    def _draw_box_text(
        self,
        image,
        box: dict,
        pos: list,
        offset: list,
        font_scale: int,
        font_thickness: int,
        text_color: tuple,
        font: int = cv2.FONT_HERSHEY_COMPLEX_SMALL,
        text_color_bg: tuple = None,
    ) -> None:
        x, y = pos
        offx, offy = offset
        pad = DRAW_TEXT_BACKGROUND_PADDING
        text = box["name"]

        # char_size, _ = cv2.getTextSize(text[0], font, font_scale, font_thickness)

        box_width = box["top_right_bb"][0] - box["top_left_bb"][0]
        # number_of_rows = int(np.ceil(text_w / box_width))
        # chars_per_row = int(max(np.floor(char_size[0]), 1))

        combined_height = 0
        curr_line = ""
        check_spot = 0
        while len(text) > 0:
            curr_line += text[check_spot]
            line_size, _ = cv2.getTextSize(curr_line, font, font_scale, font_thickness)
            line_w, line_h = line_size

            # as soon as our current line is too large for the box
            if line_w + pad + offx >= box_width or len(curr_line) == len(text):
                if len(curr_line) != len(text):
                    if len(curr_line) > 1:
                        curr_line = curr_line[:-1]
                    # update line_size for bounding box
                    line_size, _ = cv2.getTextSize(curr_line, font, font_scale, font_thickness)
                    line_w, line_h = line_size

                if text_color_bg is not None:
                    cv2.rectangle(
                        image,
                        (x + offx - pad, y + offy - pad + combined_height),
                        (x + offx + line_w + pad, y + line_h + offy + pad + combined_height),
                        text_color_bg,
                        -1,
                    )
                cv2.putText(
                    image,
                    curr_line,
                    (x + offx, y + line_h + font_scale - 1 + offy + combined_height),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                )
                check_spot = 0
                text = text[len(curr_line) :]
                combined_height += line_h + 2 * pad + 1
                curr_line = ""
            else:
                check_spot += 1

    def _draw_mouse_box(self, image: np.ndarray) -> np.ndarray:
        """
        Draw the bounding box around mouse on the given image.

        Args:
            image (np.ndarray): image to draw the mouse on
        """

        mouse = self.mouse_locations[-1]

        # Draw mouse bounding box in box mode
        if self.current_mode == 0:
            box_draw_tl = (mouse[0] - self.vertex_offset, mouse[1] + self.vertex_offset)
            box_draw_br = (mouse[0] + self.vertex_offset, mouse[1] - self.vertex_offset)
            cv2.rectangle(image, box_draw_tl, box_draw_br, MOUSE_BOX_COLOR, self.stroke_size)

            # draw vertex
            # detected_vertex = self._find_vertex((mouse[0], mouse[1]), self.vertex_offset)
            # cv2.circle(image, (detected_vertex[0], detected_vertex[1]), radius=3, color=(0, 0, 255), thickness=2)

        # Draw character if mouse in text mode
        elif self.current_mode == 1:
            cv2.putText(
                image,
                "[A]",
                (mouse[0] - 20, mouse[1] + 10),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                MOUSE_BOX_COLOR,
                self.stroke_size,
            )

        elif self.current_mode == 2:
            tl = 0.5
            # Point down
            cv2.arrowedLine(
                image, (mouse[0], mouse[1]), (mouse[0], mouse[1] + 20), MOUSE_BOX_COLOR, self.stroke_size, tipLength=tl
            )

            # Point left
            cv2.arrowedLine(
                image, (mouse[0], mouse[1]), (mouse[0] - 20, mouse[1]), MOUSE_BOX_COLOR, self.stroke_size, tipLength=tl
            )

            # Point right
            cv2.arrowedLine(
                image, (mouse[0], mouse[1]), (mouse[0] + 20, mouse[1]), MOUSE_BOX_COLOR, self.stroke_size, tipLength=tl
            )

            # Point up
            cv2.arrowedLine(
                image, (mouse[0], mouse[1]), (mouse[0], mouse[1] - 20), MOUSE_BOX_COLOR, self.stroke_size, tipLength=tl
            )

        return image

    def _draw_image(self) -> None:
        """
        Shows the census card image either with or without annotations depending on the `display_state` flag. Redrawing
        the annotations each frame makes ensures the drawn image correctly reflects the state of the drawn boxes
        """

        # if user is in image mode
        if self.current_mode == 2:
            to_show = self.shift_image.copy()
        # otherwise show main image
        else:
            to_show = self.unmodified_current.copy()

        # Drawing everything
        if self.display_state % 3 == 0:
            # Draw completed boxes
            for b in self.boxes:
                to_show = self._draw_box(b, to_show)

            # Draw current selection(s)
            for s,s_v in zip(self.selections, self.selection_vertexes):
                to_show = self._draw_selection(s, s_v, to_show)

        # Draw only current selection or most recent box
        if self.display_state % 3 == 1:
            # if user made a selection, draw that
            if len(self.selections) > 0:
                for s,s_v in zip(self.selections, self.selection_vertexes):
                    to_show = self._draw_selection(s, s_v, to_show)

                for b in self.boxes:
                    to_show = self._draw_blank_over_box(b, to_show)

            # else draw most recent box
            else:
                if len(self.boxes) > 0:
                    to_show = self._draw_box(self.boxes[self.selected_box_index], to_show)

                    for i, b in enumerate(self.boxes):
                        if self.boxes[i] != self.boxes[self.selected_box_index]:
                            to_show = self._draw_blank_over_box(b, to_show)

        # Draw bounding around mouse
        locations = self.mouse_locations

        if locations is not None and len(locations) > 0:
            # check if mouse has not moved fast recent frames
            x_sum, y_sum = 0, 0
            for loc in locations:
                x_sum += abs(loc[0] - locations[-1][0])
                y_sum += abs(loc[1] - locations[-1][1])

            x_avg = x_sum / len(locations)
            y_avg = y_sum / len(locations)
            mouse_speed = np.sqrt(pow(x_avg, 2) + pow(y_avg, 2))

            # Check if the mouse is already a box
            if self.cursor_is_custom:
                # check if we should switch to cursor
                if (
                    mouse_speed > MOUSE_BOX_SWITCH_TO_CURSOR_SPEED
                    and self.frames_since_cursor_transition > MOUSE_BOX_FLICKER_REDUCTION
                ):
                    self.frames_since_cursor_transition = 0
                    self.cursor_is_custom = False

                # keep cursor as a box
                else:
                    to_show = self._draw_mouse_box(to_show)

            # mouse is windows cursor
            else:
                # check if we should switch to box
                if (
                    mouse_speed < MOUSE_BOX_SWITCH_TO_BOX_SPEED
                    and self.frames_since_cursor_transition > MOUSE_BOX_FLICKER_REDUCTION
                ):
                    self.frames_since_cursor_transition = 0
                    self.cursor_is_custom = True

                    to_show = self._draw_mouse_box(to_show)

            self.frames_since_cursor_transition += 1

        cv2.imshow(WINDOW_TITLE, to_show)

    def _draw_box_window(self, box: dict, magnification: float = 2) -> None:
        """
        Draw the provided box in a separate window

        Args:
            image (any): image to draw it on
            box (dict): box to draw
            magnification (float): magnification multiple
        """

        if self.show_preview_box and box is not None:
            if self.current_mode != 2:
                to_show = self.unmodified_current.copy()
            else:
                to_show = self.shift_image.copy()

            image_name = self.image_names[self.current_image]
            top_left = box[image_name]["top_left_vertex"]
            bottom_right = box[image_name]["bottom_right_vertex"]
            top_left_x = max(top_left[0], 0)
            top_left_y = max(top_left[1], 0)

            bottom_right_x = bottom_right[0]
            bottom_right_y = bottom_right[1]

            cropped = to_show[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            # resized = cv2.resize(cropped, None, fx=magnification, fy=magnification, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Preview Box Window", cropped)

            # Hide the cursor from window title if user is typing
            if self.current_mode == 1:
                cv2.setWindowTitle("Preview Box Window", f"{self.word}")
            else:
                cv2.setWindowTitle("Preview Box Window", f'{box["name"]}')

        else:
            try:
                cv2.destroyWindow("Box view")
            except cv2.error:
                return

    def _create_selection(self, x: int, y: int) -> None:
        """
        Create selection around the current mouse location

        Args:
            x (int): x position of mouse (in pixels)
            y (int): y position of mouse (in pixels)
        """

        print(f"box{len(self.boxes)} selection: ({x}, {y})")
        anchor_image_path = self._create_anchor_image((x,y), self.vertex_offset, self.unmodified_current)
        self.selections.append((x, y, anchor_image_path))
        if self.no_find_vertex:
            vertex = (x,y)
        else:
            vertex = self._find_vertex((x, y), self.vertex_offset, self.unmodified_current, anchor_image_path)
        self.selection_vertexes.append(vertex)

    def _re_find_selection_vertexes(self):
        self.selection_vertexes = []
        for s in self.selections:
            x,y,anchor_image_path = s
            if self.no_find_vertex:
                vertex = (x,y)
            else:
                vertex = self._find_vertex((x, y), self.vertex_offset, self.unmodified_current, anchor_image_path)
            self.selection_vertexes.append(vertex)

    def _create_box(self) -> None:
        """
        Create a box using the current selections. Resets the `curr_box` and `selections` lists after box is created
        """

        # If a box was created in text mode, update the selected box's word to remove the cursor from its name
        if self.current_mode == 1:
            self.boxes[self.selected_box_index]["name"] = self.word

        # Setup box naming variables
        self.word = f"box{len(self.boxes)}"
        self.cursor_index = len(self.word)
        self.current_mode = 1
        self.started_typing = False
        self.selected_box_index = -1

        # append other two implicit corners
        # self.selections.append((self.selections[0][0], self.selections[1][1]))
        # self.selections.append((self.selections[1][0], self.selections[0][1]))

        # Find and sort the edges of the box
        # self.selections.sort(key=lambda i: i[0])
        # s0 = self.selections[0]
        # s1 = self.selections[1]
        # s2 = self.selections[2]
        # s3 = self.selections[3]

        # tlbb = s0 if s0[1] < s1[1] else s1
        # blbb = s1 if s0[1] < s1[1] else s0
        # trbb = s2 if s2[1] < s3[1] else s3
        # brbb = s3 if s2[1] < s3[1] else s2

        self.curr_box["top_left_bb"] = self.selections[0][:2]
        self.curr_box["top_left_ref_img_path"] = self.selections[0][2]

        self.curr_box["bottom_right_bb"] = self.selections[1][:2]
        self.curr_box["bottom_right_ref_img_path"] = self.selections[1][2]

        self.curr_box["bottom_left_bb"] = self.curr_box["top_left_bb"][0],self.curr_box["bottom_right_bb"][1]
        self.curr_box["top_right_bb"] = self.curr_box["bottom_right_bb"][0],self.curr_box["top_left_bb"][1]


        # Update vertex selection for all images
        # self._update_all_vertex(self.curr_box)
        self._update_vertex_for_image(self.curr_box, self.current_image)

        self.curr_box["name"] = self.word
        self.curr_box["vertex_offset"] = self.vertex_offset
        self.curr_box["stroke_size"] = self.stroke_size

        # Save the box and reset curr
        self.boxes.append(self.curr_box)
        self._initiate_box()
        self.selections = []
        self.selection_vertexes = []

    def _click_event(self, event: int, x: int, y: int, flags, params) -> None:
        """
        Draws a box around the selection when the user clicks the screen. After two clicks a box is added around top
        left and bottom right of of click.

        Args:
            event (int): cv2 event constant
            x (int): x location pixel value
            y (int): y location pixel value
            flags (Any): Unused, needed by cv2
            params (Any): Unused, needed by cv2
        """

        # Update list of recent mouse locations
        self.mouse_locations.append([x, y])
        self.mouse_locations.pop(0)

        # User clicks screen
        if event == cv2.EVENT_LBUTTONDOWN:
            # clicking resets quit flag
            self.last_button_q = False

            # save the selection
            self._create_selection(x, y)

            # If the user has selected a box
            if len(self.selections) == 2:
                self._create_box()

    def _update_vertex_for_image(self, box: dict, image_index: int, image_paths=None, image_names=None):
        if image_names:
            name = image_names[image_index]
        else:
            name = self.image_names[image_index]
        c = box[name]
        if image_paths:
            c_image = cv2.imread(image_paths[image_index])
        else:
            c_image = cv2.imread(self.image_paths[image_index])
        # recalculate best vertex location
        if self.no_find_vertex:
            c["top_left_vertex"] = box["top_left_bb"]
            c["bottom_right_vertex"] = box["bottom_right_bb"]
            c["top_right_vertex"] = c["top_left_vertex"][0], c["bottom_right_vertex"][1]
            c["bottom_left_vertex"] = c["bottom_right_vertex"][0], c["top_left_vertex"][1]
        else:
            c["top_left_vertex"] = self._find_vertex(box["top_left_bb"], box["vertex_offset"], original_image=c_image, reference_image=box["top_left_ref_img_path"])
            c["bottom_right_vertex"] = self._find_vertex(box["bottom_right_bb"], box["vertex_offset"], original_image=c_image, reference_image=box["bottom_right_ref_img_path"])
            c["top_right_vertex"] = c["top_left_vertex"][0], c["bottom_right_vertex"][1]
            c["bottom_left_vertex"] = c["bottom_right_vertex"][0], c["top_left_vertex"][1]

        return name, c  # return the name and c

    # def _update_all_vertex(self, box: dict) -> None:
    #     """
    #     Updates the location of the vertex position for each image using the boxes bounding box
    #     """
    #     for index, name in enumerate(tqdm(self.image_names)):
    #         self._update_vertex_for_image(box, index)


    def _update_all_vertex(self, box: dict) -> None:
        with ProcessPoolExecutor() as executor:
            num_iters = len(self.image_names)
            futures = [executor.submit(self._update_vertex_for_image, box, i, self.image_paths, self.image_names) for i in range(num_iters)]
            results = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=num_iters, desc="Updating Vertices"):
                results.append(future.result())
        for name, c in results:
            # update the main box object with the returned values
            box[name] = c

    def _update_image_vertex(self, image, image_name: str, box: dict) -> None:
        """
        Update (re-detect) the vertex for a given box on a specific image

        Args:
            image : image to update the vertex on
            image_name (str): name of the image (used to get the correct dict fields)
            box (dict): box to update the vertex on
        """
        c = box[image_name]

        # recalculate best vertex location
        if self.no_find_vertex:
            c["top_left_vertex"] = box["top_left_bb"]
            c["bottom_right_vertex"] = box["bottom_right_bb"]
            c["top_right_vertex"] = c["top_left_vertex"][0], c["bottom_right_vertex"][1]
            c["bottom_left_vertex"] = c["bottom_right_vertex"][0], c["top_left_vertex"][1]
        else:
            c["top_left_vertex"] = self._find_vertex(box["top_left_bb"], box["vertex_offset"], original_image=image, reference_image=box["top_left_ref_img_path"])
            c["bottom_right_vertex"] = self._find_vertex(box["bottom_right_bb"], box["vertex_offset"], original_image=image, reference_image=box["bottom_right_ref_img_path"])
            c["top_right_vertex"] = c["top_left_vertex"][0], c["bottom_right_vertex"][1]
            c["bottom_left_vertex"] = c["bottom_right_vertex"][0], c["top_left_vertex"][1]

    def _create_anchor_image(self, bounding_box: Tuple[int,int], vertex_offset: int, anchor_image: np.array):
        anchor_image_height, anchor_image_width = anchor_image.shape[:2]
        subimage = anchor_image[max(0, bounding_box[1]-vertex_offset):min(anchor_image_height, bounding_box[1]+vertex_offset), max(0, bounding_box[0]-vertex_offset):min(anchor_image_width, bounding_box[0]+vertex_offset)]
        filename = f"data/boxes/{uuid.uuid4()}.png"
        cv2.imwrite(filename, subimage)
        return filename


    def _find_vertex(self, bounding_box: Tuple[int,int], vertex_offset: int, original_image: np.array, reference_image: str) -> tuple:
        """
        Find the vertex given a bounding box and vertex offset

        Args:
            bounding_box (tuple): (x,y) center of bounding box
            vertex_offset (int): number of pixels off the center to check (square radius size of bounding box)
            image (optional): Image to check for vertex on, defaults to the current image if None given.

        Returns:
            tuple: (x,y) of vertex detected
        """

        # Convert the OpenCV image to Wand Image
        wand_image = Image.from_array(original_image)

        anchor_image = Image(filename=reference_image)

        # Define the region to search within
        search_region_top_left = (max(0, bounding_box[0] - vertex_offset*2), max(0, bounding_box[1] - vertex_offset*2))
        search_region_bottom_right = (min(wand_image.width, bounding_box[0] + vertex_offset*2), min(wand_image.height, bounding_box[1] + vertex_offset*2))

        # print(search_region_top_left, search_region_bottom_right)

        # Crop the larger image to the search region
        try:
            search_region = wand_image[search_region_top_left[0]:search_region_bottom_right[0], search_region_top_left[1]:search_region_bottom_right[1]]
        except IndexError:
            # If IndexError occurs, adjust the search region to fit within the image boundaries
            search_region_top_left = (max(0, min(search_region_top_left[0], wand_image.width - 12)), max(0, min(search_region_top_left[1], wand_image.height - 12)))
            search_region_bottom_right = (max(0, min(search_region_bottom_right[0], wand_image.width)), max(0, min(search_region_bottom_right[1], wand_image.height)))
            search_region = wand_image[search_region_top_left[0]:search_region_bottom_right[0], search_region_top_left[1]:search_region_bottom_right[1]]


        # Use Wand's similarity function to find the best match
        try:
            location, diff = search_region.similarity(anchor_image, threshold=0.0)
            vertex_x = location["left"] + search_region_top_left[0] + vertex_offset
            vertex_y = location["top"] + search_region_top_left[1] + vertex_offset
        except ImageError:
            vertex_x = max(0, min(bounding_box[0], wand_image.width))
            vertex_y = max(0, min(bounding_box[1], wand_image.height))

        # print(vertex_x, vertex_y)

        anchor_image.close()

        # Return the location of the best match, adjusted for the coordinates of the search region
        return (vertex_x, vertex_y)

    def _calculate_proximity_score(self, unweighted_list: list) -> list:
        """
        Calculate the proximity score given a tally list of how many black cubes are in the rows/cols

        Args:
            unweighted_list (list): row or column list

        Returns:
            list: row or column list that weights the values based on proximity to other black marks
        """
        weighted_score = [0 for i in range(len(unweighted_list))]

        # Weight so lines next to lines get much better scores
        for i in range(len(unweighted_list)):
            proximity_score = 0
            for j in range(len(unweighted_list)):
                if j != 0:
                    distance = abs(i - j)
                    proximity_score += unweighted_list[i] * 1 / pow(2, distance)
            weighted_score[i] = round(unweighted_list[i] * proximity_score, 2)

        # weight so lines closer to the center have 10% higher points -> 0% higher points at edge
        center_selection = len(weighted_score) / 2
        for i in range(len(weighted_score)):
            center_weight = 1.0 * -abs((i - center_selection) / center_selection) + 1
            weighted_score[i] = round(weighted_score[i] + center_weight * VERTEX_WEIGHT_ON_CENTER, 2)

        return weighted_score

    def _redo_last_undo(self) -> None:
        """
        Undoes the last redo action
        """
        if self.last_undo is not None:
            if self.last_undo_type == "selections":
                self.selections, self.selection_vertexes = self.last_undo
                self.last_undo = None
                print("Redoing most recent selection(s)")
            elif self.last_undo_type == "boxes":
                self.boxes.append(self.last_undo)
                self.last_undo = None
                print("Redoing most recent box")
            else:
                print("Error: unknown type in last_undo variable")
        else:
            print("Nothing to undo...")

    def _undo_last_action(self) -> None:
        """
        Removes the most recent action from its respective list. If the last action was defining a box vertex it will
        reset the box selection list. If the most recent action was creating a box it will pop the box off the top of
        the box list.
        """
        if len(self.selections) > 0:
            print("Removing current selection(s)")
            self.last_undo = (self.selections, self.selection_vertexes)
            self.last_undo_type = "selections"
            self.selections = []
            self.selection_vertexes = []
        else:
            if len(self.boxes) > 0:
                print("Removing selected box")
                self.last_undo = self.boxes[self.selected_box_index]
                self.last_undo_type = "boxes"
                self.boxes.remove(self.last_undo)
                if self.selected_box_index > 0:
                    self.selected_box_index -= 1
                else:
                    self.selected_box_index = -1
            else:
                print("Nothing to undo...")

    def _save_outline(self, path=None):
        """
        Save the selections json to be analyzed later
        """
        if len(self.boxes) < 1:
            print("You cannot save without selecting any boxes... Please make a selection")
            return

        self.box_json["metadata"]["total_boxes"] = len(self.boxes)
        self.box_json["boxes"] = self.boxes

        save_path = path or BOXED_PATH

        with open(os.path.join(save_path, "boxes.json"), "w", encoding="utf-8") as f:
            json.dump(self.box_json, f, ensure_ascii=False, indent=3)

        print("Selections have been saved")

    def _load_outline(self, path=None):
        """
        Load the selections json to continue from a previous session
        """
        load_path = path or BOXED_PATH

        with open(os.path.join(load_path, "boxes.json"), 'r', encoding='utf-8') as f:
            self.box_json = json.load(f)

        self.boxes = self.box_json.get("boxes", [])

        print("selections have been loaded")

    def _text_mode(self, key: int) -> str:
        """
        Text mode lets the user enter a name for the current selected box. This mode is entered automatically
        when a new box is created. Text mode can also be automatically entered when the user hits the appropriate
        hotkey.

        Args:
            key (int): int value of key pressed

        Returns:
            str: Word the user has typed with the curser character inserted, returns the word without the cursor when
                 user is finished
        """
        draw_text = True

        # print(f"word: \"{self.word}\" ({len(self.word)}) | cursor location: {self.cursor_index}")
        if key >= 32 and key <= 126:
            # Delete default word if user hasn't started typing yet
            if not self.started_typing:
                self.word = ""
                self.cursor_index = 0
                self.started_typing = True

            self.word = self.word[: self.cursor_index] + chr(key) + self.word[self.cursor_index :]
            self.cursor_index += 1
            # self.word = self.word[:self.cursor_index-1] + self.word[self.cursor_index:]

        # delete character ahead of cursor
        elif key == 3014656:
            if self.cursor_index < len(self.word):
                self.word = self.word[: self.cursor_index] + self.word[self.cursor_index + 1 :]

        # backspace next word (deletes all characters between cursor and white space)
        elif key == 127:
            if not self.started_typing:
                self.word = ""
                self.cursor_index = 0
                self.started_typing = True

            if self.cursor_index > 0:
                curr_char = self.word[self.cursor_index - 1]
                deleted = False
                deleted_non_whitespace = False
                while (curr_char != " " or deleted_non_whitespace is False) and self.cursor_index > 0:
                    self.word = self.word[: self.cursor_index - 1] + self.word[self.cursor_index :]
                    self.cursor_index -= 1
                    if self.cursor_index > 0:
                        curr_char = self.word[self.cursor_index - 1]
                    deleted = True
                    if curr_char != " ":
                        deleted_non_whitespace = True

                if not deleted:
                    self.word = self.word[: self.cursor_index - 1] + self.word[self.cursor_index :]
                    self.cursor_index -= 1

        # backspace character behind the cursor and shift cursor back
        elif key == 8:
            # Delete default word if user hasn't started typing yet
            if not self.started_typing:
                self.word = ""
                self.cursor_index = 0
                self.started_typing = True

            if self.cursor_index > 0:
                self.word = self.word[: self.cursor_index - 1] + self.word[self.cursor_index :]
                self.cursor_index -= 1

        # Finish typing box name
        elif key == 27 or key == 13:
            self.current_mode = 0
            self.cursor_index = 0
            return self.word

        # move cursor to end of line
        elif key == 7929856:
            self.started_typing = True
            self.cursor_index = len(self.word)

        # move to start of line
        elif key == 7864320 or key == 1:
            self.started_typing = True
            self.cursor_index = 0

        # move cursor right
        elif key == 63235 or key == 2555904:
            self.started_typing = True
            if self.cursor_index < len(self.word):
                self.cursor_index += 1

        # move cursor left
        elif key == 63234 or key == 2424832:
            self.started_typing = True
            if self.cursor_index > 0:
                self.cursor_index -= 1

            draw_text = False

        return self.word[: self.cursor_index] + "|" + self.word[self.cursor_index :] if draw_text else None

    def _box_mode(self, key: int) -> bool:
        """
        Box mode is the main mode of the program. This lets the user draw boxes around the census card sections.
        When the application is done running this function will return false.

        Args:
            key (int): int value of key press

        Returns:
            bool: True if application should keep running, false if program should stop
        """
        if key == 3014656:
            self.last_button_ret = False
            if self.last_button_q:
                print("quitting...")
                cv2.destroyAllWindows()
                return False
            else:
                print("Quit? Hit q again to quit program. Unsaved progress will be lost.")
                self.last_button_q = True
        elif key == ord("d"):
            self.last_button_q = False
            self.last_button_ret = False
            self.display_state += 1
            print(f"Display state: {self.display_state % 3}")
        elif key == ord("u"):
            self.last_button_q = False
            self.last_button_ret = False
            self._undo_last_action()
        elif key == ord("r"):
            self.last_button_q = False
            self.last_button_ret = False
            self._redo_last_undo()
        elif key == ord("h"):
            self.last_button_q = False
            self.last_button_ret = False
            self.help()
        elif key == ord("l"):
            self.last_button_q = False
            self.last_button_ret = False
            self.current_mode = 2
            print("Entering Image mode")
        elif key == ord("t"):
            self.last_button_q = False
            self.last_button_ret = False
            if len(self.boxes) > 0:
                self.word = self.boxes[self.selected_box_index]["name"]
                self.cursor_index = len(self.boxes[self.selected_box_index]["name"])
                self.current_mode = 1
                self.started_typing = False

        elif key == ord("+") or key == ord("="):
            self.last_button_q = False
            self.last_button_ret = False
            self.vertex_offset += 1
            print(f"Vertex size: {self.vertex_offset}")
            # If there is no selection made adjust selected box vertex offset
            if len(self.selections) == 0 and len(self.boxes) > 0:
                b = self.boxes[self.selected_box_index]
                b["vertex_offset"] += 1

                # self._update_all_vertex(b)

        # left arrow key
        elif key == 63234 or key == 2424832:
            self.last_button_q = False
            self.last_button_ret = False
            if self.current_image > 0:
                self.current_image -= 1
                self.unmodified_current = cv2.imread(self.image_paths[self.current_image])
                self.shift_image = self.unmodified_current
                self._re_find_selection_vertexes()
                if self.boxes:
                    self._update_vertex_for_image(self.boxes[self.selected_box_index], self.current_image)
                cv2.setWindowTitle(WINDOW_TITLE, f"{WINDOW_TITLE} ({self.current_image + 1}/{len(self.image_paths)})")

        # right arrow key
        elif key == 63235 or key == 2555904:
            self.last_button_q = False
            self.last_button_ret = False
            if self.current_image < len(self.image_paths) - 1:
                self.current_image += 1
                self.unmodified_current = cv2.imread(self.image_paths[self.current_image])
                self.shift_image = self.unmodified_current
                self._re_find_selection_vertexes()
                if self.boxes:
                    self._update_vertex_for_image(self.boxes[self.selected_box_index], self.current_image)
                cv2.setWindowTitle(WINDOW_TITLE, f"{WINDOW_TITLE} ({self.current_image + 1}/{len(self.image_paths)})")

        # Down key
        elif key == 63233 or key == 2621440:
            self.last_button_q = False
            self.last_button_ret = False
            if len(self.boxes) > 0:
                if self.selected_box_index == -1:
                    self.selected_box_index = len(self.boxes) - 1
                elif self.selected_box_index > 0:
                    self.selected_box_index -= 1
                else:
                    self.selected_box_index = len(self.boxes) - 1
            print(f"Box: {self.selected_box_index + 1}/{len(self.boxes)}")

        # Up key
        elif key == 63232 or key == 2490368:
            self.last_button_q = False
            self.last_button_ret = False
            if len(self.boxes) > 0:
                if self.selected_box_index == -1:
                    self.selected_box_index = len(self.boxes) - 1

                if self.selected_box_index < len(self.boxes) - 1:
                    self.selected_box_index += 1
                else:
                    self.selected_box_index = 0
            print(f"Box: {self.selected_box_index + 1}/{len(self.boxes)}")

        # save and exit (return)
        elif key == 13:
            self.last_button_q = False
            if self.last_button_ret:
                print("saving boxed zones...")
                # Separate file saves by day
                if self.save_dir is None:
                    now = date.today()
                    now_string = now.strftime("%m-%d-%Y")
                    base_dir = os.path.join(SLICED_CARDS, now_string)
                else:
                    base_dir = self.save_dir

                save_success = True

                self._save_outline(base_dir)

                for b in tqdm(self.boxes):
                    self._update_all_vertex(b)

                if not os.path.exists(base_dir):
                    os.mkdir(base_dir)
                else:
                    shutil.rmtree(base_dir)
                    os.mkdir(base_dir)

                # save files per box per image
                for i, path in enumerate(self.image_paths):
                    c_image = cv2.imread(path)

                    for box in self.boxes:
                        # Crop the images around the box selections
                        box_name = box["name"]
                        image_name = self.image_names[i]
                        top_left = box[image_name]["top_left_vertex"]
                        bottom_right = box[image_name]["bottom_right_vertex"]
                        top_left_x = max(top_left[0], 0)
                        top_left_y = max(top_left[1], 0)

                        bottom_right_x = bottom_right[0]
                        bottom_right_y = bottom_right[1]

                        cropped = c_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                        # make appropriate directories and image paths
                        image_dir = os.path.join(base_dir, box_name)
                        if not os.path.exists(image_dir):
                            os.mkdir(image_dir)
                        save_path = os.path.join(image_dir, image_name)
                        try:
                            cv2.imwrite(save_path, cropped)
                        except cv2.error:
                            pass
                        if not os.path.isfile(f"{save_path}"):
                            print(f"Error: failed to save image ({save_path})")
                            save_success = False

                self._save_outline(base_dir)

                if save_success:
                    print("Save successful!")
                else:
                    print("Save failed on one or more images. Please check file permissions")

                return False
            else:
                self.last_button_ret = True
                print("Split box selections and quit? Hit enter again to confirm.")

        elif key == ord("-") or key == ord("_"):
            self.last_button_ret = False
            self.last_button_q = False
            if self.vertex_offset > 1:
                self.vertex_offset -= 1
                print(f"Vertex size: {self.vertex_offset}")

            # If there is no selection made, adjust the most recent box vertex offset
            if (
                len(self.selections) == 0
                and len(self.boxes) > 0
                and self.boxes[self.selected_box_index]["vertex_offset"] > 1
            ):
                b = self.boxes[self.selected_box_index]
                b["vertex_offset"] -= 1

                # recalculate best vertex location
                # self._update_all_vertex(b)

        elif key == ord("}") or key == ord("]"):
            self.last_button_ret = False
            self.last_button_q = False
            self.stroke_size += 1
            print(f"Stroke size: {self.stroke_size}")

            # If there is no selection made adjust most recent box stroke size
            if len(self.selections) == 0 and len(self.boxes) > 0:
                self.boxes[self.selected_box_index]["stroke_size"] += 1

        elif key == ord("{") or key == ord("["):
            self.last_button_q = False
            self.last_button_ret = False
            if self.stroke_size > 1:
                self.stroke_size -= 1

            # If there is no selection made, adjust the most recent box stroke size
            if (
                len(self.selections) == 0
                and len(self.boxes) > 0
                and self.boxes[self.selected_box_index]["stroke_size"] > 1
            ):
                self.boxes[self.selected_box_index]["stroke_size"] -= 1

        elif key == ord("s"):
            self.last_button_q = False
            self.last_button_ret = False
            print("saving")
            if self.save_dir is None:
                now = date.today()
                now_string = now.strftime("%m-%d-%Y")
                base_dir = os.path.join(SLICED_CARDS, now_string)
            else:
                base_dir = self.save_dir
            self._save_outline(base_dir)
        elif key == ord("q"):
            self.last_button_ret = False
            if self.last_button_q:
                print("Quitting...")
                cv2.destroyAllWindows()
                return False
            else:
                self.last_button_q = True
                print("Quit? Hit q again to quit program. Unsaved progress will be lost.")
        return True

    def _image_mode(self, key: int) -> None:
        """
        Handle the user entering image mode. This mode lets the user make adjustments to the current image.
        The main purpose of this mode is to allow the user to manually line up images to be closer together.

        Args:
            key (int): key press int
        """
        # Right key
        if key != -1:
            print(f"key: {key}")
        if key == 63235 or key == 2555904:
            self.image_mode_last_quit = False
            self.shift_image = cv2.copyMakeBorder(
                self.shift_image, 0, 0, self.shift_size, 0, cv2.BORDER_CONSTANT, value=DEFAULT_BORDER_COLOR
            )
            for b in self.boxes:
                self._update_image_vertex(self.shift_image, self.image_names[self.current_image], b)
        # Down key
        if key == 63233 or key == 2621440:
            self.image_mode_last_quit = False
            self.shift_image = cv2.copyMakeBorder(
                self.shift_image, self.shift_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=DEFAULT_BORDER_COLOR
            )
            for b in self.boxes:
                self._update_image_vertex(self.shift_image, self.image_names[self.current_image], b)
        # Up key
        if key == 63232 or key == 2490368:
            self.image_mode_last_quit = False
            self.shift_image = self.shift_image[self.shift_size :, :]

            for b in self.boxes:
                self._update_image_vertex(self.shift_image, self.image_names[self.current_image], b)

        # Left key
        if key == 63234 or key == 2424832:
            self.image_mode_last_quit = False
            self.shift_image = self.shift_image[:, self.shift_size :]

            for b in self.boxes:
                self._update_image_vertex(self.shift_image, self.image_names[self.current_image], b)

        if key == ord("r"):
            self.image_mode_last_quit = False
            self.shift_image = self.unmodified_current.copy()
            for b in self.boxes:
                self._update_image_vertex(self.shift_image, self.image_names[self.current_image], b)

        if key == 13:
            self.image_mode_last_quit = False
            # save the edits to an image
            cv2.imwrite(self.image_paths[self.current_image], self.shift_image)

            # Rereading in the image insures what is displayed matches the file that was saved
            self.unmodified_current = cv2.imread(self.image_paths[self.current_image])
            self.current_mode = 0
            print("Saving edit and leaving image mode")
            # update the vertexes
            # for b in self.boxes:
            #     self._update_all_vertex(b)

        if key == ord("h"):
            self.image_mode_last_quit = False
            self.help()
        if key == ord("l") or key == 27:
            if self.image_mode_last_quit:
                self.shift_image = self.unmodified_current.copy()
                print("Entering box mode")
                self.current_mode = 0
                self.image_mode_last_quit = False
                for b in self.boxes:
                    self._update_image_vertex(self.shift_image, self.image_names[self.current_image], b)
            else:
                print("Quit image mode without saving?")
                self.image_mode_last_quit = True

    def help(self) -> None:
        """
        Print out the help menu to the terminal
        """
        file = open(os.path.join(SRC_PATH, "help_menu.txt"), "r", encoding="utf-8")
        print(file.read())

    def main_selection_loop(self) -> None:
        """
        Takes in the path to a census card and allows the user to select the edges of the census card to split the card
        into different sub-sections for individual analysis. When run the user will need to select a point to draw
        individual boxes around the census card.

        Args:
            path (str): path to census card scan
        """

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.destroyAllWindows()
        cv2.namedWindow(WINDOW_TITLE)
        cv2.setMouseCallback(WINDOW_TITLE, self._click_event)
        cv2.setWindowTitle(WINDOW_TITLE, f"{WINDOW_TITLE} ({self.current_image + 1}/{len(self.image_paths)})")
        while True:
            self._draw_image()

            if len(self.boxes) > 0:
                self._draw_box_window(self.boxes[self.selected_box_index], 1.8)
            else:
                self._draw_box_window(None, 2)

            # update mouse position
            self.mouse_locations.append(self.mouse_locations[-1])
            self.mouse_locations.pop(0)

            k = cv2.waitKeyEx(1)

            # ----- GLOBAL KEYS -------
            # Show a magnified version of current box
            if k == 22:
                self.show_preview_box = not self.show_preview_box
                continue

            # -------------------------

            # if user is in box mode. break out of main loop if _box_mode returns false
            if self.current_mode == 0:
                if not self._box_mode(k):
                    break

            # if user is typing box name capture inputs to type
            elif self.current_mode == 1:
                word = self._text_mode(k)
                if word:
                    self.boxes[self.selected_box_index]["name"] = word
                continue

            # If user is in image mode handle image mode inputs
            elif self.current_mode == 2:
                self._image_mode(k)
