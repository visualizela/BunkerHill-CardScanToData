import cv2
import json
import numpy as np
import os
import win32api

from constants import (DATA_DIR, DEFAULT_VERTEX_OFFSET, DEFAULT_STROKE_SIZE, IMAGES_DIR, BOXED_PATH, SLICED_CARDS,
                       MOUSE_BOX_BUFFER_SIZE, MOUSE_BOX_SWITCH_TO_BOX_SPEED, MOUSE_BOX_SWITCH_TO_CURSOR_SPEED,
                       MOUSE_BOX_FLICKER_REDUCTION, MOUSE_BOX_COLOR, DEFAULT_BLANK_BOX_COLOR, VERTEX_WEIGHT_ON_CENTER,
                       VERTEX_SIZE, SRC_PATH)


class BunkerHillCard:
    image_names = []
    image_paths = []
    current_image = 0
    unmodified_current = None
    display_state = 0
    boxes = []
    last_undo = None
    curr_box = {}
    selections = []
    selection_vertexes = []
    vertex_offset = DEFAULT_VERTEX_OFFSET
    stroke_size = DEFAULT_STROKE_SIZE
    box_json = {}
    mouse_locations = [[0, 0] for i in range(MOUSE_BOX_BUFFER_SIZE)]
    frames_since_cursor_transition = 99
    cursor_is_box = False
    last_button_q = False

    def __init__(self, path: str) -> None:
        """
        Initiate class variables

        Args:
            path (str): path to image directory
        """
        self.image_names = os.listdir(path)
        if len(self.image_names) > 0:
            for n in self.image_names:
                self.image_paths.append(os.path.join(path, n))

            # Setup initial image to display
            self.image_dir = path
            self.unmodified_current = cv2.imread(self.image_paths[0])

            # Setup metadata
            metadata = {}
            metadata["path"] = os.path.abspath(path)
            self.box_json["metadata"] = metadata

            print(self.box_json)

            # initiate empty selection box
            self._initiate_box()
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
            "color": (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        }

        # Initiate an empty vertex for each image in the image path dir
        for f in self.image_names:
            self.curr_box[f] = {
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
        tl = box["top_left_bb"]
        br = box["bottom_right_bb"]

        cv2.rectangle(image, (tl[0]+margin_size, tl[1]+margin_size), (br[0]-margin_size, br[1]-margin_size),
                      DEFAULT_BLANK_BOX_COLOR, -1)
        return image

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

        # Draw vertex at 4 detected points
        cv2.circle(image, (tlv[0], tlv[1]), radius=3, color=c, thickness=2)
        cv2.circle(image, (trv[0], trv[1]), radius=3, color=c, thickness=2)
        cv2.circle(image, (blv[0], blv[1]), radius=3, color=c, thickness=2)
        cv2.circle(image, (brv[0], brv[1]), radius=3, color=c, thickness=2)

        # Draw lines around 4 points
        cv2.line(image, (tlbb[0]+vertex_offset, tlbb[1]), (trbb[0]-vertex_offset, trbb[1]), c, ss)
        cv2.line(image, (tlbb[0], tlbb[1]+vertex_offset), (blbb[0], blbb[1]-vertex_offset), c, ss)
        cv2.line(image, (blbb[0]+vertex_offset, blbb[1]), (brbb[0]-vertex_offset, brbb[1]), c, ss)
        cv2.line(image, (brbb[0], brbb[1]-vertex_offset), (trbb[0], trbb[1]+vertex_offset), c, ss)

        # Draw search zones around 4 points
        tlz_tl = (tlbb[0]-vertex_offset, tlbb[1]+vertex_offset)
        tlz_br = (tlbb[0]+vertex_offset, tlbb[1]-vertex_offset)
        cv2.rectangle(image, tlz_tl, tlz_br, c, ss)

        trz_tl = (trbb[0]-vertex_offset, trbb[1]+vertex_offset)
        trz_br = (trbb[0]+vertex_offset, trbb[1]-vertex_offset)
        cv2.rectangle(image, trz_tl, trz_br, c, ss)

        blz_tl = (blbb[0]-vertex_offset, blbb[1]+vertex_offset)
        blz_br = (blbb[0]+vertex_offset, blbb[1]-vertex_offset)
        cv2.rectangle(image, blz_tl, blz_br, c, ss)

        brz_tl = (brbb[0]-vertex_offset, brbb[1]+vertex_offset)
        brz_br = (brbb[0]+vertex_offset, brbb[1]-vertex_offset)
        cv2.rectangle(image, brz_tl, brz_br, c, ss)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, box["name"], (tlbb[0]+20, tlbb[1]+30), font, 1, c, ss, cv2.LINE_AA)

        return image

    def _draw_selection(self, selection: list, image: np.ndarray) -> np.ndarray:
        """
        Draws the current selection on the given image.

        Args:
            selection (list): list of current selections (points the user has clicked)
            image (np.ndarray): image to draw the selection on.

        Returns:
            np.ndarray: Image with the selection drawn on it.
        """
        vertex_offset = self.vertex_offset
        ss = self.stroke_size

        box_draw_tl = (selection[0]-vertex_offset, selection[1]+vertex_offset)
        box_draw_br = (selection[0]+vertex_offset, selection[1]-vertex_offset)
        cv2.rectangle(image, box_draw_tl, box_draw_br, self.curr_box["color"], ss)

        # draw vertex
        selection_vertex = self._find_vertex(selection, self.vertex_offset)
        cv2.circle(image,
                   (selection_vertex[0], selection_vertex[1]),
                   radius=VERTEX_SIZE,
                   color=self.curr_box["color"],
                   thickness=2)

        return image

    def _draw_mouse_box(self, image: np.ndarray) -> np.ndarray:
        """
        Draw the bounding box around mouse on the given image.

        Args:
            image (np.ndarray): image to draw the mouse on
        """
        win32api.SetCursor(None)
        mouse = self.mouse_locations[-1]
        box_draw_tl = (mouse[0]-self.vertex_offset, mouse[1]+self.vertex_offset)
        box_draw_br = (mouse[0]+self.vertex_offset, mouse[1]-self.vertex_offset)
        cv2.rectangle(image, box_draw_tl, box_draw_br, MOUSE_BOX_COLOR, self.stroke_size)

        # draw vertex
        detected_vertex = self._find_vertex((mouse[0], mouse[1]), self.vertex_offset)
        cv2.circle(image, (detected_vertex[0], detected_vertex[1]), radius=3, color=(0, 0, 255), thickness=2)
        return image

    def _draw_image(self) -> None:
        """
        Shows the census card image either with or without annotations depending on the `display_state` flag. Redrawing
        the annotations each frame makes ensures the drawn image correctly reflects the state of the drawn boxes
        """
        to_show = self.unmodified_current.copy()

        if self.cursor_is_box:
            win32api.SetCursor(None)

        # Drawing everything
        if self.display_state % 3 == 0:

            # Draw completed boxes
            for b in self.boxes:
                to_show = self._draw_box(b, to_show)

            # Draw current selection(s)
            for s in self.selections:
                to_show = self._draw_selection(s, to_show)

        # Draw only current selection or most recent box
        if self.display_state % 3 == 1:

            # if user made a selection, draw that
            if len(self.selections) > 0:
                for s in self.selections:
                    to_show = self._draw_selection(s, to_show)

                for b in self.boxes:
                    to_show = self._draw_blank_over_box(b, to_show)

            # else draw most recent box
            else:
                if len(self.boxes) > 0:
                    to_show = self._draw_box(self.boxes[-1], to_show)

                    for i in range(len(self.boxes) - 1):
                        b = self.boxes[i]
                        to_show = self._draw_blank_over_box(b, to_show)

        # Draw bounding around mouse
        locations = self.mouse_locations

        if locations is not None and len(locations) > 0:
            # check if mouse has not moved fast recent frames
            x_sum, y_sum = 0, 0
            for loc in locations:
                x_sum += abs(loc[0]-locations[-1][0])
                y_sum += abs(loc[1]-locations[-1][1])

            x_avg = x_sum/len(locations)
            y_avg = y_sum/len(locations)
            mouse_speed = np.sqrt(pow(x_avg, 2) + pow(y_avg, 2))

            # Check if the mouse is already a box
            if self.cursor_is_box:

                # check if we should switch to cursor
                if (mouse_speed > MOUSE_BOX_SWITCH_TO_CURSOR_SPEED
                   and self.frames_since_cursor_transition > MOUSE_BOX_FLICKER_REDUCTION):
                    win32api.SetCursor(None)
                    self.frames_since_cursor_transition = 0
                    self.cursor_is_box = False

                # keep cursor as a box
                else:
                    to_show = self._draw_mouse_box(to_show)

            # mouse is windows cursor
            else:
                # check if we should switch to box
                if (mouse_speed < MOUSE_BOX_SWITCH_TO_BOX_SPEED
                   and self.frames_since_cursor_transition > MOUSE_BOX_FLICKER_REDUCTION):

                    self.frames_since_cursor_transition = 0
                    self.cursor_is_box = True

                    to_show = self._draw_mouse_box(to_show)

            self.frames_since_cursor_transition += 1

        cv2.imshow('image', to_show)

    def _create_selection(self, x: int, y: int) -> None:
        """
        Create selection around the current mouse location

        Args:
            x (int): x position of mouse (in pixels)
            y (int): y position of mouse (in pixels)
        """

        self.selections.append((x, y))
        print(f"box{len(self.boxes)} selection: ({x}, {y})")

        vertex = self._find_vertex((x, y), self.vertex_offset)
        self.selection_vertexes.append(vertex)

    def _create_box(self) -> None:
        """
        Create a box using the current selections. Resets the `curr_box` and `selections` lists after box is created
        """
        # append other two implicit corners
        self.selections.append((self.selections[0][0], self.selections[1][1]))
        self.selections.append((self.selections[1][0], self.selections[0][1]))

        # Find and sort the edges of the box
        self.selections.sort(key=lambda i: i[0])
        s0 = self.selections[0]
        s1 = self.selections[1]
        s2 = self.selections[2]
        s3 = self.selections[3]

        tlbb = s0 if s0[1] < s1[1] else s1
        blbb = s1 if s0[1] < s1[1] else s0
        trbb = s2 if s2[1] < s3[1] else s3
        brbb = s3 if s2[1] < s3[1] else s2

        self.curr_box["top_left_bb"] = tlbb
        self.curr_box["bottom_left_bb"] = blbb
        self.curr_box["top_right_bb"] = trbb
        self.curr_box["bottom_right_bb"] = brbb

        # Update vertex selection for all images
        self._update_all_vertex(self.curr_box)

        self.curr_box["name"] = f"box{len(self.boxes)}"
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
            flags (Any):
            params (Any):
        """

        if self.cursor_is_box:
            win32api.SetCursor(None)
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

    def _update_all_vertex(self, box: dict):
        """
        Updates the location of the vertex position for each image using the boxes bounding box
        """
        for index, name in enumerate(self.image_names):
            c = box[name]
            c_image = cv2.imread(self.image_paths[index])
            # recalculate best vertex location
            c["top_left_vertex"] = self._find_vertex(box["top_left_bb"], box["vertex_offset"], image=c_image)
            print(c["top_left_vertex"])
            c["top_right_vertex"] = self._find_vertex(box["top_right_bb"], box["vertex_offset"], image=c_image)
            c["bottom_left_vertex"] = self._find_vertex(box["bottom_left_bb"], box["vertex_offset"], image=c_image)
            c["bottom_right_vertex"] = self._find_vertex(box["bottom_right_bb"], box["vertex_offset"], image=c_image)

    def _find_vertex(self, bounding_box: tuple, vertex_offset: int, image=None) -> tuple:
        if image is None:
            image = self.unmodified_current
        top_left = (bounding_box[0] - vertex_offset, bounding_box[1] - vertex_offset)
        bottom_right = (bounding_box[0] + vertex_offset, bounding_box[1] + vertex_offset)
        tx = top_left[0] if top_left[0] >= 0 else 0
        ty = top_left[1] if top_left[1] >= 0 else 0
        bx = bottom_right[0] if bottom_right[0] < image.shape[1] else image.shape[1] - 1
        by = bottom_right[1] if bottom_right[1] < image.shape[0] else image.shape[0] - 1

        cropped = image[ty:by, tx:bx]
        width = cropped.shape[1]
        height = cropped.shape[0]

        processed = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        a, processed = cv2.threshold(processed, 127, 255, cv2.THRESH_BINARY)

        col_score = [0 for i in range(width)]
        row_score = [0 for i in range(height)]

        # Sum lines in rows and cols
        for col in range(width):
            for row in range(height):
                row_score[row] += 1 if processed[row][col] == 0 else 0

        for row in range(height):
            for col in range(width):
                col_score[col] += 1 if processed[row][col] == 0 else 0

        weighted_col_score = self._calculate_proximity_score(col_score)
        weighted_row_score = self._calculate_proximity_score(row_score)

        best_y = weighted_row_score.index(max(weighted_row_score))
        best_x = weighted_col_score.index(max(weighted_col_score))

        return (best_x+bounding_box[0]-vertex_offset, best_y+bounding_box[1]-vertex_offset)

    def _swap_displayed_card(direction: int) -> None:
        """
        Swap the currently displayed card with either the next or previous card in the images directory

        Args:
            direction (int): direction to swap. 1 = previous, 2 = next
        """
        # TODO: finish function
        if direction < 1 or direction > 2:
            raise ValueError(f"Invalid swap card direction given: {direction} (must be either 1 or 2)")

    def _calculate_proximity_score(self, unweighted_list: list) -> list:
        weighted_score = [0 for i in range(len(unweighted_list))]

        # Weight so lines next to lines get much better scores
        for i in range(len(unweighted_list)):
            proximity_score = 0
            for j in range(len(unweighted_list)):
                if j != 0:
                    distance = abs(i-j)
                    proximity_score += unweighted_list[i]*1/pow(2, distance)
            weighted_score[i] = round(unweighted_list[i] * proximity_score, 2)

        # weight so lines closer to the center have 10% higher points -> 0% higher points at edge
        center_selection = len(weighted_score)/2
        for i in range(len(weighted_score)):
            center_weight = 1.0 * -abs((i-center_selection)/center_selection) + 1
            weighted_score[i] = round(weighted_score[i] + center_weight * VERTEX_WEIGHT_ON_CENTER, 2)

        return weighted_score

    def _redo_last_undo(self) -> None:
        """
        Undoes the last redo action
        """
        if self.last_undo is not None:
            if type(self.last_undo) == list:
                self.selections = self.last_undo
                self.last_undo = None
                print("Redoing most recent selection(s)")
            elif type(self.last_undo) == dict:
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
            self.last_undo = self.selections
            self.selections = []
        else:
            if len(self.boxes) > 0:
                print("Removing most recent box")
                self.last_undo = self.boxes[-1]
                self.boxes.pop()
            else:
                print("Nothing to undo...")

    def _save_outline(self):
        """
        Save the selections json to be analyzed later
        """
        if len(self.boxes) < 1:
            print("You cannot save without selecting any boxes... Please make a selection")
            return

        self.box_json["metadata"]["total_boxes"] = len(self.boxes)
        self.box_json["boxes"] = self.boxes

        with open(os.path.join(BOXED_PATH, self.box_json["metadata"]["name"]), 'w', encoding='utf-8') as f:
            json.dump(self.box_json, f, ensure_ascii=False, indent=3)

        print("selections have been saved")

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
        # displaying the image
        # cv2.imshow('image', self.marked)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.namedWindow("image")
        cv2.setMouseCallback('image', self._click_event)
        win32api.SetCursor(None)
        while True:
            self._draw_image()
            # update mouse position
            self.mouse_locations.append(self.mouse_locations[-1])
            self.mouse_locations.pop(0)
            if self.cursor_is_box:
                win32api.SetCursor(None)

            k = cv2.waitKeyEx(1)
            if k == 3014656:
                if self.last_button_q:
                    print("quitting...")
                    cv2.destroyAllWindows()
                    break
                else:
                    print("Quit? Hit q again to quit program. Unsaved progress will be lost.")
                    self.last_button_q = True
            elif k == ord("d"):
                self.last_button_q = False
                self.display_state += 1
                print(f"Display state: {self.display_state % 3}")
            elif k == ord("u"):
                self.last_button_q = False
                self._undo_last_action()
            elif k == ord("r"):
                self.last_button_q = False
                self._redo_last_undo()
            elif k == ord("h"):
                self.last_button_q = False
                self.help()
            elif k == ord("+") or k == ord("="):
                self.last_button_q = False
                self.vertex_offset += 1
                print(f"Vertex size: {self.vertex_offset}")
                # If there is no selection made adjust most recent box vertex offset
                if len(self.selections) == 0 and len(self.boxes) > 0:
                    b = self.boxes[-1]
                    b["vertex_offset"] += 1

                    self._update_all_vertex(b)

            elif k == 2424832:
                if self.current_image > 0:
                    self.current_image -= 1
                    self.unmodified_current = cv2.imread(self.image_paths[self.current_image])
            elif k == 2555904:
                if self.current_image < len(self.image_paths) - 1:
                    self.current_image += 1
                    self.unmodified_current = cv2.imread(self.image_paths[self.current_image])
            elif k == ord("-") or k == ord("_"):
                self.last_button_q = False
                if self.vertex_offset > 1:
                    self.vertex_offset -= 1
                    print(f"Vertex size: {self.vertex_offset}")

                # If there is no selection made, adjust the most recent box vertex offset
                if len(self.selections) == 0 and len(self.boxes) > 0 and self.boxes[-1]["vertex_offset"] > 1:
                    b = self.boxes[-1]
                    b["vertex_offset"] -= 1

                    # recalculate best vertex location
                    self._update_all_vertex(b)

            elif k == ord("}") or k == ord("]"):
                self.last_button_q = False
                self.stroke_size += 1
                print(f"Stroke size: {self.stroke_size}")

                # If there is no selection made adjust most recent box stroke size
                if len(self.selections) == 0 and len(self.boxes) > 0:
                    self.boxes[-1]["stroke_size"] += 1

            elif k == ord("{") or k == ord("["):
                self.last_button_q = False
                if self.stroke_size > 1:
                    self.stroke_size -= 1

                # If there is no selection made, adjust the most recent box stroke size
                if len(self.selections) == 0 and len(self.boxes) > 0 and self.boxes[-1]["stroke_size"] > 1:
                    self.boxes[-1]["stroke_size"] -= 1
            elif k == ord("s"):
                self.last_button_q = False
                print("saving")
                self._save_outline()
            elif k == ord("q"):
                if self.last_button_q:
                    print("Quitting...")
                    cv2.destroyAllWindows()
                    break
                else:
                    self.last_button_q = True
                    print("Quit? Hit q again to quit program. Unsaved progress will be lost.")


def split_census_image(path: str) -> None:
    print(f"displaying: {path}")
    image = cv2.imread(str(path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, threshold1=100, threshold2=800)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('thresh', thresh)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        if area > 1000:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('edges', edges)
    cv2.imshow('Original image', image)
    cv2.imshow('Gray image', gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def initiate_directory() -> bool:
    """
    Setup local directories and verify everything is intact

    Returns:
        bool: True if program is ready to run, False if some user action is required
    """

    ready_to_run = True
    # Make image directory if it doesn't exist
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Image file not found at {IMAGES_DIR}")
        os.mkdir(IMAGES_DIR)
        print("Created image directory. Please populate it with images of the census cards")
        ready_to_run = False

    if not os.path.exists(DATA_DIR):
        print(f"INFO: data dir not found, creating one at: {DATA_DIR}")
        os.mkdir(DATA_DIR)

    if not os.path.exists(BOXED_PATH):
        print(f"INFO: Box json dir not found, creating one at: {BOXED_PATH}")
        os.mkdir(BOXED_PATH)

    if not os.path.exists(SLICED_CARDS):
        print(f"INFO: card slice dir not found, creating one at: {SLICED_CARDS}")
        os.mkdir(SLICED_CARDS)

    return ready_to_run


def main():
    if initiate_directory():
        print(f"Image location: {IMAGES_DIR}")
        bhc = BunkerHillCard(IMAGES_DIR)
        bhc.help()
        bhc.main_selection_loop()
    else:
        print("Quitting...")


if __name__ == "__main__":
    main()
