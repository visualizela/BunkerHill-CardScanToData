import cv2
import json
import numpy as np
import os
import win32api

from constants import (DATA_DIR, DEFAULT_VERTEX_OFFSET, DEFAULT_STROKE_SIZE, IMAGES_DIR, BOXED_PATH, SLICED_CARDS,
                       MOUSE_BOX_BUFFER_SIZE, MOUSE_BOX_SWITCH_TO_BOX_SPEED, MOUSE_BOX_SWITCH_TO_CURSOR_SPEED,
                       MOUSE_BOX_FLICKER_REDUCTION, MOUSE_BOX_COLOR)


class BunkerHillCard:
    path = None
    image = None
    display_state = 0
    boxes = []
    last_undo = None
    curr_box = {}
    selections = []
    vertex_offset = DEFAULT_VERTEX_OFFSET
    stroke_size = DEFAULT_STROKE_SIZE
    box_json = {}
    mouse_locations = [[0, 0] for i in range(MOUSE_BOX_BUFFER_SIZE)]
    frames_since_cursor_transition = 99
    cursor_is_box = False

    def __init__(self, path: str) -> None:
        """
        Initiate class variables

        Args:
            path (str): path to card
        """
        self.original = cv2.imread(str(path))
        self.path = path
        metadata = {}
        metadata["path"] = os.path.abspath(path)
        metadata["name"] = os.path.basename(path).split(".")[0]
        self.box_json["metadata"] = metadata

        self.initiate_box()

    def initiate_box(self) -> None:
        """
        Initiate an empty box json
        """

        self.curr_box = {
            "name": None,
            "color": (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)),
            "top_left_bb": (0, 0),
            "top_right_bb": (0, 0),
            "bottom_left_bb": (0, 0),
            "bottom_right_bb": (0, 0),
            "vertex_offset": self.vertex_offset,
            "stroke_size": self.stroke_size
        }

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
        tl = box["top_left_bb"]
        tr = box["top_right_bb"]
        bl = box["bottom_left_bb"]
        br = box["bottom_right_bb"]

        # Draw lines around 4 points
        cv2.line(image, (tl[0]+vertex_offset, tl[1]), (tr[0]-vertex_offset, tr[1]), c, ss)
        cv2.line(image, (tl[0], tl[1]+vertex_offset), (bl[0], bl[1]-vertex_offset), c, ss)
        cv2.line(image, (bl[0]+vertex_offset, bl[1]), (br[0]-vertex_offset, br[1]), c, ss)
        cv2.line(image, (br[0], br[1]-vertex_offset), (tr[0], tr[1]+vertex_offset), c, ss)

        # Draw search zones around 4 points
        tlz_tl = (tl[0]-vertex_offset, tl[1]+vertex_offset)
        tlz_br = (tl[0]+vertex_offset, tl[1]-vertex_offset)
        cv2.rectangle(image, tlz_tl, tlz_br, c, ss)

        trz_tl = (tr[0]-vertex_offset, tr[1]+vertex_offset)
        trz_br = (tr[0]+vertex_offset, tr[1]-vertex_offset)
        cv2.rectangle(image, trz_tl, trz_br, c, ss)

        blz_tl = (bl[0]-vertex_offset, bl[1]+vertex_offset)
        blz_br = (bl[0]+vertex_offset, bl[1]-vertex_offset)
        cv2.rectangle(image, blz_tl, blz_br, c, ss)

        brz_tl = (br[0]-vertex_offset, br[1]+vertex_offset)
        brz_br = (br[0]+vertex_offset, br[1]-vertex_offset)
        cv2.rectangle(image, brz_tl, brz_br, c, ss)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, box["name"], (tl[0]+20, tl[1]+30), font, 1, c, ss, cv2.LINE_AA)

        return image

    def _draw_selection(self, selection: list, image: np.ndarray) -> np.ndarray:
        """
        Draws the current selection on the given image.

        Args:
            selection (list): list of current selections (points the user has clicked)
            image (np.ndarray): image to draw the selection on

        Returns:
            np.ndarray: Image with the selection drawn on it
        """
        vertex_offset = self.vertex_offset
        ss = self.stroke_size

        box_draw_tl = (selection[0]-vertex_offset, selection[1]+vertex_offset)
        box_draw_br = (selection[0]+vertex_offset, selection[1]-vertex_offset)
        cv2.rectangle(image, box_draw_tl, box_draw_br, self.curr_box["color"], ss)
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
        return image

    def draw_image(self) -> None:
        """
        Shows the census card image either with or without annotations depending on the `display_state` flag. Redrawing
        the annotations each frame makes ensures the drawn image correctly reflects the state of the drawn boxes
        """
        to_show = self.original.copy()

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
            # else draw most recent box
            else:
                to_show = self._draw_box(self.boxes[-1], to_show)

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

    def click_event(self, event: int, x: int, y: int, flags, params) -> None:
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

            # save the selection and print it to terminal
            self.selections.append((x, y))
            print(f"box{len(self.boxes)} selection: ({x}, {y})")

            # If the user has selected a box
            if len(self.selections) == 2:

                # append other two implicit corners
                self.selections.append((self.selections[0][0], self.selections[1][1]))
                self.selections.append((self.selections[1][0], self.selections[0][1]))

                # Find and sort the edges of the box
                self.selections.sort(key=lambda i: i[0])
                s0 = self.selections[0]
                s1 = self.selections[1]
                s2 = self.selections[2]
                s3 = self.selections[3]
                self.curr_box["top_left_bb"] = s0 if s0[1] < s1[1] else s1
                self.curr_box["bottom_left_bb"] = s1 if s0[1] < s1[1] else s0
                self.curr_box["top_right_bb"] = s2 if s2[1] < s3[1] else s3
                self.curr_box["bottom_right_bb"] = s3 if s2[1] < self.selections[3][1] else s2
                self.curr_box["name"] = f"box{len(self.boxes)}"
                self.curr_box["vertex_offset"] = self.vertex_offset
                self.curr_box["stroke_size"] = self.stroke_size

                # Save the box and reset curr
                self.boxes.append(self.curr_box)
                self.initiate_box()
                self.selections = []

    def redo_last_undo(self) -> None:
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

    def undo_last_action(self) -> None:
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

    def save_outline(self):
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
        outp = "================================Census Selector================================\n"
        outp += "This is a function that helps the user select each sub-data field for each\n"
        outp += "census card. To use, click the screen around the vertex of each section's box.\n"
        outp += "Once you have drawn a box around each subfield hit enter and the program will\n"
        outp += "save your selections so it can run data analysis on the census cards. Below is\n"
        outp += "a list of each button you can use while drawing the boxes:\n\n"
        outp += "_______________________________________________________________________________\n"
        outp += "\"h\'=Help: print this help menu to the terminal\n"
        outp += "\'u\'=Undo: Undo your last action\n"
        outp += "\'r\'=Redo: Redo the last undo\n"
        outp += "\'d\'=Display: Toggle between displaying and hiding the boxes you have drawn\n"
        outp += "\'+\'=Increase Vertex Size: Increase the search distance for each box vertex\n"
        outp += "\'-\'=Decrease Vertex Size: Decrease the search distance for each box vertex\n"
        outp += "\'q\'=Quit: Quit the application\n"
        outp += "\'s\'=Save: saves your selections as sub-problems. Only use when you are done!\n"
        outp += "===============================================================================\n"
        print(outp)

    def define_box_edges(self) -> None:
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
        cv2.setMouseCallback('image', self.click_event)
        win32api.SetCursor(None)
        while True:
            self.draw_image()
            # update mouse position
            self.mouse_locations.append(self.mouse_locations[-1])
            self.mouse_locations.pop(0)
            if self.cursor_is_box:
                win32api.SetCursor(None)

            k = cv2.waitKey(1)
            if k == 3014656:
                cv2.destroyAllWindows()
            elif k == ord("d"):
                self.display_state += 1
                print(f"Display state: {self.display_state % 3}")
            elif k == ord("u"):
                self.undo_last_action()
            elif k == ord("r"):
                self.redo_last_undo()
            elif k == ord("h"):
                self.help()
            elif k == ord("+") or k == ord("="):
                self.vertex_offset += 1
                print(f"Vertex size: {self.vertex_offset}")
                # If there is no selection made adjust most recent box vertex offset
                if len(self.selections) == 0:
                    self.boxes[-1]["vertex_offset"] += 1

            elif k == ord("-") or k == ord("_"):
                if self.vertex_offset > 0:
                    self.vertex_offset -= 1
                    print(f"Vertex size: {self.vertex_offset}")

                # If there is no selection made, adjust the most recent box vertex offset
                if len(self.selections) == 0 and self.boxes[-1]["vertex_offset"] > 0:
                    self.boxes[-1]["vertex_offset"] -= 1

            elif k == ord("}") or k == ord("]"):
                self.stroke_size += 1
                print(f"Stroke size: {self.stroke_size}")

                # If there is no selection made adjust most recent box stroke size
                if len(self.selections) == 0:
                    self.boxes[-1]["stroke_size"] += 1

            elif k == ord("{") or k == ord("["):
                if self.stroke_size > 1:
                    self.stroke_size -= 1

                # If there is no selection made, adjust the most recent box stroke size
                if len(self.selections) == 0 and self.boxes[-1]["stroke_size"] > 0:
                    self.boxes[-1]["stroke_size"] -= 1
            elif k == ord("s"):
                print("saving")
                self.save_outline()
            elif k == ord("q"):
                cv2.destroyAllWindows()
                break


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


def process_cards(files: list[str]):

    if len(files) > 0:
        for f in files:
            bhc = BunkerHillCard(IMAGES_DIR / f)
            bhc.help()
            bhc.define_box_edges()
    else:
        print(f"Error: no images found at {IMAGES_DIR}")


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
        files = os.listdir(IMAGES_DIR)
        process_cards(files)
    else:
        print("Quitting...")


if __name__ == "__main__":
    main()
