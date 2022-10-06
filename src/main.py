from concurrent.futures import process
import cv2
import os
import numpy as np

from constants import DEFAULT_VERTEX_OFFSET, DEFAULT_STROKE_SIZE, IMAGES_DIR


class BunkerHillCard:
    path = None
    image = None
    showing_marked = True
    boxes = []
    last_undo = None
    curr_box = {}
    selections = []
    vertex_offset = DEFAULT_VERTEX_OFFSET
    stroke_size = DEFAULT_STROKE_SIZE

    def __init__(self, path: str) -> None:
        self.original = cv2.imread(str(path))
        self.path = path
        self.initiate_box()

    def initiate_box(self) -> None:
        self.curr_box = {
            "color": (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)),
            "top_left": (0, 0),
            "top_right": (0, 0),
            "bottom_left": (0, 0),
            "bottom_right": (0, 0)
        }

    def draw_image(self) -> None:
        """
        Shows the census card image either with or without annotations depending on `showing_marked` flag. Redrawing
        the annotations each frame makes ensures the drawn image correctly reflects the state of the drawn boxes
        """

        marked = self.original.copy()
        if self.showing_marked:
            vertex_offset = self.vertex_offset
            ss = self.stroke_size
            # Draw completed boxes
            for i, b in enumerate(self.boxes):
                c = b["color"]
                tl = b["top_left"]
                tr = b["top_right"]
                bl = b["bottom_left"]
                br = b["bottom_right"]

                # Draw lines around 4 points
                cv2.line(marked, (tl[0]+vertex_offset, tl[1]), (tr[0]-vertex_offset, tr[1]), c, ss)
                cv2.line(marked, (tl[0], tl[1]+vertex_offset), (bl[0], bl[1]-vertex_offset), c, ss)
                cv2.line(marked, (bl[0]+vertex_offset, bl[1]), (br[0]-vertex_offset, br[1]), c, ss)
                cv2.line(marked, (br[0], br[1]-vertex_offset), (tr[0], tr[1]+vertex_offset), c, ss)

                # Draw search zones around 4 points
                tlz_tl = (tl[0]-vertex_offset, tl[1]+vertex_offset)
                tlz_br = (tl[0]+vertex_offset, tl[1]-vertex_offset)
                cv2.rectangle(marked, tlz_tl, tlz_br, c, ss)

                trz_tl = (tr[0]-vertex_offset, tr[1]+vertex_offset)
                trz_br = (tr[0]+vertex_offset, tr[1]-vertex_offset)
                cv2.rectangle(marked, trz_tl, trz_br, c, ss)

                blz_tl = (bl[0]-vertex_offset, bl[1]+vertex_offset)
                blz_br = (bl[0]+vertex_offset, bl[1]-vertex_offset)
                cv2.rectangle(marked, blz_tl, blz_br, c, ss)

                brz_tl = (br[0]-vertex_offset, br[1]+vertex_offset)
                brz_br = (br[0]+vertex_offset, br[1]-vertex_offset)
                cv2.rectangle(marked, brz_tl, brz_br, c, ss)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(marked, f"box{i}", (tl[0]+20, tl[1]+30), font, 1, c, ss, cv2.LINE_AA)

            # Draw current selection(s)
            for s in self.selections:
                box_draw_tl = (s[0]-vertex_offset, s[1]+vertex_offset)
                box_draw_br = (s[0]+vertex_offset, s[1]-vertex_offset)
                cv2.rectangle(marked, box_draw_tl, box_draw_br, self.curr_box["color"], ss)

        cv2.imshow('image', marked if self.showing_marked else self.original)

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
                self.curr_box["top_left"] = s0 if s0[1] < s1[1] else s1
                self.curr_box["bottom_left"] = s1 if s0[1] < s1[1] else s0
                self.curr_box["top_right"] = s2 if s2[1] < s3[1] else s3
                self.curr_box["bottom_right"] = s3 if s2[1] < self.selections[3][1] else s2

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

        while True:
            self.draw_image()
            k = cv2.waitKey(1)
            if k == 3014656:
                cv2.destroyAllWindows()
            elif k == ord("d"):
                self.showing_marked = not self.showing_marked
                print("Displaying boxes" if self.showing_marked else "Hiding boxes")
            elif k == ord("u"):
                self.undo_last_action()
            elif k == ord("r"):
                self.redo_last_undo()
            elif k == ord("h"):
                self.help()
            elif k == ord("+") or k == ord("="):
                self.vertex_offset += 1
                print(f"Vertex size: {self.vertex_offset}")
            elif k == ord("-") or k == ord("_"):
                self.vertex_offset -= 1
                print(f"Vertex size: {self.vertex_offset}")
            elif k == ord("}") or k == ord("]"):
                self.stroke_size += 1
                print(f"Stroke size: {self.stroke_size}")
            elif k == ord("{") or k == ord("["):
                if self.stroke_size > 1:
                    self.stroke_size -= 1
                print(f"Stroke size: {self.stroke_size}")
            elif k == ord("q"):
                cv2.destroyAllWindows()
                break

        # close the window


def split_census_image(path: str) -> None:
    print(f"displaying: {path}")
    image = cv2.imread(str(path))
    # to_display = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, threshold1=100, threshold2=800)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('thresh', thresh)

    # for c in contours:
    # cnt = []
    # to_display = image.copy()
    # if cv2.contourArea(c) > 600:
    #     cnt.append(c)
    #     cv2.drawContours(to_display, cnt, -1, (255,0,0), 2)
    #     cv2.imshow('Original image',to_display)
    #     cv2.waitKey(0)
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


def main():
    if os.path.exists(IMAGES_DIR):
        print(f"Image location: {IMAGES_DIR}")
        files = os.listdir(IMAGES_DIR)
        process_cards(files)
    else:
        print(f"Error: Image file not found at {IMAGES_DIR}")
        os.mkdir(IMAGES_DIR)
        print("Created image directory. Please populate it with images of the census cards")


if __name__ == "__main__":
    main()
    # TODO: next implementation, manually enter a bounding box near each vertex and software will auto line up.
