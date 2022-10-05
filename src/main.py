import cv2
import os
import numpy as np

from constants import VERTEX_OFFSET, IMAGES_DIR


class BunkerHillCard:
    path = None
    image = None
    boxes = {}
    curr_box = {}
    selections = []

    def __init__(self, path: str) -> None:
        self.original = cv2.imread(str(path))
        self.marked = self.original.copy()
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

    def click_event(self, event, x, y, flags, params) -> None:

        if event == cv2.EVENT_LBUTTONDOWN:

            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            self.selections.append((x, y))
            cv2.rectangle(self.marked, (x-VERTEX_OFFSET, y+VERTEX_OFFSET),
                          (x+VERTEX_OFFSET, y-VERTEX_OFFSET), self.curr_box["color"], 2)

            cv2.imshow('image', self.marked)
            print(f"{len(self.selections)}")
            if len(self.selections) == 4:

                # Find and sort the edges of the box
                self.selections.sort(key=lambda x: x[0])
                llx = self.selections[0]
                lx = self.selections[1]
                rx = self.selections[2]
                rrx = self.selections[3]

                tl = self.curr_box["top_left"]
                tr = self.curr_box["top_right"]
                bl = self.curr_box["bottom_left"]
                br = self.curr_box["bottom_right"]

                tl = llx if llx[1] < lx[1] else lx
                bl = lx if llx[1] < lx[1] else llx
                tr = rx if rx[1] < rrx[1] else rrx
                br = rrx if rx[1] < rrx[1] else rx

                # Draw completed box
                cv2.line(self.marked, tl, tr, self.curr_box["color"], 2)
                cv2.line(self.marked, tl, bl, self.curr_box["color"], 2)
                cv2.line(self.marked, bl, br, self.curr_box["color"], 2)
                cv2.line(self.marked, br, tr, self.curr_box["color"], 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(self.marked, f"box{len(self.boxes)}",
                            (tl[0]+20, tl[1]+30), font, 1, self.curr_box["color"], 2, cv2.LINE_AA)

                # Save the box and reset curr
                self.boxes[f"box{len(self.boxes)}"] = self.curr_box
                self.initiate_box()
                self.selections = []

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

        showing_marked = True

        while True:
            # wait for a key to be pressed to exit
            cv2.imshow(
                'image', self.marked if showing_marked else self.original)
            k = cv2.waitKey(1)
            if k == 3014656:
                cv2.destroyAllWindows()
            elif k == ord("h"):
                showing_marked = not showing_marked
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


def main():
    if os.path.exists(IMAGES_DIR):
        print(IMAGES_DIR)
        files = os.listdir(IMAGES_DIR)
        if len(files) > 0:
            for f in files:
                bhc = BunkerHillCard(IMAGES_DIR / f)
                bhc.define_box_edges()
        else:
            print(f"Error: no images found at {IMAGES_DIR}")
    else:
        print(f"Error: Image file not found at {IMAGES_DIR}")
        os.mkdir(IMAGES_DIR)
        print("Created image directory. Please populate it with images of the census cards")


if __name__ == "__main__":
    main()
    # TODO: next implementation, manually enter a bounding box near each vertex and software will auto line up.
