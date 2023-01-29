import cv2
import os
import json
import random
import math
import time
from abc import ABC, abstractmethod
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


class card_analysis(ABC):
    box_name = None  # name of box being analyzed
    current_image = None
    base_path = None
    image_paths = []
    image_names = []
    detected_text = []
    # lines = {
    #     "words": []
    #     "text": "",
    #     "x": 0,
    #     "y": 0,
    #     "width": 0,
    #     "height": 0
    # }
    detected_check_boxes = []
    # check_box = {
    #     "pixels": [],
    #     "top_left": (),
    #     "bottom_right": ()
    # }

    def __init__(self, path) -> None:
        self.base_path = path
        if not os.path.exists(path):
            raise FileNotFoundError

        image_names = os.listdir(path)
        for name in image_names:
            joined = os.path.join(path, name)
            self.image_names.append(name)
            self.image_paths.append(joined)

        self.current_image = cv2.imread(self.image_paths[0])
        self.box_name = path.split(os.path.sep)[-1]
        self.detect_text_on_current_image()
        self.detect_boxes_on_current_image()

    def get_new_unique_color(self, colors: list, desired_distance: int, tries: int = 1_000) -> tuple:
        """
        Given a list of all used colors, and a desired distance of that color to all others,
        return a new color. If a unique color cannot be generated that is at least that distance
        apart it will try again with a lower desired distance.

        Args:
            colors (list): list of used colors
            desired_distance (int): radial value from all other colors

        Returns:
            tuple: unique color
            int: desired distance value that was set when the function returned (in case it had to
                 check lower values)
        """
        if desired_distance == 0:
            print("Error: cannot have desired distance = 0")
            return None

        new_dist = desired_distance
        desired_color = None
        for i in range(tries):

            # make new color
            new_blue = random.randint(0, 255)
            new_green = random.randint(0, 255)
            new_red = random.randint(0, 255)

            # Check the new color is different enough from all other colors
            outside_range = True
            for c in colors:

                blue_diff = abs(new_blue-c[0])
                green_diff = abs(new_green-c[1])
                red_diff = abs(new_red-c[2])

                if blue_diff < desired_distance and green_diff < desired_distance and red_diff < desired_distance:
                    outside_range = False
                    break

            # If the randomly chosen color is different enough return it
            if outside_range:
                desired_color = (new_blue, new_green, new_red)
                break

        # Reduce the desired distance if we cannot easily get a color that is different enough
        if desired_color is None:
            new_dist = max(0, desired_distance-max(1, int(desired_distance/25)))
            return self.get_new_unique_color(colors, new_dist, tries)
        else:
            return desired_color, desired_distance

    def plug_holes(self, image, hole_size: int = 3, min_line_len: int = 7):
        """
        Given a threshold image (white/black) this function will try to plug potential holes in the
        checkboxes. It does this by filling in gaps adjacent to white pixels that are <= hole_size
        away. For instance if the list [255, 255, 0, 255, 255] was a subset of the image passed in
        then it would plug the hole and return [255, 255, 255, 255, 255]. Note: in its state now
        this function will destroy most of the text and other information on the image, only use
        this when trying to detect the location of the boxes.

        Args:
            image (_type_): binary(threshold) image passed in
            hole_size (int, optional): Largest number of pixels that can be missing between
            two filled pixels. Defaults to 2.
            min_line_len (int, optional): Number of pixels in a row before considering hole
            plugging (prevents non-box shapes from being plugged)
        """
        plugged = image.copy()

        width = image.shape[1]
        height = image.shape[0]
        for row in range(height):
            for col in range(width):

                # How far in an adjacent direction a pixel is filled
                up_detected = 0
                down_detected = 0
                left_detected = 0
                right_detected = 0

                up_meets_min_len = True
                down_meets_min_len = True
                left_meets_min_len = True
                right_meets_min_len = True

                # only plug holes if on solid color
                if image[row][col] == 255:
                    for i in range(hole_size+1 + min_line_len):

                        offset = i + 1

                        up = row - offset
                        down = row + offset
                        left = col - offset
                        right = col + offset

                        # check if line is long enough to consider bridging gap
                        if i < min_line_len:
                            # check above
                            if up < 0 or image[up][col] != 255:
                                up_meets_min_len = False

                            # check below
                            if down >= height or image[down][col] != 255:
                                down_meets_min_len = False

                            # check left
                            if left < 0 or image[row][left] != 255:
                                left_meets_min_len = False

                            # check right
                            if right >= width or image[row][right] != 255:
                                right_meets_min_len = False

                        # check if there is gap to bridge
                        if i >= min_line_len:
                            # check above
                            if up >= 0 and image[up][col] == 255 and up_meets_min_len:
                                up_detected = offset

                            # check below
                            if down < height and image[down][col] == 255 and down_meets_min_len:
                                down_detected = offset

                            # check left
                            if left >= 0 and image[row][left] == 255 and left_meets_min_len:
                                left_detected = offset

                            # check right
                            if right < width and image[row][right] == 255 and right_meets_min_len:
                                right_detected = offset

                    # fill gaps
                    for j in range(down_detected):
                        plugged[row+j][col] = 255
                    for j in range(up_detected):
                        plugged[row-j][col] = 255
                    for j in range(left_detected):
                        plugged[row][col-j] = 255
                    for j in range(right_detected):
                        plugged[row][col+j] = 255

        return plugged

    def detect_boxes_on_current_image(self, min_width=30) -> None:
        unique_colors = []

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
        plugged = self.plug_holes(thresh)
        recolor = cv2.cvtColor(plugged, cv2.COLOR_GRAY2BGR)

        width = recolor.shape[1]
        height = recolor.shape[0]
        distance = 150

        # find each unique shape on image
        for x in range(width):
            for y in range(height):
                # cv2.waitKey(0)
                # to_show = recolor.copy()
                # to_show[y][x] = (0, 0, 255)
                # to_show = cv2.resize(to_show, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
                # cv2.imshow("Greyed", to_show)
                if not (recolor[y][x] - [0, 0, 0]).any():
                    curr_color, distance = self.get_new_unique_color(unique_colors, distance)
                    unique_colors.append(curr_color)
                    recolor = self.recursive_coloring(curr_color, (x, y), recolor)

        to_show = cv2.resize(recolor, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Greyed", to_show)
        return

    TODO: make this an iterative soultion, recursion is dangerous (max depth exceeded fast)
    def recursive_coloring(self, curr_color: tuple, pos: tuple, image) -> list:
        """
        Given a pixel, recursively get all pixels that are connected to it

        Returns:
            list: ist of pixels
        """
        x = pos[0]
        y = pos[1]
        width = image.shape[1]
        height = image.shape[0]
        black = (0, 0, 0)

        # x - 1
        if x - 1 >= 0 and not (image[y][x-1] - black).any():
            image[y][x-1] = curr_color
            image = self.recursive_coloring(curr_color, (x-1, y), image)

        # x + 1
        if x + 1 < width and not (image[y][x+1] - black).any():
            image[y][x+1] = curr_color
            image = self.recursive_coloring(curr_color, (x+1, y), image)

        # y - 1
        if y - 1 >= 0 and not (image[y-1][x] - black).any():
            image[y-1][x] = curr_color
            image = self.recursive_coloring(curr_color, (x, y-1), image)

        # y + 1
        if y + 1 < height and not (image[y+1][x] - black).any():
            image[y+1][x] = curr_color
            image = self.recursive_coloring(curr_color, (x, y+1), image)

        return image

    def detect_text_on_current_image(self) -> list:

        image_text_data = pytesseract.image_to_data(self.current_image, output_type=Output.DICT, config='--psm 3 --oem 1')
        LINE_DIST = 4
        n_boxes = len(image_text_data['text'])
        for i in range(n_boxes):
            c_text = image_text_data['text'][i]
            # check if the word is valid
            if int(image_text_data['conf'][i]) > 1 and len(c_text.replace(" ", "")) > 0:
                tmp = image_text_data['conf'][i]
                print(f"{c_text} | conf: {tmp}")
                (x, y, w, h) = (image_text_data['left'][i], image_text_data['top'][i], image_text_data['width'][i], image_text_data['height'][i])

                added = False
                mean_line_position = (y+h)/2
                for ln in self.detected_text:
                    curr_mean_line_position = (ln["y"] + ln["height"])/2
                    if abs(curr_mean_line_position - mean_line_position) < LINE_DIST:
                        new_x = min(ln["x"], x)
                        new_y = min(ln["y"], y)
                        new_height = max(h, ln["height"])
                        gap = ln["x"] - (w+x) if ln["x"] > x else x - (ln["width"]+ln["x"])
                        new_width = w + gap + ln["width"]
                        ln["text"] += " " + c_text
                        ln["words"].append(c_text)
                        ln["x"] = new_x
                        ln["width"] = new_width
                        ln["y"] = new_y
                        ln["height"] = new_height
                        added = True
                if not added:
                    new_line = {
                        "words": [c_text],
                        "text": c_text,
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h
                    }
                    self.detected_text.append(new_line)

    def draw_boxes_around_text(self) -> None:
        image_copy = self.current_image.copy()
        cv2.imshow("Image2", image_copy)
        # draw boxes and print words
        for line in self.detected_text:
            print(line)
            image_copy = cv2.rectangle(image_copy,
                                       (line["x"], line["y"]),
                                       (line["x"] + line["width"], line["y"] + line["height"]),
                                       (0, 255, 0),
                                       2)

        resized = cv2.resize(image_copy, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Image", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @abstractmethod
    def analyze_card(self, image) -> dict:
        pass


class race_of_household(card_analysis):

    def analyze_card(self) -> dict:
        print("beep boop running analysis", self.box_name)


if __name__ == "__main__":
    tmp = race_of_household("C:\\Users\\Admin\\Desktop\\USC\\Internships\\Library Data Visualization\\BunkerHill-CardScanToData\\data\\sliced_cards\\11-02-2022\\race_of_household")
    tmp.analyze_card()
    tmp.draw_boxes_around_text()
