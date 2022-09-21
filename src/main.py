import cv2
import os
import numpy as np

from constants import *

def main():
    if os.path.exists(IMAGES_DIR):
        print(IMAGES_DIR)
        files = os.listdir(IMAGES_DIR)
        if len(files) > 0:
            for f in files:
                split_census_image(IMAGES_DIR / f)
        else:
            print(f"Error: no images found at {IMAGES_DIR}")
    else:
        print(f"Error: Image file not found at {IMAGES_DIR}")
        os.mkdir(IMAGES_DIR)
        print("Created image directory. Please populate it with images of the census cards")


def split_census_image(path: str) -> None:
    print(f"displaying: {path}")
    image = cv2.imread(str(path))
    to_display = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, threshold1=100, threshold2=800)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imshow('thresh',thresh)
    
    # for c in contours:
        # cnt = []
        # to_display = image.copy()
        # if cv2.contourArea(c) > 600:
        #     cnt.append(c)
        #     cv2.drawContours(to_display, cnt, -1, (255,0,0), 2)
        #     cv2.imshow('Original image',to_display)
        #     cv2.waitKey(0)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area > 1000:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        

    cv2.imshow('edges',edges)
    cv2.imshow('Original image',image)
    cv2.imshow('Gray image', gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def hough_lines_test(path):
#     print(f"displaying: {path}")
#     image = cv2.imread(str(path))
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edge = cv2.Canny(gray, 400, 800, None)

#     rho, theta, thresh, = 3, np.pi/180, 600
#     threshold, minLineLength, maxLineGap =  None, 200, 100
#     # lines = cv2.HoughLines(edge, rho, theta, thresh)
#     lines = cv2.HoughLinesP(edge, rho, theta, thresh, threshold, minLineLength, maxLineGap)

#     if lines is not None:
#         for i in range(0, len(lines)):
#             l = lines[i][0]
#             cv2.line(gray, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

#     cv2.imshow('edges',edge)
#     cv2.imshow('Original image',image)
#     cv2.imshow('Gray image', gray)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    # TODO: next implementation, manually enter a bounding box near each vertecy and software will auto line up.