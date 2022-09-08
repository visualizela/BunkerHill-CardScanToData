import cv2
import os

from constants import *

def main():
    if os.path.exists(IMAGES_DIR):
        print(IMAGES_DIR)
        files = os.listdir(IMAGES_DIR)
        if len(files) > 0:
            for f in files:
                split_cencus_image(IMAGES_DIR / f)
        else:
            print(f"Error: no images found at {IMAGES_DIR}")
    else:
        print(f"Error: Image file not found at {IMAGES_DIR}")
        os.mkdir(IMAGES_DIR)
        print("Created image directory. Please populate it with images of the cencus cards")


def split_cencus_image(path: str) -> None:
    image = cv2.imread(str(path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original image',image)
    cv2.imshow('Gray image', gray)
if __name__ == "__main__":
    main()