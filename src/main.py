import os

import argparse

from src.constants import BOXED_PATH, DATA_DIR, IMAGES_DIR, SLICED_CARDS
from src.card_selection_ui import BunkerHillCard

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


def main(args):
    if initiate_directory():
        print(f"Image location: {args.images_dir}")
        bhc = BunkerHillCard(args.images_dir, args.no_find_vertex, save_dir=args.save_dir)
        bhc.help()
        bhc.main_selection_loop()
    else:
        print("Quitting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--images_dir', type=str, default=IMAGES_DIR, help='Path to the images directory')
    parser.add_argument('--save_dir', type=str, default=None, help='Sliced images save to')
    parser.add_argument('--no_find_vertex', type=bool, default=False, help='Don\'t search for verticies')
    args = parser.parse_args()
    main(args)
