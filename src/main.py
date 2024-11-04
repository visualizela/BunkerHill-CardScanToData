import os
from pathlib import Path

import argparse

from src.constants import BOXED_PATH, DATA_DIR, IMAGES_DIR, SLICED_CARDS
from src.card_selection_ui import BunkerHillCard


def initiate_directory(images_dir: str = IMAGES_DIR, data_dir: str = DATA_DIR) -> bool:
    """
    Setup local directories and verify everything is intact

    Args:
        images_dir (str): Path to the images directory. Defaults to IMAGES_DIR.
        data_dir (str): Path to the data directory. Defaults to DATA_DIR.

    Returns:
        bool: True if program is ready to run, False if some user action is required
    """

    ready_to_run = True

    if not os.path.exists(images_dir):
        print(f"Error: Image file not found at {images_dir}")
        os.mkdir(images_dir)
        print("Created image directory. Please populate it with images of the census cards")
        ready_to_run = False

    if not os.path.exists(data_dir):
        print(f"INFO: data dir not found, creating one at: {data_dir}")
        os.mkdir(data_dir)

    if not os.path.exists(BOXED_PATH):
        print(f"INFO: Box json dir not found, creating one at: {BOXED_PATH}")
        os.mkdir(BOXED_PATH)

    if not os.path.exists(SLICED_CARDS):
        print(f"INFO: card slice dir not found, creating one at: {SLICED_CARDS}")
        os.mkdir(SLICED_CARDS)

    return ready_to_run



def main():
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--images_dir', type=Path, default=IMAGES_DIR, help='Path to the images directory')
    parser.add_argument('--save_dir', type=Path, default=None, help='Sliced images save to')
    parser.add_argument('--region_mode', action='store_true', help='Segment regions instead of cards')
    args = parser.parse_args()

    if not initiate_directory(args.images_dir, args.save_dir):
        print("Quitting...")
    if args.region_mode:
        subfolders = [f for f in args.images_dir.iterdir() if f.is_dir()]
        for subfolder in subfolders:
            print(f"Image location: {subfolder}")

            subfolder_name = subfolder.name
            save_subdir = Path(args.save_dir, subfolder_name) if args.save_dir else None
            if save_subdir and (save_subdir / 'boxes.json').exists():
                print(f"Skipping {subfolder} as 'boxes.json' already exists in {save_subdir}")
                continue
            bhc = BunkerHillCard(subfolder, no_find_vertex=True, save_dir=save_subdir)
            bhc.help()
            bhc.main_selection_loop()
            
    else:    
        print(f"Image location: {args.images_dir}")
        bhc = BunkerHillCard(args.images_dir, no_find_vertex=False, save_dir=args.save_dir)
        bhc.help()
        bhc.main_selection_loop()

if __name__ == "__main__":
    main()
