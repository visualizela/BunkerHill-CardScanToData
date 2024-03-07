from pathlib import Path

from wand.image import Image
from wand.image import Image
from wand.color import Color
from wand.display import display
from pydantic import BaseModel


class ProcessImageArgs(BaseModel):
    inpath: Path 
    outfolder: Path

def deskew_image(args: ProcessImageArgs):
    with Image(filename=str(args.inpath)) as img:
        img.deskew(0.4*img.quantum_range)
        img.save(filename=args.outfolder / args.inpath.name)


def trim_image(original_img, original_width, original_height, color, fuzz, ):
    with original_img.clone() as trimmed_img:
        trimmed_img.border(color, 3, 3)
        trimmed_img.trim(color=color, fuzz=fuzz, percent_background=0.0, reset_coords=True)

        new_width, new_height = trimmed_img.width, trimmed_img.height
        width_cut_percentage = (original_width - new_width) / original_width * 100
        height_cut_percentage = (original_height - new_height) / original_height * 100

        if width_cut_percentage > 10 or height_cut_percentage > 15:
            trimmed_img.close() 
            new_img = original_img.clone() 
            reverted_trim = True
        else:
            new_img = trimmed_img.clone()  
            reverted_trim = False
    return new_img, reverted_trim


def edgecut(args: ProcessImageArgs):
    
    with Image(filename=str(args.inpath)) as img:
        original_width, original_height = img.width, img.height
        
        # first clone the image and trim the white borders, if it trims too much, revert and go to next step
        white_trimmed, reverted_trim = trim_image(img, original_width, original_height, color=Color("white"), fuzz=0.30*img.quantum_range)
        if reverted_trim:
            print(f"reverted big fuzz white trim: {args.inpath}")
            white_trimmed, reverted_small_trim = trim_image(img, original_width, original_height, color=Color("white"), fuzz=30)
            if reverted_small_trim: 
                print(f"reverted small fuzz white trim: {args.inpath}")
                
        black_white_trimmed, reverted_trim = trim_image(white_trimmed, original_width, original_height, color=Color("black"), fuzz=0.30*white_trimmed.quantum_range)
        if reverted_trim:
            print(f"reverted big fuzz black trim: {args.inpath}")
            black_white_trimmed, reverted_small_trim = trim_image(white_trimmed, original_width, original_height, color=Color("black"), fuzz=30)
            if reverted_small_trim: 
                print(f"reverted small fuzz black trim: {args.inpath}")

        black_white_trimmed.save(filename=str(args.outfolder / args.inpath.name))


class MatchAndCropArgs(BaseModel):
    image_path: Path
    reference_path: Path
    output_dir: Path
    crop_size: tuple = (70, 70)
    similarity_threshold: float = 0.05
    dissimilarity_threshold: float = 0.618
    bad_match_dir: Path = Path("bad_match")

def match_and_crop_image(args: MatchAndCropArgs):
    with Image(filename=str(args.image_path)) as orig_img:
        # Crop the image to the top-left 70x70
        with orig_img.clone() as img:
            img.crop(0, 0, width=args.crop_size[0], height=args.crop_size[1])
            with Image(filename=str(args.reference_path)) as reference:
                location, diff = img.similarity(reference, args.similarity_threshold)
                if diff > args.dissimilarity_threshold:
                    print(f'Images too dissimilar to match: {args.image_path}')
                    orig_img.save(filename=args.bad_match_dir / args.image_path.name)
                elif diff <= args.similarity_threshold:
                    # print('First match @ {left}x{top}'.format(**location))
                    # Crop the image from the top left x and y where form.jpg was matched
                    orig_img.crop(location['left'], location['top'], width=orig_img.width, height=orig_img.height)
                    orig_img.save(filename=args.output_dir / args.image_path.name)
                else:
                    # print('Best match @ {left}x{top}'.format(**location))
                    # Crop the image from the top left x and y where form.jpg was matched
                    orig_img.crop(location['left'], location['top'], width=orig_img.width, height=orig_img.height)
                    orig_img.save(filename=args.output_dir / args.image_path.name)
