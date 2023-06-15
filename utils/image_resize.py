import argparse
from pathlib import Path
import cv2
from tqdm import tqdm 
import imutils 

ALLOWED_IMAGE_EXTENSIONS = [".jpeg", ".png", ".jpg", ".webp"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--size', type=int, default=640,  help="Size to resize to (height)")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print paths of resized images")
    parser.add_argument('--in-place', action='store_true', help="Overwrite image with its same name")
    parser.add_argument('images', nargs=argparse.REMAINDER, help="Images to rename, accepts a glob")
     
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    images = [Path(ii) for ii in args.images if Path(ii).suffix in ALLOWED_IMAGE_EXTENSIONS]

    for path in tqdm(images):
        image = cv2.imread(path.as_posix())
        # resized_image = cv2.resize(image, (args.size[0], args.size[1]))
        resized_image = imutils.resize(image, width=None, height=args.size) # keep aspect ratio
       
        if args.in_place:
            dest_path = path
        else:
            dest_path = path.with_name(path.stem + "_resized" + path.suffix)
    
        cv2.imwrite(dest_path.as_posix(), resized_image)
        if args.verbose:
            print(f"Resized {path} -> {dest_path}")