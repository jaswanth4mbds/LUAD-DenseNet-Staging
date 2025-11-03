import os
import sys

# === Set up OpenSlide DLL path (must be FIRST on Windows) ===
openslide_bin = r"C:\Users\jaswa\Desktop\openslide\bin"
if os.path.isdir(openslide_bin):
    try:
        # Python 3.8+
        os.add_dll_directory(openslide_bin)
    except AttributeError:
        # Fallback for older Python: prepend to PATH
        os.environ['PATH'] = openslide_bin + os.pathsep + os.environ.get('PATH', '')

# === Now safe to import everything else ===
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from openslide import OpenSlide
from skimage.filters import threshold_otsu
from tqdm import tqdm
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_PATCH_SIZE = 224
DEFAULT_PATCHES_PER_SLIDE = 300
DEFAULT_WHITE_THRESH = 210
DEFAULT_TISSUE_RATIO = 0.6
DEFAULT_MASK_LEVEL = 2
DEFAULT_RANDOM_SEED = 42


def parse_args():
    """CLI arguments for the patch extractor."""
    parser = argparse.ArgumentParser(description="WSI Patch Extractor")
    parser.add_argument('--input_folder', type=str, required=True, help='Folder with .svs slides')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save patches')
    parser.add_argument('--patch_size', type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument('--patches_per_slide', type=int, default=DEFAULT_PATCHES_PER_SLIDE)
    parser.add_argument('--white_thresh', type=int, default=DEFAULT_WHITE_THRESH)
    parser.add_argument('--tissue_ratio', type=float, default=DEFAULT_TISSUE_RATIO)
    parser.add_argument('--mask_level', type=int, default=DEFAULT_MASK_LEVEL)
    parser.add_argument('--use_otsu', action='store_true', help='Use Otsu threshold for tissue detection')
    parser.add_argument('--normalize', action='store_true', help='Apply histogram normalization')
    parser.add_argument('--threads', type=int, default=1, help='Number of slides to process in parallel')
    return parser.parse_args()


def setup_logging():
    """Basic console logger."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def is_tissue(patch_np: np.ndarray, white_thresh: int, ratio_thresh: float) -> bool:
    """Heuristic tissue check: require enough non-white pixels.

    We mark a pixel as tissue if any channel is below `white_thresh`.
    Then require at least `ratio_thresh` fraction of tissue pixels.
    """
    return np.mean(np.any(patch_np < white_thresh, axis=2)) >= ratio_thresh


def safe_coordinates(x: int, y: int, slide_w: int, slide_h: int, patch_size: int):
    """Center a patch around (x, y) but clamp within slide bounds."""
    x0 = min(max(0, x - patch_size // 2), slide_w - patch_size)
    y0 = min(max(0, y - patch_size // 2), slide_h - patch_size)
    return x0, y0


def process_slide(slide_path: str, args) -> None:
    """Extract patches from a single WSI slide and save metadata.

    Creates one subfolder per slide inside `output_folder`.
    """
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    output_folder = os.path.join(args.output_folder, slide_name)
    os.makedirs(output_folder, exist_ok=True)

    try:
        slide = OpenSlide(slide_path)
        level = min(args.mask_level, slide.level_count - 1)
        downsample = slide.level_downsamples[level]
        mask_size = slide.level_dimensions[level]

        # --- Tissue mask (thumbnail at selected level) ---
        thumb = slide.read_region((0, 0), level, mask_size).convert("L")
        thumb_np = np.array(thumb)
        if args.use_otsu:
            threshold = threshold_otsu(thumb_np)
            mask = thumb_np < threshold
        else:
            mask = thumb_np < args.white_thresh

        coords_thumb = np.argwhere(mask)
        if len(coords_thumb) == 0:
            logging.warning(f"[{slide_name}] No tissue found, skipping.")
            return

        # Map thumbnail coords back to full-res coords
        coords_full = [(int(x * downsample), int(y * downsample)) for y, x in coords_thumb]
        random.shuffle(coords_full)

        patch_data = []
        saved = 0
        slide_w, slide_h = slide.dimensions

        for cx, cy in coords_full:
            if saved >= args.patches_per_slide:
                break

            x0, y0 = safe_coordinates(cx, cy, slide_w, slide_h, args.patch_size)
            patch = slide.read_region((x0, y0), 0, (args.patch_size, args.patch_size)).convert("RGB")
            patch_np = np.array(patch)

            # Skip mostly white patches
            if not is_tissue(patch_np, args.white_thresh, args.tissue_ratio):
                continue

            # Optional normalization (simple histogram equalization)
            if args.normalize:
                patch = ImageOps.equalize(patch)

            patch_filename = f"{slide_name}_patch_{saved + 1:03d}.png"
            patch_path = os.path.join(output_folder, patch_filename)
            patch.save(patch_path)

            patch_data.append({
                "filename": patch_filename,
                "x": x0,
                "y": y0,
                "slide": slide_name,
            })

            saved += 1

        if saved < args.patches_per_slide:
            logging.warning(f"[{slide_name}] Only saved {saved}/{args.patches_per_slide} patches (not enough valid tissue).")

        if patch_data:
            df = pd.DataFrame(patch_data)
            df.to_csv(os.path.join(output_folder, f"{slide_name}_patches.csv"), index=False)

        logging.info(f"[{slide_name}] Saved {saved} patches.")

    except Exception as e:
        logging.error(f"Error processing {slide_name}: {e}")
        traceback.print_exc()


def main():
    args = parse_args()
    setup_logging()
    os.makedirs(args.output_folder, exist_ok=True)

    slides = [f for f in os.listdir(args.input_folder) if f.lower().endswith('.svs')]
    full_paths = [os.path.join(args.input_folder, s) for s in slides]
    logging.info(f"Found {len(slides)} slides.")

    random.seed(DEFAULT_RANDOM_SEED)
    np.random.seed(DEFAULT_RANDOM_SEED)

    if args.threads > 1:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            # tqdm around the iterator to show per-slide progress
            list(tqdm(executor.map(lambda path: process_slide(path, args), full_paths), total=len(full_paths), desc="Extracting patches"))
    else:
        for path in tqdm(full_paths, desc="Extracting patches"):
            process_slide(path, args)


if __name__ == "__main__":
    main()
