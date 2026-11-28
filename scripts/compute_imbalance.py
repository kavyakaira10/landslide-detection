import os
import glob
import h5py
import numpy as np

# Dynamic path resolution to ensure it works from any directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_DIR = os.path.join(PROJECT_ROOT, "Data", "TrainData", "mask")

mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*.h5")))
print(f"Found {len(mask_files)} mask files. Counting pixels across the dataset...")

total_landslide_pixels = 0
total_background_pixels = 0

for idx, mf in enumerate(mask_files):
    with h5py.File(mf, "r") as f:
        mask = f["mask"][:]
        
    total_landslide_pixels += int(np.sum(mask == 1))
    total_background_pixels += int(np.sum(mask == 0))
    
    if (idx + 1) % 1000 == 0 or (idx + 1) == len(mask_files):
        print(f"  Processed {idx + 1}/{len(mask_files)} files...")

total_pixels = total_landslide_pixels + total_background_pixels
landslide_pct = (total_landslide_pixels / total_pixels) * 100
background_pct = (total_background_pixels / total_pixels) * 100
imbalance_ratio = total_background_pixels / max(total_landslide_pixels, 1)

print("\n" + "=" * 50)
print("TASK 9: CLASS IMBALANCE RESULTS")
print("=" * 50)
print(f"Total Masks Analyzed    : {len(mask_files):,}")
print(f"Total Pixels            : {total_pixels:,}")
print(f"Background Pixels (0)   : {total_background_pixels:,} ({background_pct:.2f}%)")
print(f"Landslide Pixels (1)    : {total_landslide_pixels:,} ({landslide_pct:.2f}%)")
print(f"Imbalance Ratio         : 1 Landslide pixel : {imbalance_ratio:.1f} Background pixels")
print("=" * 50)
