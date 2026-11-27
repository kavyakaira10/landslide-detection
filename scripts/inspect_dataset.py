import os
import glob
import h5py
import numpy as np

CHANNEL_NAMES = [
    "B01 (Coastal Aerosol)",
    "B02 (Blue)",
    "B03 (Green)",
    "B04 (Red)",
    "B05 (Red Edge 1)",
    "B06 (Red Edge 2)",
    "B07 (Red Edge 3)",
    "B08 (Near Infrared - NIR)",
    "B08A (Narrow NIR)",
    "B09 (Water Vapour)",
    "B11 (SWIR 1)",
    "B12 (SWIR 2)",
    "DEM (Elevation in meters)",
    "Slope (Steepness in degrees)"
]

def inspect_single_pair(img_path, mask_path):
    print("=" * 70)
    print("STEP 1: DETAILED SINGLE-FILE INSPECTION")
    print("=" * 70)
    print(f"Sample Image File: {img_path}")
    print(f"Sample Mask File : {mask_path}")
    print("-" * 70)
    
    with h5py.File(img_path, "r") as f_img:
        img_keys = list(f_img.keys())
        print(f"Image HDF5 Keys     : {img_keys}")
        img_data = f_img[img_keys[0]][:]
        print(f"Image Array Shape   : {img_data.shape} (Expected: (128, 128, 14))")
        print(f"Image Data Type     : {img_data.dtype}")
        
    with h5py.File(mask_path, "r") as f_mask:
        mask_keys = list(f_mask.keys())
        print(f"Mask HDF5 Keys      : {mask_keys}")
        mask_data = f_mask[mask_keys[0]][:]
        print(f"Mask Array Shape    : {mask_data.shape}")
        print(f"Mask Data Type      : {mask_data.dtype}")
        print(f"Mask Unique Values  : {np.unique(mask_data).tolist()} (Expected: [0, 1] or [0])")
        
    print("\n" + "-" * 70)
    print(f"{'Idx':<4} | {'Channel Name':<30} | {'Min':>10} | {'Max':>10} | {'Mean':>10}")
    print("-" * 70)
    for i, name in enumerate(CHANNEL_NAMES):
        ch = img_data[:, :, i]
        print(f"{i:<4} | {name:<30} | {ch.min():>10.2f} | {ch.max():>10.2f} | {ch.mean():>10.2f}")
    print("-" * 70)


def scan_dataset(base_dir="Data"):
    print("\n" + "=" * 70)
    print("STEP 2: DATASET-WIDE HEALTH SCAN")
    print("=" * 70)
    
    splits = {
        "Train Images": os.path.join(base_dir, "TrainData", "img", "*.h5"),
        "Train Masks" : os.path.join(base_dir, "TrainData", "mask", "*.h5"),
        "Valid Images": os.path.join(base_dir, "ValidData", "img", "*.h5"),
        "Test Images" : os.path.join(base_dir, "TestData", "img", "*.h5"),
    }
    
    # 1. Check file counts
    counts = {}
    for name, pattern in splits.items():
        files = glob.glob(pattern)
        counts[name] = len(files)
        print(f"Found {len(files):>5} files for {name}")
        
    # 2. Comprehensive check on TrainData
    train_img_files = sorted(glob.glob(splits["Train Images"]))
    train_mask_files = sorted(glob.glob(splits["Train Masks"]))
    
    print("\nScanning all 3,799 Training pairs for integrity & anomalies...")
    
    corrupt_files = 0
    shape_mismatches = 0
    nan_count = 0
    inf_count = 0
    
    # Track global min/max for DEM and Slope
    global_dem_min = float("inf")
    global_dem_max = float("-inf")
    global_slope_min = float("inf")
    global_slope_max = float("-inf")
    
    # Check all files
    for idx, (img_f, mask_f) in enumerate(zip(train_img_files, train_mask_files)):
        try:
            with h5py.File(img_f, "r") as fi:
                arr = fi["img"][:]
            with h5py.File(mask_f, "r") as fm:
                marr = fm["mask"][:]
                
            if arr.shape != (128, 128, 14):
                shape_mismatches += 1
            if marr.shape != (128, 128) and marr.shape != (1, 128, 128):
                shape_mismatches += 1
                
            if np.isnan(arr).any() or np.isnan(marr).any():
                nan_count += 1
            if np.isinf(arr).any() or np.isinf(marr).any():
                inf_count += 1
                
            # Track terrain extremes
            dem_ch = arr[:, :, 12]
            slope_ch = arr[:, :, 13]
            if dem_ch.min() < global_dem_min: global_dem_min = float(dem_ch.min())
            if dem_ch.max() > global_dem_max: global_dem_max = float(dem_ch.max())
            if slope_ch.min() < global_slope_min: global_slope_min = float(slope_ch.min())
            if slope_ch.max() > global_slope_max: global_slope_max = float(slope_ch.max())
            
        except Exception as e:
            corrupt_files += 1
            print(f"Error reading pair {img_f}: {e}")
            
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(train_img_files):
            print(f"  Processed {idx + 1}/{len(train_img_files)} files...")
            
    print("\n" + "-" * 70)
    print("TRAIN INTEGRITY RESULTS:")
    print(f"  - Corrupt Files       : {corrupt_files} (Expected: 0)")
    print(f"  - Shape Mismatches    : {shape_mismatches} (Expected: 0)")
    print(f"  - Files with NaNs     : {nan_count} (Expected: 0)")
    print(f"  - Files with Infs     : {inf_count} (Expected: 0)")
    print(f"  - Global DEM Range    : {global_dem_min:.1f} m to {global_dem_max:.1f} m")
    print(f"  - Global Slope Range  : {global_slope_min:.1f}° to {global_slope_max:.1f}°")
    print("-" * 70)
    
    # Quick check on Valid and Test dimensions
    print("\nVerifying Validation & Test image shapes (first 50 sample check)...")
    val_files = glob.glob(splits["Valid Images"])[:50]
    test_files = glob.glob(splits["Test Images"])[:50]
    
    val_ok = all(h5py.File(f, "r")["img"][:].shape == (128, 128, 14) for f in val_files)
    test_ok = all(h5py.File(f, "r")["img"][:].shape == (128, 128, 14) for f in test_files)
    
    print(f"  - ValidData Shape Match (128, 128, 14): {'PASS' if val_ok else 'FAIL'}")
    print(f"  - TestData Shape Match  (128, 128, 14): {'PASS' if test_ok else 'FAIL'}")
    print("=" * 70)
    print("TASK 7 INSPECTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

if __name__ == "__main__":
    sample_img = os.path.join(DATA_DIR, "TrainData", "img", "image_1.h5")
    sample_mask = os.path.join(DATA_DIR, "TrainData", "mask", "mask_1.h5")
    inspect_single_pair(sample_img, sample_mask)
    scan_dataset(base_dir=DATA_DIR)
