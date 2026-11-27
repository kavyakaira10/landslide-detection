import os
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to save PNGs without opening pop-up windows
import matplotlib.pyplot as plt

# Dynamic project root path resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "visualizations")

def normalize_display(bands):
    """
    Applies 2nd-98th percentile scaling for optimal satellite image contrast.
    """
    p2, p98 = np.percentile(bands, (2, 98))
    if p98 > p2:
        scaled = np.clip((bands - p2) / (p98 - p2), 0.0, 1.0)
    else:
        scaled = np.clip(bands, 0.0, 1.0)
    return scaled

def find_interesting_samples(num_samples=3):
    """
    Finds sample files with varying landslide sizes to make visual inspection informative.
    """
    mask_files = sorted(glob.glob(os.path.join(DATA_DIR, "TrainData", "mask", "*.h5")))
    found_samples = []
    
    print(f"Scanning masks to find {num_samples} diverse landslide examples...")
    for mf in mask_files:
        with h5py.File(mf, "r") as f:
            mask = f["mask"][:]
        landslide_pixels = int(np.sum(mask == 1))
        total_pixels = mask.size
        ratio = (landslide_pixels / total_pixels) * 100
        
        # Pick samples with prominent landslides (> 2% area)
        if ratio >= 2.5:
            base_name = os.path.basename(mf).replace("mask_", "image_")
            img_file = os.path.join(DATA_DIR, "TrainData", "img", base_name)
            found_samples.append({
                "img_path": img_file,
                "mask_path": mf,
                "name": base_name.replace(".h5", ""),
                "ratio": ratio,
                "pixels": landslide_pixels
            })
            if len(found_samples) >= num_samples:
                break
                
    return found_samples

def generate_visualization(sample, save_dir):
    """
    Generates a 6-panel visual comparison for a sample tile.
    """
    # 1. Load data
    with h5py.File(sample["img_path"], "r") as fi:
        img_data = fi["img"][:]
    with h5py.File(sample["mask_path"], "r") as fm:
        mask_data = fm["mask"][:]
        
    # Ensure 2D mask
    if mask_data.ndim == 3:
        mask_data = mask_data[0]
        
    # 2. Extract Channels
    # True Color RGB: B04 (Red, ch 3), B03 (Green, ch 2), B02 (Blue, ch 1)
    rgb = img_data[:, :, [3, 2, 1]]
    rgb_scaled = normalize_display(rgb)
    
    # False Color CIR: B08 (NIR, ch 7), B04 (Red, ch 3), B03 (Green, ch 2)
    cir = img_data[:, :, [7, 3, 2]]
    cir_scaled = normalize_display(cir)
    
    # Terrain: DEM (ch 12) & Slope (ch 13)
    dem = img_data[:, :, 12]
    slope = img_data[:, :, 13]
    
    # 3. Create 6-panel Figure
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))
    fig.patch.set_facecolor("#1e1e1e")
    
    titles = [
        "1. True Color (RGB)",
        "2. False Color (NIR)",
        "3. Elevation (DEM)",
        "4. Slope (Steepness)",
        "5. Ground-Truth Mask",
        "6. Satellite + Mask Overlay"
    ]
    
    # 1. RGB
    axes[0].imshow(rgb_scaled)
    
    # 2. False Color CIR (Healthy forest = red; bare landslide = grey/cyan)
    axes[1].imshow(cir_scaled)
    
    # 3. DEM
    im_dem = axes[2].imshow(dem, cmap="terrain")
    plt.colorbar(im_dem, ax=axes[2], fraction=0.046, pad=0.04)
    
    # 4. Slope
    im_slope = axes[3].imshow(slope, cmap="magma")
    plt.colorbar(im_slope, ax=axes[3], fraction=0.046, pad=0.04)
    
    # 5. Mask
    axes[4].imshow(mask_data, cmap="gray", vmin=0, vmax=1)
    
    # 6. Overlay on RGB
    axes[5].imshow(rgb_scaled)
    overlay_mask = np.zeros((*mask_data.shape, 4))
    overlay_mask[mask_data == 1] = [1.0, 0.0, 0.0, 0.55]  # Semi-transparent red
    axes[5].imshow(overlay_mask)
    
    # Formatting
    for ax, title in zip(axes, titles):
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#444444")
            
    fig.suptitle(
        f"Tile: {sample['name']} | Landslide Area: {sample['pixels']:,} pixels ({sample['ratio']:.2f}% of tile)",
        color="#00d4ff",
        fontsize=14,
        fontweight="bold",
        y=1.03
    )
    
    plt.tight_layout()
    output_filename = os.path.join(save_dir, f"{sample['name']}_visualization.png")
    plt.savefig(output_filename, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"Saved visualization: {output_filename}")
    return output_filename

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    samples = find_interesting_samples(num_samples=3)
    
    if not samples:
        print("No samples found matching criteria.")
        return
        
    print(f"\nGenerating visual plots for {len(samples)} tiles...")
    saved_files = []
    for s in samples:
        path = generate_visualization(s, OUTPUT_DIR)
        saved_files.append(path)
        
    print("\n" + "=" * 70)
    print("TASK 8 VISUALIZATION COMPLETED SUCCESSFULLY!")
    print(f"All images saved in: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
