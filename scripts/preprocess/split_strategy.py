""""
split_strategy.py — Script to define train/val splits by blocking rasters into contiguous row segments.
Rationale:
- Avoids spatial leakage across train/val boundaries that would occur with random sampling of tiles.
- Respects the geographic AOI of each campaign by splitting along the row axis, which corresponds to the north-south direction in our data.
- Systematic assignment of blocks to train/val ensures consistent splits across campaigns while allowing flexibility for shorter rasters.

Test set is held out separately due to observable geographic AOI in the full dataset campaigns together, 
which would make it difficult to assign blocks without leakage. Test rasters will be reserved for final evaluation only.
"""

import numpy as np
import json
from pathlib import Path
import rasterio

# Config
data_dir = Path("/share/home/e2406751/Superresolution-TIR/data/HR_downsampled")
tile_size = 256
overlap = 0.5
step = int(tile_size * (1 - overlap))  # 128 pixels
# Block must be wider than one tile to prevent leakage across boundaries
# block_size_pixels = tile_size * 4  # 1024 pixels = ~7680m, safely > one tile
block_size_pixels = tile_size * 8  # larger block size for sequential sr


# Train/val rasters only
trainval_rasters = [
    "BAS_2025.tif",
    "DZM_2013.tif", "DZM_2014.tif", "DZM_2019.tif", "DZM_2023.tif",
    "PDR_2013.tif", "PDR_2014.tif", "PDR_2023.tif"
]

# set aside because of observable geographic AOI in full dataset campaigns together
# test set rasters only
# test_rasters = [
#     "BRC_2022.tif", "BRC_2023.tif", 
#     "HAUT_2025.tif"
# ]

split_map = {}

for fname in trainval_rasters:
    path = data_dir / fname
    with rasterio.open(path) as src:
        n_rows = src.height

    # Divide raster into blocks along row axis
    block_starts = list(range(0, n_rows, block_size_pixels))
    blocks = []
    n_blocks = len(block_starts)

    for i, start in enumerate(block_starts):
        raw_end = start + block_size_pixels
        
        # Check if this is the very last block in the campaign loop
        if i == n_blocks - 1:
            # Round up to the absolute nearest multiple of tile_size (256) past n_rows
            end = int(np.ceil(n_rows / tile_size) * tile_size)
        else:
            # Middle blocks stay perfectly scaled to block_size_pixels
            end = raw_end
            
        if n_blocks <= 3:
            split = "val" if i == 1 else "train"
        else:
            split = "val" if (i % 5 == 2) else "train"
            
        blocks.append({
            "block_id": i,
            "row_start": start,
            "row_end": end,                  
            "n_rows": end - start,
            "split": split
        })

    n_train = sum(1 for b in blocks if b["split"] == "train")
    n_val = sum(1 for b in blocks if b["split"] == "val")
    total_blocks = len(blocks)
    
    split_map[fname] = {
        "n_rows_total": n_rows,
        "n_blocks": total_blocks,
        "n_train_blocks": n_train,
        "n_val_blocks": n_val,
        "train_pct": round(100 * n_train / total_blocks, 1),
        "val_pct": round(100 * n_val / total_blocks, 1),
        "blocks": blocks
    }

    print(f"{fname}: {n_rows} rows → {total_blocks} blocks "
          f"({n_train} train / {n_val} val) "
          f"≈ {round(100*n_train/total_blocks)}% / {round(100*n_val/total_blocks)}%")

# Save blocking map
output_path = Path("/share/home/e2406751/Superresolution-TIR/data/seq_split_map.json")
with open(output_path, "w") as f:
    json.dump(split_map, f, indent=2)

print(f"\nSplit map saved to {output_path}")