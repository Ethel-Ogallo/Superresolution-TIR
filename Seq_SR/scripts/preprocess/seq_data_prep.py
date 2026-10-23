import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.morphology import skeletonize, remove_small_objects
import networkx as nx
from pathlib import Path
import h5py

# ============================================================
# Configuration
# ============================================================
BASE = Path("/share/home/e2406751/Superresolution-TIR/data")
HR_DIR = BASE / "HR_downsampled"
LS_ALIGNED_DIR = BASE / "Landsat_aligned"
WATER_MASK_DIR = BASE / "AUX" / "water_masks"
SPLIT_JSON = BASE / "split_map.json"
OUT_DIR = BASE / "processed/seq_patches2"

HR_TILE = 256
LR_TILE = 64
SEQ_LEN = 5

# Baseline rules for image validation
BASE_MIN_VALID_FRAC = 0.10       
BIT_CLOUD = 3
BIT_CLOUD_SHADOW = 4
MAX_CLOUD_FRAC = 0.40
MAX_SHADOW_FRAC = 0.10

TEST_CAMPAIGNS = {"BRC_2022.tif", "BRC_2023.tif", "HAUT_2025.tif"}

OUT_DIR.mkdir(parents=True, exist_ok=True)
HR_H5_PATH = OUT_DIR / "hr_frames.h5"
LR_H5_PATH = OUT_DIR / "lr_frames.h5"

with open(SPLIT_JSON) as f:
    split_map = json.load(f)

# ============================================================
# Helper Functions
# ============================================================
def extract_bit(arr, bit):
    return ((arr.astype(np.uint16) >> bit) & 1) == 1

def get_river_skeleton_ordered(water_mask, row_start, row_end):
    sub = water_mask[row_start:row_end, :].astype(bool)
    if sub.sum() < 30:
        return []
    
    sub = remove_small_objects(sub, min_size=8)
    skel = skeletonize(sub)
    ys, xs = np.where(skel)
    if len(ys) < 15:
        return []

    G = nx.Graph()
    pt_set = set(zip(ys, xs))
    for y, x in pt_set:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: 
                    continue
                if (y + dy, x + dx) in pt_set:
                    G.add_edge((y, x), (y + dy, x + dx))

    if len(G.nodes) == 0:
        return []

    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)

    ends = [n for n in subgraph if subgraph.degree(n) == 1]
    
    if not ends:
        # Fallback if closed loop: start at northernmost node
        top_node = min(subgraph.nodes, key=lambda p: p[0])
        dist = nx.single_source_shortest_path_length(subgraph, top_node)
        bottom_node = max(dist, key=dist.get)
        path = nx.shortest_path(subgraph, top_node, bottom_node)
    else:
        # Strictly anchor top-to-bottom endpoints within this block boundary
        top_end = min(ends, key=lambda p: p[0])
        bottom_end = max(ends, key=lambda p: p[0])
        
        try:
            path = nx.shortest_path(subgraph, top_end, bottom_end)
        except:
            return []

    # Directional safeguard: ensure row coordinates increase (top -> bottom)
    if path[0][0] > path[-1][0]:
        path = path[::-1]

    return [(y + row_start, x) for y, x in path]

def make_tile_or_none(r_orig, c_orig, hr_data, lr_celsius, qa_data, row_start, row_end, tile_min_frac):
    if r_orig < row_start or (r_orig + HR_TILE) > row_end:
        return None, "boundary"
    if c_orig < 0 or (c_orig + HR_TILE) > hr_data.shape[1]:
        return None, "boundary"

    hr_tile = hr_data[r_orig:r_orig + HR_TILE, c_orig:c_orig + HR_TILE]
    lr_r = r_orig // 4
    lr_c = c_orig // 4
    lr_tile = lr_celsius[lr_r:lr_r + LR_TILE, lr_c:lr_c + LR_TILE]
    qa_tile = qa_data[lr_r:lr_r + LR_TILE, lr_c:lr_c + LR_TILE]

    if hr_tile.shape != (HR_TILE, HR_TILE) or lr_tile.shape != (LR_TILE, LR_TILE):
        return None, "shape"

    valid_fraction = np.isfinite(hr_tile).mean()
    if valid_fraction < tile_min_frac:
        return None, "nodata"

    if extract_bit(qa_tile, BIT_CLOUD).mean() > MAX_CLOUD_FRAC or extract_bit(qa_tile, BIT_CLOUD_SHADOW).mean() > MAX_SHADOW_FRAC:
        return None, "cloud"

    return {"hr": hr_tile, "lr": lr_tile}, None

# ============================================================
# Core Pipeline Execution
# ============================================================
def main():
    frame_metadata = {}
    sequence_manifest = []
    total_saved = {"train": 0, "val": 0, "test": 0}
    frame_counter = 0

    all_hr = []
    all_lr = []

    for hr_path in sorted(HR_DIR.glob("*.tif")):
        campaign = hr_path.stem
        print(f"\n=== Processing {campaign} ===")

        with rasterio.open(hr_path) as src:
            hr_data = src.read(1).astype(np.float32)
            hr_nodata = src.nodata
            hr_bounds = src.bounds
            hr_transform = src.transform
            hr_crs = src.crs

        lst_path = LS_ALIGNED_DIR / "LST_Celsius" / f"{campaign}.tif"
        qa_path = LS_ALIGNED_DIR / "QA_Pixel" / f"{campaign}_qa.tif"
        if not lst_path.exists() or not qa_path.exists():
            print(f"  Missing matching Landsat alignment files for {campaign}. Skipping.")
            continue

        with rasterio.open(lst_path) as src:
            lr_celsius = src.read(1).astype(np.float32)
        with rasterio.open(qa_path) as src:
            qa_data = src.read(1)

        with rasterio.open(WATER_MASK_DIR / f"{campaign}_water_mask.tif") as src:
            wm = np.zeros(hr_data.shape, dtype="uint8")
            reproject(source=rasterio.band(src, 1), destination=wm,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=hr_transform, dst_crs=hr_crs, resampling=Resampling.nearest)
        water_mask = wm > 0

        if hr_nodata is not None:
            hr_data[hr_data == hr_nodata] = np.nan
        hr_data[hr_data < -1e30] = np.nan
        hr_data[hr_data == -9999] = np.nan  

        is_test = hr_path.name in TEST_CAMPAIGNS
        blocks = [{"block_id":0,"row_start":0,"row_end":hr_data.shape[0],"split":"test"}] if is_test else split_map.get(hr_path.name, {}).get("blocks", [])

        for b in blocks:
            split = b["split"]
            bid = b["block_id"]
            rs = b["row_start"]
            re = min(b["row_end"], hr_data.shape[0])
            if re - rs < HR_TILE:
                continue

            # Calculate actual data presence in this block slice
            block_raw_slice = hr_data[rs:re, :]
            block_valid_pct = np.isfinite(block_raw_slice).mean()
            
            # Lower the requirement for sparse blocks to avoid structural zero-save rates
            tile_min_frac = BASE_MIN_VALID_FRAC if block_valid_pct > 0.08 else 0.005

            centerline = get_river_skeleton_ordered(water_mask, rs, re)
            if len(centerline) < SEQ_LEN:
                print(f"  Block {bid}: Centerline too short")
                continue

            container = []
            used_origins = set()
            skips = {"boundary": 0, "nodata": 0, "cloud": 0, "shape": 0}

            STRIDE = 13
            for i in range(0, len(centerline), STRIDE):
                r, c = centerline[i]
                if not (rs <= r < re):
                    continue

                # 1. Base default position centered on the centerline coordinate
                r_orig = int(max(rs, min(r - HR_TILE // 2, re - HR_TILE)))
                c_orig = int(max(0, min(c - HR_TILE // 2, hr_data.shape[1] - HR_TILE)))

                # 2. SWATH-SEEKING ADJUSTMENT:
                # If the centered window lands on missing flight paths/data boundaries,
                # shift the canvas coverage to recover valid pixels while anchoring the river element.
                best_fraction = np.isfinite(hr_data[r_orig:r_orig + HR_TILE, c_orig:c_orig + HR_TILE]).mean()

                if best_fraction < tile_min_frac:
                    # Check shifts (-64, 0, 64 pixels) to capture the swath path
                    for offset_r in [-64, 0, 64]:
                        for offset_c in [-64, 0, 64]:
                            test_r = int(max(rs, min(r_orig + offset_r, re - HR_TILE)))
                            test_c = int(max(0, min(c_orig + offset_c, hr_data.shape[1] - HR_TILE)))
                            
                            if test_r < rs or (test_r + HR_TILE) > re or test_c < 0 or (test_c + HR_TILE) > hr_data.shape[1]:
                                continue
                                
                            # Essential Check: Ensure the target centerline tracking point (r, c) remains inside the shifted tile
                            if not (test_r <= r < test_r + HR_TILE and test_c <= c < test_c + HR_TILE):
                                continue
                                
                            test_fraction = np.isfinite(hr_data[test_r:test_r + HR_TILE, test_c:test_c + HR_TILE]).mean()
                            if test_fraction > best_fraction:
                                best_fraction = test_fraction
                                r_orig, c_orig = test_r, test_c

                # 3. Post-validation coordinate deduplication verification
                if (r_orig, c_orig) in used_origins:
                    continue

                tile, reason = make_tile_or_none(r_orig, c_orig, hr_data, lr_celsius, qa_data, rs, re, tile_min_frac)

                if tile is not None:
                    used_origins.add((r_orig, c_orig))
                    container.append({
                        "row_origin": r_orig, 
                        "col_origin": c_orig, 
                        "hr": tile["hr"], 
                        "lr": tile["lr"]
                    })
                else:
                    skips[reason] = skips.get(reason, 0) + 1

            print(f"  Block {bid} ({split}): Raw Density={block_valid_pct:.2%} | Filter Threshold={tile_min_frac:.2%} | Saved Tiles={len(container)} | Skips={skips}")

            # Assemble clean sequential structures
            if len(container) >= SEQ_LEN:
                for i in range(len(container) - SEQ_LEN + 1):
                    seq = container[i:i + SEQ_LEN]
                    
                    frame_ids = []
                    for f in seq:
                        fname = f"f{frame_counter:08d}"
                        frame_ids.append(fname)

                        all_hr.append(f['hr'])
                        all_lr.append(f['lr'])

                        left = hr_bounds.left + f['col_origin'] * 7.5
                        top = hr_bounds.top - f['row_origin'] * 7.5
                        frame_metadata[fname] = {
                            "campaign": campaign,
                            "split": split,
                            "block_id": bid,
                            "frame_idx": frame_counter,
                            "row_origin": f['row_origin'],
                            "col_origin": f['col_origin'],
                            "hr_source": hr_path.name,
                            "bounds_2154": {"left": left, "right": left + HR_TILE*7.5, "top": top, "bottom": top - HR_TILE*7.5}
                        }
                        frame_counter += 1

                    sequence_manifest.append({
                        "sequence_name": f"{campaign}_b{bid}_s{len(sequence_manifest):07d}",
                        "split": split,
                        "campaign": campaign,
                        "block_id": bid,
                        "frame_ids": frame_ids
                    })
                    total_saved[split] += 1

    # Finalize HDF5 generation
    if len(all_hr) > 0:
        print("\nWriting final compressed HDF5 datasets...")
        all_hr = np.array(all_hr, dtype=np.float32)
        all_lr = np.array(all_lr, dtype=np.float32)

        with h5py.File(HR_H5_PATH, 'w') as hf:
            hf.create_dataset('frames', data=all_hr, compression='gzip', compression_opts=4)
        with h5py.File(LR_H5_PATH, 'w') as hf:
            hf.create_dataset('frames', data=all_lr, compression='gzip', compression_opts=4)

        with open(OUT_DIR / "frame_metadata.json", "w") as f:
            json.dump(frame_metadata, f, indent=2)
        with open(OUT_DIR / "sequence_manifest.json", "w") as f:
            json.dump(sequence_manifest, f, indent=2)

        print("\n=== PIPELINE RUN SUCCESSFUL ===")
        print(f"Sequence Breakdown Summary: {total_saved}")
        print(f"Total structured frames saved in dataset: {len(all_hr)}")
    else:
        print("\nERROR: No valid frames survived sequencing filters across any block context.")

if __name__ == "__main__":
    main()