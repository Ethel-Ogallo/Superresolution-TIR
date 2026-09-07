import json
import tarfile
import tempfile
import shutil
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from pathlib import Path

# ============================================================
# Paths
# ============================================================
BASE            = Path("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/TIR_data")
HR_DIR          = BASE / "HR"
LS_DIR          = BASE / "LR"
PAIRS_JSON      = BASE / "hr_lr_pairs.json"
OUT_LS_DIR      = BASE / "Landsat_aligned"

(OUT_LS_DIR / "LST_Celsius").mkdir(parents=True, exist_ok=True)
(OUT_LS_DIR / "QA_Pixel").mkdir(parents=True, exist_ok=True)

CLIP_BUFFER_M = 300

with open(PAIRS_JSON) as f:
    pairs_data = json.load(f)

pair_lookup = {p["hr_name"]: {"ls_name": c["ls_name"]} 
               for p in pairs_data["pairs"] 
               for c in p["candidates"] if c["selected"]}

# ============================================================
# Core Functions
# ============================================================
def parse_mtl_numeric(path):
    data = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"')
                try: data[k] = float(v)
                except ValueError: continue
    return data

def extract_scene_files(tar_path, tmp_dir):
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp_dir)
    files = list(tmp_dir.iterdir())
    b10 = next((f for f in files if "ST_B10" in f.name and f.suffix == ".TIF"), None)
    qa  = next((f for f in files if "QA_PIXEL" in f.name and f.suffix == ".TIF"), None)
    mtl = next((f for f in files if "_MTL.txt" in f.name), None)
    if not b10 or not qa or not mtl:
        raise FileNotFoundError(f"Missing core bands in {tar_path.name}")
    return b10, qa, mtl

def clip_and_reproject(input_path, hr_path, output_path, resampling, out_dtype="float32"):
    TARGET_LR_RES = 30.0  # Force native 30m Landsat resolution
    
    with rasterio.open(hr_path) as ref:
        hr_crs = ref.crs
        hr_bounds = ref.bounds
        
        # 1. Calculate dimensions in true 30m pixels based on HR extent
        lr_width = int(round((hr_bounds.right - hr_bounds.left) / TARGET_LR_RES))
        lr_height = int(round((hr_bounds.top - hr_bounds.bottom) / TARGET_LR_RES))
        
        # 2. Construct clean 30m transform covering exact HR geographic extent
        lr_transform = rasterio.transform.from_bounds(
            hr_bounds.left, hr_bounds.bottom, 
            hr_bounds.right, hr_bounds.top, 
            lr_width, lr_height
        )

    # 3. Reproject & crop raw Landsat to target 30m grid
    with rasterio.open(input_path) as src:
        dst_data = np.zeros((lr_height, lr_width), dtype=np.dtype(out_dtype))
        
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=lr_transform,
            dst_crs=hr_crs,
            resampling=resampling
        )
        
    out_meta = {
        "driver": "GTiff", 
        "height": lr_height, 
        "width": lr_width, 
        "count": 1, 
        "dtype": out_dtype, 
        "crs": hr_crs, 
        "transform": lr_transform,
        "nodata": np.nan if out_dtype == "float32" else 0
    }
                
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(dst_data, 1)

def main():
    for hr_path in sorted(HR_DIR.glob("*.tif")):
        fname, campaign = hr_path.name, hr_path.stem
        if fname not in pair_lookup: continue
            
        print(f"--- Aligning Landsat for: {campaign} ---")
        ls_tar = LS_DIR / pair_lookup[fname]["ls_name"]
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            b10_path, qa_path, mtl_path = extract_scene_files(ls_tar, tmp_dir)
            mtl_data = parse_mtl_numeric(mtl_path)
            scale, offset = mtl_data["TEMPERATURE_MULT_BAND_ST_B10"], mtl_data["TEMPERATURE_ADD_BAND_ST_B10"]
            
            b10_aligned = tmp_dir / "b10_aligned.tif"
            qa_final = OUT_LS_DIR / "QA_Pixel" / f"{campaign}_qa.tif"
            
            clip_and_reproject(b10_path, hr_path, b10_aligned, Resampling.bilinear, "float32")
            clip_and_reproject(qa_path, hr_path, qa_final, Resampling.nearest, "uint16")
            
            with rasterio.open(b10_aligned) as src:
                b10_data = src.read(1).astype(np.float32)
                meta = src.meta.copy()
                
            lst_celsius = np.where(b10_data > 0, b10_data * scale + offset - 273.15, np.nan).astype(np.float32)
            
            with rasterio.open(OUT_LS_DIR / "LST_Celsius" / f"{campaign}.tif", "w", **meta) as dst:
                dst.write(lst_celsius, 1)
        finally:
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()