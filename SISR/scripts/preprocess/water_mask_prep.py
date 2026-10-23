import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import box

COSIA_DIR = Path("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/COSIA_1-0_GPKG_LAMB93_RHONE")
HR_DIR    = Path("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/TIR_data/processed/HR_downsampled")
MASK_DIR  = Path("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/TIR_data/processed/water_masks_v2")
MASK_DIR.mkdir(parents=True, exist_ok=True)

WATER_CLASS = 6

CAMPAIGNS = [
    "BRC_2022.tif", "BRC_2023.tif", "HAUT_2025.tif",  
    "BAS_2025.tif", "DZM_2013.tif", "DZM_2014.tif",   
    "DZM_2019.tif", "DZM_2023.tif",
    "PDR_2013.tif", "PDR_2014.tif", "PDR_2023.tif",
]

# --- REPLACED: Build spatial index by reading actual file metadata ---
print("Building accurate tile index from metadata...")
tile_index = []
for gpkg in sorted(COSIA_DIR.glob("*.gpkg")):
    try:
        # Read file metadata to get the true spatial extent
        gdf_header = gpd.read_file(gpkg, rows=0) 
        bounds = gdf_header.total_bounds 
        tile_index.append({"path": gpkg, "geometry": box(*bounds)})
    except Exception as e:
        print(f"  Skipping {gpkg.name}: {e}")
        continue

tile_gdf = gpd.GeoDataFrame(tile_index, crs="EPSG:2154")
print(f"Indexed {len(tile_gdf)} tiles")

# --- Process each HR raster ---
for fname in CAMPAIGNS:
    hr_path = HR_DIR / fname
    if not hr_path.exists():
        continue

    print(f"\nProcessing {fname}...")
    with rasterio.open(hr_path) as src:
        hr_crs, hr_transform = src.crs, src.transform
        hr_width, hr_height = src.width, src.height
        hr_box = box(*src.bounds)

    # Intersection using verified bounds
    overlapping = tile_gdf[tile_gdf.intersects(hr_box)]
    print(f"  Overlapping tiles: {len(overlapping)}")

    if len(overlapping) == 0:
        water_mask = np.zeros((hr_height, hr_width), dtype=np.uint8)
    else:
        water_gdfs = []
        for _, row in overlapping.iterrows():
            gdf = gpd.read_file(row["path"])
            water = gdf[gdf["numero"] == WATER_CLASS]
            if not water.empty:
                water_gdfs.append(water)

        if not water_gdfs:
            water_mask = np.zeros((hr_height, hr_width), dtype=np.uint8)
        else:
            water_gdf = pd.concat(water_gdfs, ignore_index=True)
            water_gdf = gpd.GeoDataFrame(water_gdf, crs=tile_gdf.crs)
            
            # Rasterize
            water_mask = rasterize(
                [(geom, 1) for geom in water_gdf.geometry],
                out_shape   = (hr_height, hr_width),
                transform   = hr_transform,
                fill        = 0,
                dtype       = np.uint8,
                all_touched = True,  # Ensures thin rivers aren't missed
            )

    # Save
    out_path = MASK_DIR / fname.replace(".tif", "_water_mask.tif")
    with rasterio.open(out_path, "w", driver="GTiff", height=hr_height, width=hr_width,
                       count=1, dtype=np.uint8, crs=hr_crs, transform=hr_transform,
                       nodata=255) as dst:
        dst.write(water_mask, 1)

    print(f"  Saved: {out_path.name}")