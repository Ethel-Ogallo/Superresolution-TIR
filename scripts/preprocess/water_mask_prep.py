import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import box

COSIA_DIR = Path("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/COSIA_1-0_GPKG_LAMB93_RHONE")
HR_DIR    = Path("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/TIR_data/processed/HR_downsampled")
MASK_DIR  = Path("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/TIR_data/processed/water_masks")
MASK_DIR.mkdir(parents=True, exist_ok=True)

WATER_CLASS = 6

CAMPAIGNS = [
    "BRC_2022.tif", "BRC_2023.tif", "HAUT_2025.tif",  
    "BAS_2025.tif", "DZM_2013.tif", "DZM_2014.tif",   
    "DZM_2019.tif", "DZM_2023.tif",
    "PDR_2013.tif", "PDR_2014.tif", "PDR_2023.tif",
]

#  Parse tile extents from filenames 
def parse_tile_bounds(gpkg_path):
    """
    Extract 10km tile bounds from filename.
    e.g. D001_2024_840_6530 → x=840000, y=6530000
    tile covers x to x+10000, y to y+10000
    """
    parts = gpkg_path.stem.split("_")
    x = int(parts[2]) * 1000      # easting in metres
    y = int(parts[3]) * 1000      # northing in metres
    return box(x, y, x + 10000, y + 10000)

# Build spatial index of all tiles
print("Building tile index...")
tile_index = []
for gpkg in sorted(COSIA_DIR.glob("*.gpkg")):
    try:
        bounds = parse_tile_bounds(gpkg)
        tile_index.append({"path": gpkg, "geometry": bounds})
    except Exception:
        continue

tile_gdf = gpd.GeoDataFrame(tile_index, crs="EPSG:2154")
print(f"Indexed {len(tile_gdf)} tiles")

#  Process each HR raster 
for fname in CAMPAIGNS:
    hr_path = HR_DIR / fname
    if not hr_path.exists():
        print(f"[WARNING] Not found: {fname}")
        continue

    print(f"\nProcessing {fname}...")

    with rasterio.open(hr_path) as src:
        hr_crs       = src.crs
        hr_transform = src.transform
        hr_width     = src.width
        hr_height    = src.height
        hr_bounds    = src.bounds

    # Find which tiles overlap this HR raster
    hr_box        = box(*hr_bounds)
    overlapping   = tile_gdf[tile_gdf.intersects(hr_box)]
    print(f"  Overlapping CoSIA tiles: {len(overlapping)}")

    if len(overlapping) == 0:
        print(f"  No CoSIA tiles found — saving empty mask")
        water_mask = np.zeros((hr_height, hr_width), dtype=np.uint8)
    else:
        # Load and merge water polygons from overlapping tiles only
        water_gdfs = []
        for _, row in overlapping.iterrows():
            try:
                gdf   = gpd.read_file(row["path"])
                water = gdf[gdf["numero"] == WATER_CLASS]
                if len(water) > 0:
                    water_gdfs.append(water)
            except Exception as e:
                print(f"  [WARNING] Could not read {row['path'].name}: {e}")

        if not water_gdfs:
            print(f"  No water polygons in overlapping tiles")
            water_mask = np.zeros((hr_height, hr_width), dtype=np.uint8)
        else:
            water_gdf = gpd.GeoDataFrame(
                pd.concat(water_gdfs, ignore_index=True),
                crs=water_gdfs[0].crs
            )
            # Clip to exact HR extent
            water_gdf = water_gdf.clip(hr_box)
            print(f"  Water polygons after clip: {len(water_gdf)}")

            # Rasterize
            shapes = [
                (geom, 1) for geom in water_gdf.geometry
                if geom is not None and not geom.is_empty
            ]
            water_mask = rasterize(
                shapes,
                out_shape   = (hr_height, hr_width),
                transform   = hr_transform,
                fill        = 0,
                dtype       = np.uint8,
                all_touched = False,
            )

    pct = 100 * water_mask.sum() / (hr_height * hr_width)
    print(f"  Water pixels: {water_mask.sum():,} ({pct:.2f}%)")

    # Save
    out_path = MASK_DIR / fname.replace(".tif", "_water_mask.tif")
    with rasterio.open(out_path, "w",
                       driver="GTiff", height=hr_height, width=hr_width,
                       count=1, dtype=np.uint8,
                       crs=hr_crs, transform=hr_transform,
                       nodata=255) as dst:
        dst.write(water_mask, 1)

    print(f"  Saved: {out_path.name}")

print("\nAll water masks saved.")