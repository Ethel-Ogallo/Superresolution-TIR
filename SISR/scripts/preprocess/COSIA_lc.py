"""
cosia_landcover_rasterize.py
────────────────────────────
Rasterize COSIA land-cover classes to match the extent and resolution of 
existing HR (TIR) rasters. Outputs are full-extent uint8 GeoTIFFs.
classes:
1  Bâtiment - building
2  Zone perméable - permeable zone
3  Zone imperméable - impermeable zone
4  Piscine - swimming pool
5  Sol nu -  bare soil
6  Surface eau - water surface
8  Conifère  -  conifer
9  Feuillu  - deciduous tree
10  Broussaille -  brush
12  Pelouse - lawn
13  Culture - crop
14  Terre labourée -  plowed soil
15  Serre  -   greenhouse
"""

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import box

# ─── Paths ────────────────────────────────────────────────────────────────────
COSIA_DIR = Path("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/COSIA_1-0_GPKG_LAMB93_RHONE")
HR_DIR    = Path("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/TIR_data/processed/HR_downsampled")
LC_DIR    = Path("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/TIR_data/auxiliary/landcover")
LC_DIR.mkdir(parents=True, exist_ok=True)

CAMPAIGNS = [
    "BRC_2022.tif", "BRC_2023.tif", "HAUT_2025.tif",
    "BAS_2025.tif", "DZM_2013.tif", "DZM_2014.tif",
    "DZM_2019.tif", "DZM_2023.tif",
    "PDR_2013.tif", "PDR_2014.tif", "PDR_2023.tif",
]

# Later classes overwrite earlier ones
RENDER_ORDER = [13, 14, 12, 10, 9, 8, 2, 3, 5, 1, 15, 6, 4]
NODATA_VAL = 0 

def build_tile_index(cosia_dir: Path) -> gpd.GeoDataFrame:
    records = []
    # Loop through files and extract the ACTUAL bounding box from metadata
    for gpkg in sorted(cosia_dir.glob("*.gpkg")):
        try:
            # Using rows=0 is the fastest way to get metadata without loading all features
            gdf_meta = gpd.read_file(gpkg, rows=0)
            bounds = gdf_meta.total_bounds
            records.append({"path": gpkg, "geometry": box(*bounds)})
        except Exception as e:
            print(f"  Skipping {gpkg.name}: {e}")
            continue
    return gpd.GeoDataFrame(records, crs="EPSG:2154")

def rasterize_landcover(lc_gdf, 
                        hr_height, 
                        hr_width, 
                        hr_transform, 
                        render_order, 
                        nodata):
    
    canvas = np.full((hr_height, hr_width), nodata, dtype=np.uint8)

    for class_id in render_order:
        subset = lc_gdf[lc_gdf["numero"] == class_id]
        if subset.empty: continue
        shapes = [(geom, class_id) for geom in subset.geometry if geom and not geom.is_empty]
        layer = rasterize(shapes, 
                          out_shape=(hr_height, hr_width), 
                          transform=hr_transform, 
                          fill=0, 
                          dtype=np.uint8)
        mask = (layer != 0)
        canvas[mask] = layer[mask]
    return canvas


def main():
    tile_gdf = build_tile_index(COSIA_DIR)
    for fname in CAMPAIGNS:
        hr_path = HR_DIR / fname
        if not hr_path.exists(): continue
        print(f"Processing {fname} …")

        with rasterio.open(hr_path) as src:
            hr_crs, hr_transform = src.crs, src.transform
            hr_height, hr_width = src.height, src.width
            hr_box = box(*src.bounds)

        overlapping = tile_gdf[tile_gdf.intersects(hr_box)]
        gdfs = [gpd.read_file(row["path"], columns=["numero", "geometry"]) for _, row in overlapping.iterrows()]
        
        if not gdfs:
            lc_raster = np.zeros((hr_height, 
                                  hr_width), 
                                 dtype=np.uint8)
        else:
            lc_gdf = gpd.GeoDataFrame(pd.concat(gdfs, 
                                                ignore_index=True), 
                                      crs="EPSG:2154").clip(hr_box)
            lc_raster = rasterize_landcover(lc_gdf, 
                                            hr_height, 
                                            hr_width, 
                                            hr_transform, 
                                            RENDER_ORDER, 
                                            NODATA_VAL)

        out_path = LC_DIR / fname.replace(".tif", "_landcover.tif")
        with rasterio.open(out_path, "w", 
                           driver="GTiff", 
                           height=hr_height, 
                           width=hr_width, count=1, 
                           dtype=np.uint8, 
                           crs=hr_crs, 
                           transform=hr_transform, 
                           nodata=NODATA_VAL, 
                           compress="lzw") as dst:
            dst.write(lc_raster, 1)
            dst.update_tags(1, 
                            COSIA_classes="1=Building 2=Permeable 3=Impermeable 4=Pool 5=Bare_soil 6=Water 8=Conifer 9=Deciduous 10=Brush 12=Lawn 13=Crop 14=Plowed 15=Greenhouse")
        print(f"  Saved → {out_path}")

if __name__ == "__main__":
    main()