"""
download_aux.py
───────────────
Downloads auxiliary optical/spectral data from Copernicus openEO for each
TIR campaign, then reprojects to EPSG:2154 in-place.

Band mapping
────────────
                     Sentinel-2 L2A    Landsat 8/9 (Collection 2 L2)
  Blue               B02               B02
  Green              B03               B03
  Red                B04               B04
  NIR                B08               B05   ← was B06 (SWIR1) — fixed
  SWIR1              B11               B06   ← was B06 (used as NIR) — fixed
  (SWIR2 not used for indices but kept for completeness if needed)

Indices (computed on-the-fly in openEO before download)
────────────────────────────────────────────────────────
  NDVI = (NIR - Red)  / (NIR + Red)
  NDWI = (Green - NIR) / (Green + NIR)
  NDMI = (NIR - SWIR1) / (NIR + SWIR1)

Output band layout in the final EPSG:2154 GeoTIFF
───────────────────────────────────────────────────
  1  Blue
  2  Green
  3  Red
  4  NIR   (B08 / B05)
  5  SWIR1 (B11 / B06)
  6  NDVI
  7  NDWI
  8  NDMI
  9  ESA WorldCover LULC  (kept but will be replaced by COSIA in patching)
  10 DEM (Copernicus GLO-30)

Note: LULC is kept in the download for archival / fallback. The patching
script (patch_aux_cosia.py) skips band 9 and uses the COSIA raster instead.
"""

import os
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import openeo

# ─── Config ───────────────────────────────────────────────────────────────────

os.chdir("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR")

OUT_DIR  = Path("TIR_data/auxiliary")
TMP_DIR  = Path("TIR_data/auxiliary_4326")   # intermediate WGS84 downloads
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

sites = [
    {"name": "BAS_2025",  "bbox": [4.6235, 44.2034, 4.8088, 44.7715], "date": "2025-07-22"},
    {"name": "BRC_2022",  "bbox": [5.5370, 45.6079, 5.6693, 45.7161], "date": "2022-07-20"},
    {"name": "BRC_2023",  "bbox": [5.5330, 45.6079, 5.6677, 45.7161], "date": "2023-07-17"},
    {"name": "DZM_2013",  "bbox": [4.6350, 44.1432, 4.7346, 44.4709], "date": "2013-07-25"},
    {"name": "DZM_2014",  "bbox": [4.6378, 44.2068, 4.7148, 44.4467], "date": "2014-06-22"},
    {"name": "DZM_2019",  "bbox": [4.6432, 44.2953, 4.6988, 44.4460], "date": "2019-06-26"},
    {"name": "DZM_2023",  "bbox": [4.6399, 44.2110, 4.7113, 44.5493], "date": "2023-07-12"},
    {"name": "HAUT_2025", "bbox": [5.4079, 45.5908, 5.8598, 45.9916], "date": "2025-07-01"},
    {"name": "PDR_2013",  "bbox": [4.7325, 45.2729, 4.8221, 45.4149], "date": "2013-07-16"},
    {"name": "PDR_2014",  "bbox": [4.7337, 45.2711, 4.8193, 45.4144], "date": "2014-07-16"},
    {"name": "PDR_2023",  "bbox": [4.7350, 45.2837, 4.8831, 45.7157], "date": "2023-07-18"},
]

# ─── Sensor router ────────────────────────────────────────────────────────────

def get_sensor(year: int) -> str:
    return "LANDSAT" if year <= 2014 else "SENTINEL"

# ─── Collection loaders ───────────────────────────────────────────────────────

def load_sentinel(conn, spatial_extent, start, end):
    """Sentinel-2 L2A — NIR=B08, SWIR1=B11."""
    return conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=[start, end],
        bands=["B02", "B03", "B04", "B08", "B11"],
        max_cloud_cover=15,
    )

def load_landsat(conn, spatial_extent, start, end):

    return conn.load_collection(
        "LANDSAT_BIMONTHLY_MOSAIC",
        spatial_extent=spatial_extent,
        temporal_extent=[start, end],
        bands=["B01", "B02", "B03", "B04", "B05"],
    )

# ─── Aux cube builder ─────────────────────────────────────────────────────────

def build_aux_cube(conn, bbox, start, end, sensor):
    spatial_extent = {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]}

    # ── Spectral cube ────────────────────────────────────────────────────────
    if sensor == "SENTINEL":
        cube = load_sentinel(conn, spatial_extent, start, end)
        green_band, red_band = "B03", "B04"
        nir_band,   swir_band = "B08", "B11"
    else:
        cube = load_landsat(conn, spatial_extent, start, end)
        green_band, red_band = "B02", "B03"
        nir_band,   swir_band = "B04", "B05"

    cube = cube.reduce_dimension("t", "median")

    # ── Map variables to the correct sensor-specific bands ───────────────────
    green = cube.band(green_band)
    red   = cube.band(red_band)
    nir   = cube.band(nir_band)
    swir  = cube.band(swir_band)

    # ── Spectral indices ─────────────────────────────────────────────────────
    ndvi = (nir - red)   / (nir + red)
    ndwi = (green - nir) / (green + nir)
    ndmi = (nir - swir)  / (nir + swir)

    ndvi = ndvi.add_dimension("bands", "NDVI", type="bands")
    ndwi = ndwi.add_dimension("bands", "NDWI", type="bands")
    ndmi = ndmi.add_dimension("bands", "NDMI", type="bands")

    # ── LULC — ESA WorldCover 2021 ───────────────────────────────────────────
    # Kept for archival; replaced by COSIA in patching.
    lulc = conn.load_collection(
        "ESA_WORLDCOVER_10M_2021_V2",
        spatial_extent=spatial_extent,
        bands=["MAP"],
    ).rename_labels("bands", ["LULC"])

    # ── DEM — Copernicus GLO-30 ──────────────────────────────────────────────
    dem = conn.load_collection(
        "COPERNICUS_30",
        spatial_extent=spatial_extent,
    ).reduce_dimension("t", "mean")

    # ── Merge into single cube ───────────────────────────────────────────────
    # Band order: Blue Green Red NIR SWIR1 NDVI NDWI NDMI LULC DEM
    return (
        cube
        .merge_cubes(ndvi)
        .merge_cubes(ndwi)
        .merge_cubes(ndmi)
        .merge_cubes(lulc)
        .merge_cubes(dem)
    )

# ─── Reprojection helper ──────────────────────────────────────────────────────

def reproject_to_2154(src_path: Path, dst_path: Path) -> bool:
    """
    Reproject a GeoTIFF from any CRS to EPSG:2154 using gdalwarp.
    Uses LZW compression and bilinear resampling for continuous bands.
    Returns True on success.
    """
    cmd = [
        "gdalwarp",
        "-t_srs", "EPSG:2154",
        "-r",     "bilinear",
        "-co",    "COMPRESS=LZW",
        "-co",    "TILED=YES",
        "-co",    "BIGTIFF=IF_SAFER",
        "-overwrite",
        str(src_path),
        str(dst_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [gdalwarp ERROR] {result.stderr.strip()}")
        return False
    return True

# ─── Connect ──────────────────────────────────────────────────────────────────

conn = openeo.connect(
    "https://openeo.dataspace.copernicus.eu"
).authenticate_oidc()

# ─── Submit jobs ──────────────────────────────────────────────────────────────

jobs = {}   # name → job_id

for s in sites:

    name   = s["name"]
    date   = datetime.strptime(s["date"], "%Y-%m-%d")
    start  = (date - timedelta(days=20)).strftime("%Y-%m-%d")
    end    = (date + timedelta(days=20)).strftime("%Y-%m-%d")
    sensor = get_sensor(date.year)

    final_path = OUT_DIR / f"{name}_aux.tif"
    if final_path.exists():
        print(f"[SKIP] {name} — already exists")
        continue

    print(f"[BUILD] {name}  sensor={sensor}  window={start} → {end}")

    cube = build_aux_cube(conn, s["bbox"], start, end, sensor)

    # Download in native CRS (4326); we reproject locally after
    job = cube.create_job(
        title=f"{name}_AUX",
        out_format="GTiff",
    )
    job.start_job()

    jobs[name] = job.job_id
    print(f"  [SUBMITTED] job_id={job.job_id}")

# ─── Poll ─────────────────────────────────────────────────────────────────────

if jobs:
    print("\nPolling job statuses …\n")
    while True:
        statuses = {n: conn.job(j).status() for n, j in jobs.items()}
        print("  " + " | ".join(f"{k}: {v}" for k, v in statuses.items()), end="\r")
        if all(v in ("finished", "error", "canceled") for v in statuses.values()):
            break
        time.sleep(30)
    print("\n")

# ─── Download + reproject ─────────────────────────────────────────────────────

for name, job_id in jobs.items():

    job = conn.job(job_id)

    if job.status() != "finished":
        print(f"[SKIP] {name} → job status: {job.status()}")
        continue

    tmp_path   = TMP_DIR / f"{name}_aux_4326.tif"
    final_path = OUT_DIR / f"{name}_aux.tif"

    # ── Find the GeoTIFF asset ───────────────────────────────────────────────
    try:
        assets = job.get_results().get_metadata().get("assets", {})
    except Exception as e:
        print(f"[ERROR] metadata {name}: {e}")
        continue

    tif_url = next(
        (a["href"] for a in assets.values() if "tif" in a.get("href", "").lower()),
        None,
    )
    if not tif_url:
        print(f"[ERROR] No GeoTIFF asset found for {name}")
        continue

    # ── Stream download to tmp ───────────────────────────────────────────────
    print(f"[DOWNLOAD] {name} → {tmp_path.name}")
    r = conn.session.get(tif_url, stream=True)
    r.raise_for_status()
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(8 * 1024 * 1024):
            if chunk:
                f.write(chunk)
    print(f"  Downloaded ({tmp_path.stat().st_size / 1e6:.1f} MB)")

    # ── Reproject to EPSG:2154 ───────────────────────────────────────────────
    print(f"  Reprojecting to EPSG:2154 → {final_path.name}")
    ok = reproject_to_2154(tmp_path, final_path)
    if ok:
        print(f"  [DONE] {final_path.name}")
        tmp_path.unlink()   # remove the 4326 intermediate
    else:
        print(f"  [ERROR] Reprojection failed — keeping raw download at {tmp_path}")

print("\nAll done.")