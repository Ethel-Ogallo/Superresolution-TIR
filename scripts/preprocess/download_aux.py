import openeo
from datetime import datetime, timedelta
import os
import time

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
os.chdir("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR")

OUT_DIR = "TIR_data/auxiliary"
os.makedirs(OUT_DIR, exist_ok=True)

sites = [
    {"name": "BAS_2025","bbox": [4.6235,44.2034,4.8088,44.7715],"date": "2025-07-22"},
    {"name": "BRC_2022","bbox": [5.5370,45.6079,5.6693,45.7161],"date": "2022-07-20"},
    {"name": "BRC_2023","bbox": [5.5330,45.6079,5.6677,45.7161],"date": "2023-07-17"},
    {"name": "DZM_2013","bbox": [4.6350,44.1432,4.7346,44.4709],"date": "2013-07-25"},
    {"name": "DZM_2014","bbox": [4.6378,44.2068,4.7148,44.4467],"date": "2014-06-22"},
    {"name": "DZM_2019","bbox": [4.6432,44.2953,4.6988,44.4460],"date": "2019-06-26"},
    {"name": "DZM_2023","bbox": [4.6399,44.2110,4.7113,44.5493],"date": "2023-07-12"},
    {"name": "HAUT_2025","bbox": [5.4079,45.5908,5.8598,45.9916],"date": "2025-07-01"},
    {"name": "PDR_2013","bbox": [4.7325,45.2729,4.8221,45.4149],"date": "2013-07-16"},
    {"name": "PDR_2014","bbox": [4.7337,45.2711,4.8193,45.4144],"date": "2014-07-16"},
    {"name": "PDR_2023","bbox": [4.7350,45.2837,4.8831,45.7157],"date": "2023-07-18"},
]

# --------------------------------------------------
# CONNECT
# --------------------------------------------------
conn = openeo.connect(
    "https://openeo.dataspace.copernicus.eu"
).authenticate_oidc()

# --------------------------------------------------
# SENSOR ROUTER
# --------------------------------------------------
def get_sensor(year):
    return "LANDSAT" if year <= 2014 else "SENTINEL"

# --------------------------------------------------
# SAFE COLLECTION LOADER
# --------------------------------------------------
def load_landsat(conn, spatial_extent, start, end):
    return conn.load_collection(
        "LANDSAT_BIMONTHLY_MOSAIC",
        spatial_extent=spatial_extent,
        temporal_extent=[start, end],
        bands=["B02","B03","B04","B06","B07"]
    )

def load_sentinel(conn, spatial_extent, start, end):
    return conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=[start, end],
        bands=["B02","B03","B04","B08","B11"],
        max_cloud_cover=15
    )

# --------------------------------------------------
# AUX BUILDER
# --------------------------------------------------
def build_aux_cube(conn, bbox, start, end, sensor):

    spatial_extent = {
        "west": bbox[0],
        "south": bbox[1],
        "east": bbox[2],
        "north": bbox[3]
    }

    if sensor == "SENTINEL":
        cube = load_sentinel(conn, spatial_extent, start, end)
    else:
        cube = load_landsat(conn, spatial_extent, start, end)

    cube = cube.reduce_dimension("t", "median")

    # spectral bands mapping
    b02 = cube.band("B02")
    b03 = cube.band("B03")
    b04 = cube.band("B04")

    if sensor == "SENTINEL":
        b08 = cube.band("B08")   # NIR
        b11 = cube.band("B11")   # SWIR
    else:
        b08 = cube.band("B06")   # SWIR1 (proxy for NIR/SWIR mismatch handling)
        b11 = cube.band("B07")   # SWIR2

    # indices
    ndvi = (b08 - b04) / (b08 + b04)
    ndwi = (b03 - b08) / (b03 + b08)
    ndmi = (b08 - b11) / (b08 + b11)

    ndvi = ndvi.add_dimension("bands", "NDVI", type="bands")
    ndwi = ndwi.add_dimension("bands", "NDWI", type="bands")
    ndmi = ndmi.add_dimension("bands", "NDMI", type="bands")

    # LULC
    lulc = conn.load_collection(
        "ESA_WORLDCOVER_10M_2021_V2",
        spatial_extent=spatial_extent,
        bands=["MAP"]
    ).rename_labels("bands", ["LULC"])

    # DEM
    dem = conn.load_collection(
        "COPERNICUS_30",
        spatial_extent=spatial_extent
    ).reduce_dimension("t", "mean")

    return (
        cube
        .merge_cubes(ndvi)
        .merge_cubes(ndwi)
        .merge_cubes(ndmi)
        .merge_cubes(lulc)
        .merge_cubes(dem)
    )

# --------------------------------------------------
# SUBMIT JOBS
# --------------------------------------------------
jobs = {}

for s in sites:

    name = s["name"]
    date = datetime.strptime(s["date"], "%Y-%m-%d")

    start = (date - timedelta(days=20)).strftime("%Y-%m-%d")
    end   = (date + timedelta(days=20)).strftime("%Y-%m-%d")

    sensor = get_sensor(date.year)

    out_path = os.path.join(OUT_DIR, f"{name}_aux.tif")

    if os.path.exists(out_path):
        print(f"[SKIP] {name}")
        continue

    print(f"[BUILD] {name} → {sensor}")

    cube = build_aux_cube(conn, s["bbox"], start, end, sensor)

    job = cube.create_job(
        title=f"{name}_AUX",
        out_format="GTiff",
        job_options={"crs": "EPSG:2154"}
    )

    job.start_job()

    jobs[name] = job.job_id
    print(f"[SUBMITTED] {name}")

# --------------------------------------------------
# POLLING
# --------------------------------------------------
print("\nWaiting for jobs...\n")

while True:

    statuses = {n: conn.job(j).status() for n, j in jobs.items()}

    print(" | ".join([f"{k}: {v}" for k, v in statuses.items()]), end="\r")

    if all(v in ["finished", "error", "canceled"] for v in statuses.values()):
        break

    time.sleep(30)

print("\nAll jobs finished.")

# --------------------------------------------------
# DOWNLOAD (SAFE FIXED)
# --------------------------------------------------
for name, job_id in jobs.items():

    job = conn.job(job_id)

    if job.status() != "finished":
        print(f"[SKIP] {name} → {job.status()}")
        continue

    out_path = os.path.join(OUT_DIR, f"{name}_aux.tif")

    try:
        meta = job.get_results().get_metadata()
        assets = meta.get("assets", {})

        print(f"\n{name} assets:")
        for k, v in assets.items():
            print(" ", k, "→", v)

    except Exception as e:
        print(f"[ERROR] metadata {name}: {e}")
        continue

    tif_url = None

    for asset in assets.values():
        href = asset.get("href", "")
        if "tif" in href.lower():
            tif_url = href
            break

    if not tif_url:
        print(f"[ERROR] No GeoTIFF found for {name}")
        continue

    print(f"[DOWNLOAD] {name}")

    r = conn.session.get(tif_url, stream=True)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(8 * 1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"[DONE] {name}")


# | Index | Channel              | Meaning                           |
# | ----- | -------------------- | --------------------------------- |
# | 0     | B02                  | Blue                              |
# | 1     | B03                  | Green                             |
# | 2     | B04                  | Red                               |
# | 3     | B08                  | NIR (S2) / SWIR1 (Landsat proxy)  |
# | 4     | B11                  | SWIR (S2) / SWIR2 (Landsat proxy) |
# | 5     | NDVI                 | vegetation index                  |
# | 6     | NDWI                 | water index                       |
# | 7     | NDMI                 | moisture index                    |
# | 8     | LULC (one-hot stack) | land cover                        |
# | 9     | DEM                  | elevation                         |
