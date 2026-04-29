import openeo
from datetime import datetime, timedelta
import os
import time

# ----------------------------
# CONFIG
# ----------------------------
os.chdir("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR")

OUT_DIR = "TIR_data/sentinel2_aux"
os.makedirs(OUT_DIR, exist_ok=True)

sites = [
    {"name": "PDR", "bbox": [4.7326, 45.2711, 4.8831, 45.7158], "date": "2023-07-18"},
    {"name": "DZM", "bbox": [4.6350, 44.1432, 4.7346, 44.5493], "date": "2023-07-12"},
    {"name": "BRC", "bbox": [5.5330, 45.6079, 5.6693, 45.7161], "date": "2023-07-17"},
    {"name": "BAS_2025",  "bbox": [4.6235, 44.2034, 4.8088, 44.7715], "date": "2025-07-22"},
    {"name": "HAUT_2025", "bbox": [5.4080, 45.5908, 5.8599, 45.9916], "date": "2025-07-01"},
]

conn = openeo.connect("https://openeo.dataspace.copernicus.eu").authenticate_oidc()


# ----------------------------
# AUX CUBE (PGDM STYLE)
# ----------------------------
def build_aux_cube(conn, bbox, start, end):

    # Sentinel-2
    s2 = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={
            "west": bbox[0],
            "south": bbox[1],
            "east": bbox[2],
            "north": bbox[3]
        },
        temporal_extent=[start, end],
        bands=["B02", "B03", "B04", "B05", "B08", "B11"],
        max_cloud_cover=15
    )

    s2 = s2.reduce_dimension(dimension="t", reducer="median")

    b02 = s2.band("B02")
    b03 = s2.band("B03")
    b04 = s2.band("B04")
    b05 = s2.band("B05")
    b08 = s2.band("B08")
    b11 = s2.band("B11")

    # Indices
    ndvi = (b08 - b04) / (b08 + b04)
    ndwi = (b03 - b08) / (b03 + b08)
    ndmi = (b08 - b11) / (b08 + b11)

    ndvi = ndvi.add_dimension("bands", "NDVI", type="bands")
    ndwi = ndwi.add_dimension("bands", "NDWI", type="bands")
    ndmi = ndmi.add_dimension("bands", "NDMI", type="bands")

    # ----------------------------
    # LULC (PGDM PRIOR)
    # ----------------------------
    lulc = conn.load_collection(
        "ESA_WORLDCOVER_10M_2021_V2",
        spatial_extent={
            "west": bbox[0],
            "south": bbox[1],
            "east": bbox[2],
            "north": bbox[3]
        },
        bands=["MAP"]
    )

    lulc = lulc.rename_labels("bands", ["LULC"])

    # Merge everything
    return (
        s2
        .merge_cubes(ndvi)
        .merge_cubes(ndwi)
        .merge_cubes(ndmi)
        .merge_cubes(lulc)
    )


# ----------------------------
# SUBMIT JOBS
# ----------------------------
conn = openeo.connect("https://openeo.dataspace.copernicus.eu").authenticate_oidc()

jobs = {}

for s in sites:
    name = s["name"]

    out_path = os.path.join(OUT_DIR, f"{name}_aux.tif")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 50 * 1024 * 1024:
        print(f"[SKIP] {name} already exists")
        continue

    date = datetime.strptime(s["date"], "%Y-%m-%d")
    start = (date - timedelta(days=20)).strftime("%Y-%m-%d")
    end   = (date + timedelta(days=20)).strftime("%Y-%m-%d")

    print(f"[BUILD] {name}")

    cube = build_aux_cube(conn, s["bbox"], start, end)

    job = cube.create_job(
        title=f"{name}_AUX_PGDM",
        out_format="GTiff",
        job_options={"crs": "EPSG:2154"}
    )

    job.start_job()

    jobs[name] = job.job_id
    print(f"[SUBMITTED] {name} → {job.job_id}")


# ----------------------------
# POLLING
# ----------------------------
print("\nWaiting for jobs...\n")

while True:
    statuses = {}

    for name, job_id in jobs.items():
        job = conn.job(job_id)
        statuses[name] = job.status()

    print(" | ".join([f"{k}: {v}" for k, v in statuses.items()]), end="\r")

    if all(v in ["finished", "error", "canceled"] for v in statuses.values()):
        break

    time.sleep(30)


print("\nAll jobs finished.")


# ----------------------------
# DOWNLOAD
# ----------------------------
for name, job_id in jobs.items():

    out_path = os.path.join(OUT_DIR, f"{name}_aux.tif")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 50 * 1024 * 1024:
        print(f"[SKIP DOWNLOAD] {name}")
        continue

    job = conn.job(job_id)
    assets = job.get_results().get_metadata()["assets"]

    tif_url = None
    for k, v in assets.items():
        if k.endswith(".tif"):
            tif_url = v["href"]
            break

    if tif_url is None:
        print(f"[ERROR] No TIFF for {name}")
        continue

    print(f"[DOWNLOAD] {name}")

    response = conn.session.get(tif_url, stream=True, timeout=300)
    response.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[DONE] {name} → {size_mb:.1f} MB")