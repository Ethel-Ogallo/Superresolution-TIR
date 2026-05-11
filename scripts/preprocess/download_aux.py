import openeo
from datetime import datetime, timedelta
import os
import time

# CONFIG
os.chdir("/share/home/e2406751/Superresolution-TIR")

OUT_DIR = "data/AUX/auxiliary"
os.makedirs(OUT_DIR, exist_ok=True)

sites = [
    {"name": "BAS_2025","bbox": [4.6235292131943595,44.20341474159712,4.8087985220913465,44.77149984473723],"date": "2025-07-22"},
    {"name": "BRC_2022","bbox": [5.537033405913344,45.60791424652272,5.66930433527257,45.70340982406573],"date": "2022-07-20"},
    {"name": "BRC_2023","bbox": [5.532976680440144,45.60829001750337,5.667723843425785,45.716136240747225],"date": "2023-07-17"},
    {"name": "DZM_2013","bbox": [4.635043627068157,44.14319994073103,4.7345843918671,44.47091486635755],"date": "2013-07-25"},
    {"name": "DZM_2014","bbox": [4.63786602478749,44.206803726831424,4.714878331501317,44.44676111126777],"date": "2014-06-22"},
    {"name": "DZM_2019","bbox": [4.6432161083050945,44.29534194032175,4.6988318347868505,44.44609235611887],"date": "2019-06-26"},
    {"name": "DZM_2023","bbox": [4.639897560179281,44.21104258477187,4.711294490724891,44.54933234676181],"date": "2023-07-12"},
    {"name": "HAUT_2025","bbox": [5.4079769335788015,45.590820615153746,5.859889397217935,45.991607534748766],"date": "2025-07-01"},
    {"name": "PDR_2013","bbox": [4.732594410555663,45.27292940345975,4.8221466801989585,45.41493723542345],"date": "2013-07-16"},
    {"name": "PDR_2014","bbox": [4.733757337013955,45.27114213632399,4.819312653772652,45.414425093195234],"date": "2014-07-16"},
    {"name": "PDR_2023","bbox": [4.735005672204379,45.28372290653185,4.883105433567413,45.71579997560222],"date": "2023-07-18"},
]

# CONNECT
conn = openeo.connect(
    "https://openeo.dataspace.copernicus.eu"
).authenticate_oidc()

# AUX CUBE
def build_aux_cube(conn, bbox, start, end):

    spatial_extent = {
        "west": bbox[0],
        "south": bbox[1],
        "east": bbox[2],
        "north": bbox[3]
    }

    # SENTINEL-2
    s2 = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=[start, end],
        bands=["B02", "B03", "B04", "B05", "B08", "B11"],
        max_cloud_cover=15
    )

    # Temporal median composite
    s2 = s2.reduce_dimension(dimension="t", reducer="median")

    # RAW BANDS
    b02 = s2.band("B02")
    b03 = s2.band("B03")
    b04 = s2.band("B04")
    b05 = s2.band("B05")
    b08 = s2.band("B08")
    b11 = s2.band("B11")

    # SPECTRAL INDICES
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
    )

    lulc = lulc.rename_labels("bands", ["LULC"])

    # DEM
    dem = conn.load_collection(
        "COPERNICUS_30",
        spatial_extent=spatial_extent
    )

    dem = dem.reduce_dimension(
        dimension="t",
        reducer="mean"
    )

    # MERGE
    aux_cube = (
        s2
        .merge_cubes(ndvi)
        .merge_cubes(ndwi)
        .merge_cubes(ndmi)
        .merge_cubes(lulc)
        .merge_cubes(dem)
    )

    return aux_cube


# SUBMIT JOBS
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

    print(f"[BUILD] {name} | window: {start} → {end}")

    cube = build_aux_cube(
        conn,
        s["bbox"],
        start,
        end
    )

    job = cube.create_job(
        title=f"{name}_AUX_STACK",
        out_format="GTiff",
        job_options={"crs": "EPSG:2154"}
    )

    job.start_job()

    jobs[name] = job.job_id

    print(f"[SUBMITTED] {name} → {job.job_id}")


# POLLING
print("\nWaiting for jobs...\n")

while True:

    statuses = {}

    for name, job_id in jobs.items():
        job = conn.job(job_id)
        statuses[name] = job.status()

    print(
        " | ".join([f"{k}: {v}" for k, v in statuses.items()]),
        end="\r"
    )

    if all(v in ["finished", "error", "canceled"] for v in statuses.values()):
        break

    time.sleep(30)

print("\nAll jobs finished.")

# DOWNLOAD
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


# AUX CHANNEL ORDER 
AUX_CHANNELS = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B08",
    "B11",
    "NDVI",
    "NDWI",
    "NDMI",
    "LULC",
    "DEM"
]

print("\nAUX CHANNELS:")
for i, ch in enumerate(AUX_CHANNELS):
    print(f"{i:02d}: {ch}")