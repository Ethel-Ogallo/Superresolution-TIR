import openeo
from datetime import datetime, timedelta
import os
import time
import glob


os.chdir("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR")
os.makedirs("TIR_data/sentinel2_aux", exist_ok=True)

# sites = [
#     {"name": "PDR", "bbox": [4.7326, 45.2711, 4.8831, 45.7158], "date": "2023-07-18"},
#     {"name": "DZM", "bbox": [4.6350, 44.1432, 4.7346, 44.5493], "date": "2023-07-12"},
#     {"name": "BRC", "bbox": [5.5330, 45.6079, 5.6693, 45.7161], "date": "2023-07-17"},
# ]
sites = [
    {"name": "BAS_2025",  "bbox": [4.6235, 44.2034, 4.8088, 44.7715], "date": "2025-07-22"},
    {"name": "HAUT_2025", "bbox": [5.4080, 45.5908, 5.8599, 45.9916], "date": "2025-07-01"},
]
conn = openeo.connect("https://openeo.dataspace.copernicus.eu").authenticate_oidc()

def build_aux_cube(conn, bbox, start, end):
    s2 = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]},
        temporal_extent=[start, end],
        bands=["B02", "B03", "B04", "B05", "B08", "B11"],
        max_cloud_cover=15
    )
    s2_median = s2.reduce_dimension(dimension="t", reducer="median")

    b02 = s2_median.band("B02")
    b03 = s2_median.band("B03")
    b04 = s2_median.band("B04")
    b05 = s2_median.band("B05")
    b08 = s2_median.band("B08")
    b11 = s2_median.band("B11")

    ndvi = (b08 - b04) / (b08 + b04)
    ndwi = (b03 - b08) / (b03 + b08)
    ndmi = (b08 - b11) / (b08 + b11)

    ndvi_cube = ndvi.add_dimension("bands", "NDVI", type="bands")
    ndwi_cube = ndwi.add_dimension("bands", "NDWI", type="bands")
    ndmi_cube = ndmi.add_dimension("bands", "NDMI", type="bands")

    return s2_median.merge_cubes(ndvi_cube).merge_cubes(ndwi_cube).merge_cubes(ndmi_cube)

# --- submit all jobs  ---
jobs = {}
for s in sites:
    name = s["name"]
    out_path = f"TIR_data/sentinel2_aux/{name}_aux.tif"

    if os.path.exists(out_path) and os.path.getsize(out_path) > 50 * 1024 * 1024:
        print(f"  {name} already complete, skipping.")
        continue

    date  = datetime.strptime(s["date"], "%Y-%m-%d")
    start = (date - timedelta(days=20)).strftime("%Y-%m-%d")
    end   = (date + timedelta(days=20)).strftime("%Y-%m-%d")

    aux = build_aux_cube(conn, s["bbox"], start, end)
    job = aux.create_job(
        title=f"{name}_aux",
        out_format="GTiff",
        job_options={"crs": "EPSG:2154"}
    )
    job.start_job()
    jobs[name] = job
    print(f" Submitted {name} → job_id: {job.job_id}")

# --- poll all jobs until all finished ---
print("\nWaiting for all jobs to complete...")
while True:
    statuses = {name: job.status() for name, job in jobs.items()}
    print("  " + " | ".join(f"{n}: {s}" for n, s in statuses.items()), end="\r")

    all_done = all(s in ("finished", "error", "canceled") for s in statuses.values())
    if all_done:
        break
    time.sleep(30)


# Download results
finished_jobs = {
    # "PDR": "j-2604270840184d44990736429d9efc36",
    # "DZM": "j-26042708403849969f82fa436b742296",
    # "BRC": "j-2604270840554cb299090d55ac31c7b7",
    "BAS":   "j-2604271543194ad8a1b5d321a884de1a",
    "HAUT":  "j-2604271543364bbd8c269dca7a3b825e",
}

os.chdir("/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR")
os.makedirs("TIR_data/sentinel2_aux", exist_ok=True)

for name, job_id in finished_jobs.items():
    out_path = f"TIR_data/sentinel2_aux/{name}_aux.tif"
    
    if os.path.exists(out_path) and os.path.getsize(out_path) > 50 * 1024 * 1024:
        print(f"  {name} already complete, skipping.")
        continue

    job = conn.job(job_id)
    assets = job.get_results().get_metadata()['assets']
    tif_asset = next((v for k, v in assets.items() if k.endswith('.tif')), None)
    url = tif_asset['href']

    print(f"Downloading {name}...")
    response = conn.session.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(out_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk)

    size = os.path.getsize(out_path) / (1024 * 1024)
    print(f" {name} saved → {out_path} ({size:.1f} MB)")

print("\nAll done.")