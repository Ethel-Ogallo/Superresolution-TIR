import json
from datetime import datetime

# ---------------------------
# Load metadata
# ---------------------------
with open("/share/home/e2406751/Superresolution-TIR/data/processed/patches/metadata.json", "r") as f:
    metadata = json.load(f)


TIME_MAP = {
    "PDR_2023.tif": {"hr_time": "11:24-12:43", "lr_time": "10:22-10:23"},
    "DZM_2023.tif": {"hr_time": "11:35-12:19", "lr_time": "10:29-10:30"},
    "BRC_2023.tif": {"hr_time": "11:48-12:57", "lr_time": "10:22-10:23"},
    "BRC_2022.tif": {"hr_time": "12:07-13:14", "lr_time": "10:23-10:24"},
    "DZM_2019.tif": {"hr_time": "15:40-16:30", "lr_time": "10:23-10:24"},
    "PDR_2013.tif": {"hr_time": "17:47-18:18", "lr_time": "10:25-10:26"},
    "PDR_2014.tif": {"hr_time": "16:06-16:31", "lr_time": "10:23-10:24"},
    "DZM_2013.tif": {"hr_time": "14:40-15:15", "lr_time": "10:25-10:26"},
    "DZM_2014.tif": {"hr_time": "15:40-16:00", "lr_time": "10:29-10:30"},
    "BAS_2025.tif": {"hr_time": "15:02-16:37", "lr_time": "10:29-10:30"},
    "HAUT_2025.tif": {"hr_time": "14:43-16:21", "lr_time": "10:22-10:23"}
}

# ---------------------------
# Convert time string → decimal hours
# ---------------------------
def to_hours(t):
    if t is None:
        return None

    t = str(t).strip()

    # range format "HH:MM-HH:MM"
    if "-" in t:
        a, b = t.split("-")
        t1 = datetime.strptime(a.strip(), "%H:%M")
        t2 = datetime.strptime(b.strip(), "%H:%M")

        mid_seconds = (t1.timestamp() + t2.timestamp()) / 2
        mid = datetime.fromtimestamp(mid_seconds)

        return round(mid.hour + mid.minute / 60, 4)

    # single timestamp "HH:MM"
    t1 = datetime.strptime(t, "%H:%M")
    return round(t1.hour + t1.minute / 60, 4)

# ---------------------------
# Inject into metadata
# ---------------------------
missing = []

for key, entry in metadata.items():

    hr_source = entry.get("hr_source")

    if hr_source not in TIME_MAP:
        missing.append(hr_source)
        continue

    hr_time_raw = TIME_MAP[hr_source].get("hr_time")
    lr_time_raw = TIME_MAP[hr_source].get("lr_time")

    hr_time = to_hours(hr_time_raw)
    lr_time = to_hours(lr_time_raw)

    entry["hr_time"] = hr_time
    entry["lr_time"] = lr_time

    if hr_time is not None and lr_time is not None:
        entry["time_gap_hours"] = round(hr_time - lr_time, 4)

# ---------------------------
# Save updated metadata
# ---------------------------
with open("metadata2.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Done: metadata updated with hr_time, lr_time, time_gap_hours")

if missing:
    print("\nMissing HR scenes in TIME_MAP:")
    for m in sorted(set(missing)):
        print(" -", m)