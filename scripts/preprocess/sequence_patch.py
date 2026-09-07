"""
patch_sequences_basicvsr.py
===========================
Build sequential LR/HR patch sequences for BasicVSR++ from paired river TIR
rasters (Rhône system, 4× SR: 30 m LR → 7 m HR).

Supports all three training tiers — patch everything once, load selectively:

  Tier 1  LR thermal + HR thermal + WM                (baseline)
  Tier 2  + time scalars in metadata.json             (time-aware)
  Tier 3  + AUX_LR (23-ch aux at LR res)              (full aux)

Output layout
-------------
sequences_basicvsr/
  train/sequences/<raster>_seq<NNNN>/
    LR/      0000.npy …   float32 (1, 64,  64)   LR thermal
    HR/      0000.npy …   float32 (1, 256, 256)  HR thermal
    WM/      0000.npy …   float32 (1, 256, 256)  water mask (HR res)
    AUX_LR/  0000.npy …   float32 (23, 64,  64)  aux stack (LR res)
  val/  (same)
  test/ (same)
  metadata.json   one entry per sequence:
                  { lr_time, time_gap_hours, date_gap_days,
                    raster, n_frames }

Aux channel order (23 ch, matches SISR dataset.py):
  [0:5]   spectral bands 1-5        normalised [0,1]
  [5]     NDVI                      normalised [-1,1]
  [6]     NDWI                      normalised [-1,1]
  [7]     NDMI                      normalised [-1,1]
  [8:22]  COSIA one-hot (14 cls)    {0,1}
  [22]    DEM                       normalised [0,1]

Scale: HR 256 px @ 7 m  ↔  LR 64 px @ 30 m  (factor 4)
Overlap: 50 % for train, val AND test (needed for seam-free inference)
"""

import json
import numpy as np
import rasterio
import rasterio.windows
import rasterio.transform
from rasterio.warp import reproject, Resampling
from pathlib import Path
from scipy.ndimage import label as scipy_label
from skimage.morphology import skeletonize, binary_dilation, disk

# ──────────────────────────────── CONFIG ─────────────────────────────────────

# ── Thermal rasters ──
HR_THERMAL_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/HR")
LR_THERMAL_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/LR_30m")

# ── Aux rasters ──
AUX_DIR      = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/auxiliary")
COSIA_LC_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/COSIA")
WM_DIR       = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/water_masks")
DEM_STATS    = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/dem_stats.json")

# ── Time metadata ──
# JSON with one entry per campaign: { "BAS_2025": { "lr_time": 10.37,
#   "time_gap_hours": 4.2, "date_gap_days": 12 }, ... }
TIME_META_PATH = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/time_metadata.json")

# ── Output ──
OUT_ROOT  = Path("/share/home/e2406751/Superresolution-TIR/data/sequences_basicvsr")
SPLIT_MAP = Path("/share/home/e2406751/Superresolution-TIR/data/split_map.json")

# ── Geometry ──
HR_PATCH           = 256
SCALE              = 4          # LR patch = 64
OVERLAP            = 0.5        # applied to train, val AND test
MIN_SEQ_LEN        = 5
BEND_THRESHOLD_DEG = 90
MIN_WATER_FRAC     = 0.25       # reject patch if < 25 % water in HR WM

# ── COSIA class list (matches SISR patching) ──
COSIA_CLASSES = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 0]
N_COSIA = len(COSIA_CLASSES)   # 14

TRAINVAL_RASTERS = [
    "BAS_2025.tif",
    "DZM_2013.tif", "DZM_2014.tif", "DZM_2019.tif", "DZM_2023.tif",
    "PDR_2013.tif", "PDR_2014.tif", "PDR_2023.tif",
]
TEST_RASTERS = [
    "BRC_2022.tif", "BRC_2023.tif", "BRC_2024.tif",
    "HAUT_2025.tif",
]

# ─────────────────────────── AUX NORMALISATION ───────────────────────────────
# Mirrors your SISR aux patching script exactly.

def _get_sensor(year: int) -> str:
    return "LANDSAT" if year <= 2014 else "SENTINEL"

def _norm_spectral(x: np.ndarray, year: int) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0)
    divisor = 10_000.0 if _get_sensor(year) == "SENTINEL" else 120.0
    return np.clip(x / divisor, 0.0, 1.0)

def _norm_index(x: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(x, nan=0.0), -1.0, 1.0)

def _norm_dem(x: np.ndarray, p2: float, p98: float) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0)
    return np.clip((x - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

def _encode_cosia(x: np.ndarray) -> np.ndarray:
    """(H, W) int → (14, H, W) one-hot float32."""
    H, W = x.shape
    out = np.zeros((N_COSIA, H, W), dtype=np.float32)
    for i, cls in enumerate(COSIA_CLASSES):
        out[i] = (x == cls).astype(np.float32)
    return out

# ────────────────────────── CENTERLINE EXTRACTION ────────────────────────────

def extract_centerline(raster_name: str) -> np.ndarray:
    """
    Build river centerline from HR water mask (preferred).
    Falls back to cold-pixel threshold on HR thermal if mask missing.
    Returns (N, 2) array of (row, col) in HR pixel coordinates.
    """
    stem    = Path(raster_name).stem
    wm_path = WM_DIR / f"{stem}_water_mask.tif"

    if wm_path.exists():
        with rasterio.open(wm_path) as src:
            mask = (src.read(1) == 1).astype(np.uint8)
    else:
        # fallback: cold-pixel threshold
        with rasterio.open(HR_THERMAL_DIR / raster_name) as src:
            data   = src.read(1).astype(float)
            nodata = src.nodata
        valid = (data != nodata) if nodata is not None else (data > 0)
        thr   = np.percentile(data[valid], 15)
        mask  = (valid & (data <= thr)).astype(np.uint8)

    # keep largest connected component
    mask = binary_dilation(mask, disk(2)).astype(np.uint8)
    labeled, n = scipy_label(mask)
    if n == 0:
        raise ValueError(f"Empty river mask for {raster_name}")
    sizes    = np.bincount(labeled.ravel()); sizes[0] = 0
    mask     = (labeled == sizes.argmax()).astype(np.uint8)

    skel     = skeletonize(mask)
    ys, xs   = np.where(skel)
    if len(ys) == 0:
        raise ValueError(f"Empty skeleton for {raster_name}")

    return _walk_skeleton(np.column_stack([ys, xs]))


def _walk_skeleton(coords: np.ndarray) -> np.ndarray:
    """Greedy NN walk from topmost point — orders skeleton along the river."""
    idx       = int(np.argmin(coords[:, 0]))
    order     = [idx]
    remaining = np.ones(len(coords), dtype=bool)
    remaining[idx] = False
    cur = coords[idx]
    while remaining.any():
        pool   = coords[remaining]
        nn_rel = int(np.argmin(((pool - cur) ** 2).sum(axis=1)))
        nn_abs = np.where(remaining)[0][nn_rel]
        order.append(nn_abs); remaining[nn_abs] = False
        cur = coords[nn_abs]
    return coords[order]

# ──────────────────────────── ANCHOR PLACEMENT ───────────────────────────────

def place_anchors(centerline: np.ndarray, stride_hr: int) -> np.ndarray:
    """Sample (row, col) positions every stride_hr px of arc-length."""
    segs    = np.diff(centerline, axis=0)
    arc     = np.concatenate([[0.0], np.cumsum(np.hypot(segs[:, 0], segs[:, 1]))])
    pos     = np.arange(0, arc[-1], stride_hr)
    rows    = np.interp(pos, arc, centerline[:, 0].astype(float))
    cols    = np.interp(pos, arc, centerline[:, 1].astype(float))
    return np.column_stack([rows, cols])


def split_at_bends(anchors: np.ndarray, threshold_deg: float = BEND_THRESHOLD_DEG):
    """Yield anchor sub-arrays, splitting where direction change > threshold."""
    if len(anchors) < 2:
        yield anchors; return
    d      = np.diff(anchors, axis=0)
    angles = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    diff_a = np.abs(np.diff(angles))
    diff_a = np.minimum(diff_a, 360 - diff_a)
    splits = list(np.where(diff_a > threshold_deg)[0] + 1)
    bounds = [0] + splits + [len(anchors)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= 2:
            yield anchors[a:b]


def label_anchors(anchors: np.ndarray, blocks: list[dict]) -> list[str | None]:
    """Map each anchor row to its train/val block label (None if outside)."""
    out = []
    for row, _ in anchors:
        lbl = None
        for blk in blocks:
            if blk["row_start"] <= row < blk["row_end"]:
                lbl = blk["split"]; break
        out.append(lbl)
    return out


def split_by_block(anchors: np.ndarray, labels: list[str | None]):
    """Yield (sub_anchors, label) with no sequence crossing a block boundary."""
    if not len(anchors): return
    cur_lbl, cur_start = labels[0], 0
    for i, lbl in enumerate(labels[1:], 1):
        if lbl != cur_lbl or lbl is None:
            if cur_lbl is not None and (i - cur_start) >= 2:
                yield anchors[cur_start:i], cur_lbl
            cur_start, cur_lbl = i, lbl
    if cur_lbl is not None and (len(anchors) - cur_start) >= 2:
        yield anchors[cur_start:], cur_lbl

# ─────────────────────────── PATCH EXTRACTION ────────────────────────────────

def _crop(ds: rasterio.DatasetReader,
          center_row: float, center_col: float,
          patch_px: int) -> np.ndarray | None:
    """
    Read patch_px × patch_px centred on (center_row, center_col).
    Returns (C, H, W) float32 or None if out of bounds.
    """
    half = patch_px // 2
    r0   = int(round(center_row)) - half
    c0   = int(round(center_col)) - half
    if r0 < 0 or c0 < 0 or r0 + patch_px > ds.height or c0 + patch_px > ds.width:
        return None
    win = rasterio.windows.Window(c0, r0, patch_px, patch_px)
    return ds.read(window=win).astype(np.float32)


def _crop_and_reproject(src_ds: rasterio.DatasetReader,
                        band_indices: list[int],
                        hr_ds: rasterio.DatasetReader,
                        center_row: float, center_col: float,
                        patch_px: int,
                        resampling=Resampling.bilinear) -> np.ndarray | None:
    """
    Reproject band_indices from src_ds into a patch_px × patch_px grid
    aligned to the HR raster at (center_row, center_col).
    Uses the same bounds-based approach as the SISR aux patching script.
    Returns (C, patch_px, patch_px) float32 or None if OOB.
    """
    half = patch_px // 2
    r0   = int(round(center_row)) - half
    c0   = int(round(center_col)) - half
    if r0 < 0 or c0 < 0 or r0 + patch_px > hr_ds.height or c0 + patch_px > hr_ds.width:
        return None

    # derive geographic bounds of this patch from the HR transform
    win     = rasterio.windows.Window(c0, r0, patch_px, patch_px)
    bounds  = rasterio.windows.bounds(win, hr_ds.transform)
    dst_tf  = rasterio.transform.from_bounds(*bounds, patch_px, patch_px)

    n_bands = len(band_indices)
    dst     = np.zeros((n_bands, patch_px, patch_px), dtype=np.float32)
    reproject(
        source      = rasterio.band(src_ds, band_indices),
        destination = dst,
        src_transform = src_ds.transform,
        src_crs       = src_ds.crs,
        dst_transform = dst_tf,
        dst_crs       = hr_ds.crs,
        resampling    = resampling,
    )
    return dst


def extract_aux_patch(aux_ds, cosia_ds, hr_ds,
                      center_row: float, center_col: float,
                      patch_px: int,
                      year: int,
                      dem_stats: dict,
                      campaign: str) -> np.ndarray | None:
    """
    Build a (23, patch_px, patch_px) aux array at the given resolution.
    Mirrors the SISR aux patching logic exactly:
      bands 1-5  → normalised spectral
      bands 6-8  → normalised indices (NDVI, NDWI, NDMI)
      band  10   → normalised DEM
      COSIA lc   → 14-class one-hot
    """
    # ── spectral + indices (bands 1-8 from aux tif) ──
    si = _crop_and_reproject(aux_ds, list(range(1, 9)), hr_ds,
                             center_row, center_col, patch_px,
                             Resampling.bilinear)
    if si is None:
        return None

    # ── DEM (band 10) ──
    dem = _crop_and_reproject(aux_ds, [10], hr_ds,
                              center_row, center_col, patch_px,
                              Resampling.bilinear)
    if dem is None:
        return None

    # ── COSIA landcover ──
    lc_raw = _crop_and_reproject(cosia_ds, [1], hr_ds,
                                 center_row, center_col, patch_px,
                                 Resampling.nearest)
    if lc_raw is None:
        return None

    # ── normalise (same as SISR script) ──
    si[:5] = _norm_spectral(si[:5], year)
    si[5:] = _norm_index(si[5:])
    dem[0] = _norm_dem(dem[0],
                       dem_stats[campaign]["p2"],
                       dem_stats[campaign]["p98"])

    # ── stack: [spectral×5, NDVI, NDWI, NDMI, COSIA×14, DEM] = 23 ch ──
    cosia_oh = _encode_cosia(lc_raw[0].astype(np.int32))
    return np.concatenate([si, cosia_oh, dem], axis=0)  # (23, H, W)


def is_valid(wm_patch: np.ndarray, min_water_frac: float = MIN_WATER_FRAC) -> bool:
    """Reject patch if water fraction in HR water mask is below threshold."""
    return float(wm_patch[0].mean()) >= min_water_frac

# ──────────────────────────────── I/O ────────────────────────────────────────

def _save_frame(seq_dir: Path, frame_idx: int,
                lr: np.ndarray, hr: np.ndarray,
                wm: np.ndarray, aux_lr: np.ndarray | None):
    name = f"{frame_idx:04d}.npy"
    np.save(seq_dir / "LR"  / name, lr.astype(np.float32))
    np.save(seq_dir / "HR"  / name, hr.astype(np.float32))
    np.save(seq_dir / "WM"  / name, wm.astype(np.float32))
    if aux_lr is not None:
        np.save(seq_dir / "AUX_LR" / name, aux_lr.astype(np.float32))


def _make_seq_dir(out_root: Path, split_lbl: str,
                  raster_stem: str, seq_idx: int,
                  save_aux: bool) -> Path:
    seq_dir = out_root / split_lbl / "sequences" / f"{raster_stem}_seq{seq_idx:04d}"
    for sub in ("LR", "HR", "WM"):
        (seq_dir / sub).mkdir(parents=True, exist_ok=True)
    if save_aux:
        (seq_dir / "AUX_LR").mkdir(parents=True, exist_ok=True)
    return seq_dir

# ──────────────────────────── MAIN PIPELINE ──────────────────────────────────

def build_sequences(raster_name: str,
                    split_blocks: list[dict] | None,
                    split_label_override: str | None,
                    out_root: Path,
                    time_meta: dict,
                    dem_stats: dict,
                    all_metadata: dict):
    """
    Process one raster: extract centerline → place anchors → split at
    bends + block boundaries → crop & save LR / HR / WM / AUX_LR frames.

    split_blocks         : from split_map.json  (None for test rasters)
    split_label_override : "test" for test rasters
    time_meta            : { campaign_stem: {lr_time, time_gap_hours, date_gap_days} }
    dem_stats            : { campaign: {p2, p98} }
    all_metadata         : dict updated in-place with per-sequence entries
    """
    stem     = Path(raster_name).stem
    campaign = stem                          # e.g. "BAS_2025"
    year     = int(stem.split("_")[1])

    hr_patch  = HR_PATCH
    lr_patch  = HR_PATCH // SCALE
    stride_hr = int(HR_PATCH * (1 - OVERLAP))   # 128 px

    print(f"\n{'─'*60}")
    print(f"  {raster_name}  split={'test' if split_label_override else 'train/val'}")

    # ── open rasters ──
    missing = []
    for p in [HR_THERMAL_DIR / raster_name,
              LR_THERMAL_DIR / raster_name,
              AUX_DIR        / f"{stem}_aux.tif",
              COSIA_LC_DIR   / f"{stem}_landcover.tif",
              WM_DIR         / f"{stem}_water_mask.tif"]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print(f"  [SKIP] missing files:\n    " + "\n    ".join(missing))
        return

    hr_ds    = rasterio.open(HR_THERMAL_DIR / raster_name)
    lr_ds    = rasterio.open(LR_THERMAL_DIR / raster_name)
    aux_ds   = rasterio.open(AUX_DIR        / f"{stem}_aux.tif")
    cosia_ds = rasterio.open(COSIA_LC_DIR   / f"{stem}_landcover.tif")
    wm_ds    = rasterio.open(WM_DIR         / f"{stem}_water_mask.tif")

    # ── centerline ──
    try:
        centerline = extract_centerline(raster_name)
    except Exception as e:
        print(f"  [SKIP] centerline: {e}")
        for ds in (hr_ds, lr_ds, aux_ds, cosia_ds, wm_ds): ds.close()
        return
    print(f"  Centerline: {len(centerline)} pts")

    anchors = place_anchors(centerline, stride_hr)
    print(f"  Anchors: {len(anchors)}")

    # ── time scalars for this campaign (constant across all frames) ──
    t_meta = time_meta.get(stem, {})
    lr_time        = float(t_meta.get("lr_time",        10.0))
    time_gap_hours = float(t_meta.get("time_gap_hours",  0.0))
    date_gap_days  = float(t_meta.get("date_gap_days",   0.0))

    seq_counters  = {}
    total_frames  = {}

    for seg_anchors in split_at_bends(anchors, BEND_THRESHOLD_DEG):

        if split_label_override is not None:
            groups = [(seg_anchors, split_label_override)]
        else:
            lbls   = label_anchors(seg_anchors, split_blocks)
            groups = list(split_by_block(seg_anchors, lbls))

        for sub_anchors, split_lbl in groups:
            seq_counters.setdefault(split_lbl, 0)
            total_frames.setdefault(split_lbl, 0)

            # buffers for current sequence
            buf_lr, buf_hr, buf_wm, buf_aux = [], [], [], []
            seq_dir = None   # created lazily on first valid frame

            def _flush():
                nonlocal buf_lr, buf_hr, buf_wm, buf_aux, seq_dir
                if len(buf_lr) >= MIN_SEQ_LEN:
                    for fi, (lr_f, hr_f, wm_f, aux_f) in enumerate(
                            zip(buf_lr, buf_hr, buf_wm, buf_aux)):
                        _save_frame(seq_dir, fi, lr_f, hr_f, wm_f, aux_f)
                    # metadata entry for this sequence
                    seq_key = seq_dir.name
                    all_metadata[seq_key] = {
                        "raster":          raster_name,
                        "campaign":        campaign,
                        "split":           split_lbl,
                        "n_frames":        len(buf_lr),
                        "lr_time":         lr_time,
                        "time_gap_hours":  time_gap_hours,
                        "date_gap_days":   date_gap_days,
                    }
                    total_frames[split_lbl]  += len(buf_lr)
                    seq_counters[split_lbl]  += 1
                buf_lr, buf_hr, buf_wm, buf_aux = [], [], [], []
                seq_dir = None

            for row, col in sub_anchors:
                lr_row = row / SCALE
                lr_col = col / SCALE

                # ── HR thermal ──
                hr_p = _crop(hr_ds, row, col, hr_patch)
                if hr_p is None: _flush(); continue

                # ── HR water mask (validity gate) ──
                wm_p = _crop(wm_ds, row, col, hr_patch)
                if wm_p is None or not is_valid(wm_p): _flush(); continue

                # ── LR thermal ──
                lr_p = _crop(lr_ds, lr_row, lr_col, lr_patch)
                if lr_p is None: _flush(); continue

                # ── aux at LR resolution ──
                aux_p = extract_aux_patch(
                    aux_ds, cosia_ds, hr_ds,
                    row, col, lr_patch,       # anchor in HR coords; bounds derived inside
                    year, dem_stats, campaign,
                )
                # aux failure is non-fatal: save None, Tier 3 loader will skip
                # or you can treat it as fatal with: if aux_p is None: _flush(); continue

                # ── create sequence directory lazily on first good frame ──
                if seq_dir is None:
                    seq_dir = _make_seq_dir(
                        out_root, split_lbl, stem,
                        seq_counters[split_lbl],
                        save_aux=(aux_p is not None),
                    )

                buf_lr.append(lr_p)
                buf_hr.append(hr_p)
                buf_wm.append(wm_p)
                buf_aux.append(aux_p)

            _flush()   # end of sub-sequence

    for ds in (hr_ds, lr_ds, aux_ds, cosia_ds, wm_ds):
        ds.close()

    print("  Sequences saved:")
    for spl, cnt in seq_counters.items():
        print(f"    {spl}: {cnt} seqs, {total_frames[spl]} frames")


# ─────────────────────────────── ENTRY POINT ─────────────────────────────────

def main():
    with open(SPLIT_MAP)     as f: split_map  = json.load(f)
    with open(DEM_STATS)     as f: dem_stats  = json.load(f)
    with open(TIME_META_PATH) as f: time_meta = json.load(f)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_metadata: dict = {}

    # ── train / val ──
    for rname in TRAINVAL_RASTERS:
        if rname not in split_map:
            print(f"[WARN] {rname} not in split_map — skipping")
            continue
        build_sequences(
            raster_name          = rname,
            split_blocks         = split_map[rname]["blocks"],
            split_label_override = None,
            out_root             = OUT_ROOT,
            time_meta            = time_meta,
            dem_stats            = dem_stats,
            all_metadata         = all_metadata,
        )

    # ── test ──
    for rname in TEST_RASTERS:
        build_sequences(
            raster_name          = rname,
            split_blocks         = None,
            split_label_override = "test",
            out_root             = OUT_ROOT,
            time_meta            = time_meta,
            dem_stats            = dem_stats,
            all_metadata         = all_metadata,
        )

    # ── write metadata.json ──
    meta_out = OUT_ROOT / "metadata.json"
    with open(meta_out, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\nMetadata written → {meta_out}  ({len(all_metadata)} sequences)")

    # ── summary ──
    print("\nFinal counts:")
    for split in ("train", "val", "test"):
        root = OUT_ROOT / split / "sequences"
        if not root.exists(): continue
        seqs   = list(root.glob("*"))
        frames = sum(len(list((s / "HR").glob("*.npy"))) for s in seqs)
        print(f"  {split:5s}: {len(seqs):4d} sequences  {frames:6d} frames")


if __name__ == "__main__":
    main()