"""
Plant-count pipeline for clean drone captures (v2 - consolidated).

Design
------
* RGB is registered to the MS grid; plastic mulch rows are segmented there with an
  ORIENTATION-AWARE pipeline (rows may be horizontal, vertical or diagonal).
* Plant detection runs at NATIVE RGB resolution (GSD ~0.7 cm) using RGB green cues
  (ExG + G>R + G>B) restricted to the dilated row mask; NDVI/NDRE (MS grid) are only
  a LOOSE sanity gate so tiny seedlings are not erased.
* Centroids are mapped native->MS with a homography for lat/lon georeferencing.

* After all images in a folder are processed, duplicates are removed with
  dedupe_plants.py (same folder): close double circles inside one image, and the
  same plant seen in several overlapping images (matched by GPS after aligning
  consecutive frames).

Outputs per folder under <folder>/output/:
  plant_count_batch.csv        per image: raw detections + plants counted in it
  plant_instances_batch.csv    every raw detection
  plants_unique.csv            one row per unique plant  (= field plant count)
  plants_detection_map.csv     every detection -> unique plant + duplicate type
  frame_alignment.csv          GPS shift applied per image (quality check)
  plant_map.html               interactive map of plants_unique.csv (double-click)
  final_overlays/<stem>_final.jpg  (farm boundary corners marked in orange)
                               filtered evidence images: yellow = counted in this
                               image, magenta = already counted in an earlier image
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:  # hide harmless "Unknown field with tag 40092" warnings from DJI TIFF metadata
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "plant-count-config.json"

_spec = importlib.util.spec_from_file_location("ip", ROOT / "image-processing.py")
ip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ip)

_dd_spec = importlib.util.spec_from_file_location("dedupe_plants", ROOT / "dedupe_plants.py")
dd = importlib.util.module_from_spec(_dd_spec)
_dd_spec.loader.exec_module(dd)

_pm_spec = importlib.util.spec_from_file_location("plant_map", ROOT / "plant_map.py")
pm = importlib.util.module_from_spec(_pm_spec)
_pm_spec.loader.exec_module(pm)


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
def load_config(path: Path | None) -> dict:
    cfg_path = path or DEFAULT_CONFIG
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    defaults = {
        # plastic rows (segmented on registered RGB, working width 1600)
        "plastic_working_width": 1600,
        "plastic_min_area": 300,
        "plastic_min_aspect": 1.5,
        "row_bridge_kernel_px": 51,
        "plastic_v_min": "auto",
        "plastic_s_max": 60,
        "plant_exg_min": 30,          # ExG veto inside plastic mask
        # native-RGB plant detection
        "native_exg_min": 20,
        "native_ndvi_min": 0.10,      # loose gate: kills tags/white specks only
        "native_ndre_min": 0.03,
        "native_dilate_px": 9,
        # plant size / counting
        "min_plant_diameter_cm": 3,
        "max_plant_diameter_cm": 60,
        "min_plant_area_px": 6,
        "watershed_fg_fraction": 0.45,
        # dark-spot (planting hole) detection for tiny seedlings
        "dark_spot_detection": True,
        "hole_diameter_cm": 8,
        "dark_spot_min_contrast": 30,
        "dark_spot_edge_margin": 0.8,
        "dark_spot_max_size": 3.0,
        "dark_spot_min_separation": 1.0,
        "dark_spot_require_green": False,
        "dark_spot_exg_min": 5,
        "dark_spot_ndvi_gate": False,
        # row-geometry constraints
        "row_constraints": True,
        "row_center_band_frac": 0.25,     # plant within +/- this x row width of centre
        "row_min_length_frac": 0.25,      # row piece must be >= this x longest row
        "row_gap_split_frac": 0.04,       # gap (x image size) that splits a row piece
        "row_profile_threshold": 0.3,
        "plant_spacing_cm": None,         # None = measured from the rows
        "plant_spacing_min_cm": 8,
        "plant_spacing_max_cm": 80,
        "crowd_min_gap_frac": 0.6,        # min gap between plants (x spacing)
        # farm boundary [[lat, lon], ...] around the perimeter. Plants outside are
        # drawn red and not counted. A file <folder>/farm_boundary.json overrides it.
        "farm_boundary": None,
        # de-duplication
        "gps_rtk": False,                 # True = trust GPS, skip frame alignment
        "dedupe_spacing_m": None,         # plant spacing; None = measured automatically
        "dedupe_search_radius_m": 1.0,    # max GPS shift between consecutive images
        "dedupe_min_overlap": 15,
        "hide_repeats_in_images": False,  # True = don't draw magenta repeats
        "keep_base_images": False,        # keep output/_base (temporary files)
        "write_overlay": True,
        "write_instance_csv": True,
    }
    defaults.update(cfg)
    return defaults


# ----------------------------------------------------------------------------
# plastic row segmentation (orientation-aware B/S/V + morphology)
# ----------------------------------------------------------------------------
def estimate_row_angle_deg(mask_u8: np.ndarray, min_area: int = 400) -> float:
    """Dominant row direction (deg, 0 = horizontal) via PCA of elongated components."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    angs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.stack([xs, ys], 1).astype(np.float64)
        ev = np.sort(np.linalg.eigvalsh(np.cov(pts, rowvar=False)))   # [small, large]
        if ev[0] > 1e-6 and ev[1] / ev[0] < 3.0:
            continue                                                  # blob-like, skip
        _, evec = cv2.PCACompute(pts, mean=None)   # 2 outputs on all OpenCV versions
        a = math.degrees(math.atan2(float(evec[0][1]), float(evec[0][0]))) % 180.0
        if a > 90.0:
            a -= 180.0
        angs.append(a)
    if not angs:
        return 0.0
    r = np.radians(np.asarray(angs) * 2.0)         # circular mean mod 180
    return float(math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean())) / 2.0)


def segment_plastic_rows(rgb: np.ndarray, cfg: dict):
    """Detect plastic mulch rows (any orientation). Returns (mask uint8, snr)."""
    h0, w0 = rgb.shape[:2]
    scale = int(cfg.get("plastic_working_width", 1600)) / w0
    img = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sm = cv2.bilateralFilter(img, 9, 50, 50)
    b_ch, g_ch, r_ch = cv2.split(sm)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)
    gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)

    _, m_b = cv2.threshold(b_ch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # brightness threshold: "auto" adapts to the batch (cloudy days, darker
    # exposure, light fall-off across the frame); a number forces a fixed value
    v_cfg = cfg.get("plastic_v_min", "auto")
    if isinstance(v_cfg, str) and v_cfg.lower() == "auto":
        # flatten slow brightness changes across the frame first
        bgk = max(31, (min(v_ch.shape) // 8) | 1)
        v_bg = cv2.GaussianBlur(cv2.dilate(v_ch, np.ones((bgk // 2, bgk // 2), np.uint8)), (bgk, bgk), 0)
        v_norm = np.clip(v_ch.astype(np.float32) / np.maximum(v_bg, 1) * 200, 0, 255).astype(np.uint8)
        v_thr, _ = cv2.threshold(v_norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        v_ok = v_norm > max(v_thr, 60)
        print(f"  plastic brightness: auto (Otsu {v_thr:.0f} on normalised V)")
    else:
        v_ok = v_ch > float(v_cfg)
    s_max = int(cfg.get("plastic_s_max", 60))
    mask = m_b & (s_ch < s_max).astype(np.uint8) * 255 & v_ok.astype(np.uint8) * 255
    exg = 2 * g_ch.astype(np.int16) - r_ch.astype(np.int16) - b_ch.astype(np.int16)
    mask[exg > int(cfg.get("plant_exg_min", 30))] = 0

    # rotate rows to horizontal -> morphology/aspect filter orientation-invariant
    angle = estimate_row_angle_deg(mask)
    print(f"  row angle={angle:.1f} deg")
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    W2, H2 = int(h * sin_a + w * cos_a), int(h * cos_a + w * sin_a)
    M[0, 2] += W2 / 2 - w / 2
    M[1, 2] += H2 / 2 - h / 2
    mask_r = cv2.warpAffine(mask, M, (W2, H2), flags=cv2.INTER_LINEAR)
    mask_r = (mask_r > 127).astype(np.uint8) * 255

    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    row_len = int(cfg.get("row_bridge_kernel_px", 51))   # bridge canopy-sized gaps
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (row_len, 3)))
    cnts, _ = cv2.findContours(mask_r, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_r, cnts, -1, 255, cv2.FILLED)  # fill crown/splash holes
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))

    min_area = int(cfg.get("plastic_min_area", 300))
    min_aspect = float(cfg.get("plastic_min_aspect", 1.5))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_r, 8)
    clean_r = np.zeros_like(mask_r)
    for i in range(1, n):
        _, _, bw, bh, area = stats[i]
        if area < min_area or bw / max(bh, 1) < min_aspect:
            continue
        clean_r[lab == i] = 255

    clean = cv2.warpAffine(clean_r, M, (w, h),
                           flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)

    fg = gray[clean == 255].astype(np.float64)
    bg = gray[clean == 0].astype(np.float64)
    snr = 0.0
    if fg.size and bg.size:
        snr = float((fg.mean() - bg.mean()) / np.hypot(fg.std() + 1e-6, bg.std() + 1e-6))
        print(f"  plastic mu={fg.mean():.0f} sig={fg.std():.0f} | "
              f"bg mu={bg.mean():.0f} sig={bg.std():.0f} | SNR={snr:.2f}")
    full = cv2.resize(clean, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return full, snr


# ----------------------------------------------------------------------------
# native RGB <-> MS mapping
# ----------------------------------------------------------------------------
def estimate_h_rgb_to_ms(rgb: np.ndarray, ms_ref: np.ndarray) -> np.ndarray:
    """3x3 homography native-RGB px -> MS px (ORB+RANSAC on gradient magnitude)."""
    s1, s2 = 1600.0 / rgb.shape[1], 1600.0 / ms_ref.shape[1]
    a = cv2.resize(rgb, None, fx=s1, fy=s1, interpolation=cv2.INTER_AREA)
    b = cv2.resize(ms_ref, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA)

    def grad_u8(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        g = np.float32(g)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
        m = cv2.GaussianBlur(cv2.magnitude(gx, gy), (5, 5), 0)
        return cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    orb = cv2.ORB_create(20000)
    k1, d1 = orb.detectAndCompute(grad_u8(b), None)
    k2, d2 = orb.detectAndCompute(grad_u8(a), None)
    H = np.eye(3)
    if d1 is not None and d2 is not None:
        mm = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, 2)
        good = [m[0] for m in mm if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        print(f"  H rgb->ms inliers={len(good)}")
        if len(good) >= 8:
            src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            Hs, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if Hs is not None:
                H = Hs.astype(np.float64)
    return np.diag([1 / s2, 1 / s2, 1.0]) @ H @ np.diag([s1, s1, 1.0])


# ----------------------------------------------------------------------------
# plant detection at native RGB resolution
# ----------------------------------------------------------------------------
def excess_green(rgb: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(rgb)
    return 2 * g.astype(np.int16) - r.astype(np.int16) - b.astype(np.int16)


def plant_area_bounds(gsd_m: float, cfg: dict):
    min_d = cfg["min_plant_diameter_cm"] / 100.0
    max_d = cfg["max_plant_diameter_cm"] / 100.0
    min_a_cfg = int(cfg.get("min_plant_area_px", 0))
    min_a = max(min_a_cfg, int(math.pi * (min_d / gsd_m / 2) ** 2 * 0.3))
    max_a = int(math.pi * (max_d / gsd_m / 2) ** 2)
    return min_a, max_a


def region_centroid(region: np.ndarray):
    ys, xs = np.where(region)
    return float(xs.mean()), float(ys.mean())


def blob_passes(region, ndvi, ndre, ndvi_min, ndre_min) -> bool:
    if region.sum() == 0:
        return False
    return (float(ndvi[region].mean()) >= ndvi_min
            and float(ndre[region].mean()) >= ndre_min)


def instance_from_region(region, ndvi, ndre, method):
    cx, cy = region_centroid(region)
    return {
        "centroid_x": cx,
        "centroid_y": cy,
        "area_px": int(region.sum()),
        "ndvi_mean": float(ndvi[region].mean()),
        "ndre_mean": float(ndre[region].mean()),
        "method": method,
    }


def split_large_blob(comp_mask, ndvi, ndre, min_a, max_a, fg_fraction,
                     ndvi_min, ndre_min) -> list:
    """Watershed split for merged crowns (distance-transform sure foreground)."""
    comp_u8 = comp_mask.astype(np.uint8) * 255
    dist = cv2.distanceTransform(comp_u8, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []
    _, sure_fg = cv2.threshold(dist, fg_fraction * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(comp_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers.astype(np.int32) + 1
    markers[unknown == 255] = 0
    ws_img = cv2.cvtColor(comp_u8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(ws_img, markers)
    instances = []
    for lbl in range(2, n_markers + 2):
        region = (markers == lbl) & comp_mask
        area = int(region.sum())
        if area < min_a or area > max_a:
            continue
        if blob_passes(region, ndvi, ndre, ndvi_min, ndre_min):
            instances.append(instance_from_region(region, ndvi, ndre, "watershed_split"))
    if not instances and blob_passes(comp_mask, ndvi, ndre, ndvi_min, ndre_min):
        est = max(1, int(round(comp_mask.sum() / max(min_a, 1))))
        est = min(est, 30)
        instances.append(instance_from_region(comp_mask, ndvi, ndre, f"area_estimate_x{est}"))
    return instances


def detect_dark_spots(rgb, plast_core, exg, ndvi_nat, ndre_nat, gsd_nat,
                      existing, cfg) -> list:
    """Find seedlings as small dark spots (planting holes) on the bright plastic.

    Tiny seedlings are only a few pixels of green inside a dark hole, so the ExG
    test alone misses most of them. A black-hat filter sized to the hole picks out
    every spot that is darker than the plastic around it.
    """
    hole_px = cfg.get("hole_diameter_cm", 5) / 100.0 / max(gsd_nat, 1e-6)
    hole_px = float(np.clip(hole_px, 3, 200))
    max_size = float(cfg.get("dark_spot_max_size", 3.0))
    k = int(round(1.5 * max_size * hole_px)) | 1      # kernel must exceed biggest hole
    k = max(k, 9)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    # stay off the row edges (plastic/soil border also looks like a dark "spot")
    e = max(3, int(round(hole_px * float(cfg.get("dark_spot_edge_margin", 0.8)))))
    core = cv2.erode(plast_core.astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
    thr = int(cfg.get("dark_spot_min_contrast", 30))
    m = ((bh > thr) & (core > 0)).astype(np.uint8) * 255

    min_a = math.pi * (0.3 * hole_px / 2) ** 2
    max_a = math.pi * (max_size * hole_px / 2) ** 2
    require_green = bool(cfg.get("dark_spot_require_green", False))
    exg_lo = int(cfg.get("dark_spot_exg_min", 5))
    use_gate = bool(cfg.get("dark_spot_ndvi_gate", False))
    ndvi_lo = float(cfg.get("native_ndvi_min", 0.10))
    ndre_lo = float(cfg.get("native_ndre_min", 0.03))
    min_sep = hole_px * float(cfg.get("dark_spot_min_separation", 1.0))

    taken = [(i["centroid_x"], i["centroid_y"]) for i in existing]
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bhh, area = stats[i]
        if area < min_a or area > max_a:
            continue
        if max(bw, bhh) / max(1, min(bw, bhh)) > 3.0:          # streaks/creases
            continue
        cx, cy = float(cents[i][0]), float(cents[i][1])
        if (cx < k / 2 or cy < k / 2 or cx > rgb.shape[1] - k / 2
                or cy > rgb.shape[0] - k / 2):
            continue                                           # image border artefacts
        if any((cx - tx) ** 2 + (cy - ty) ** 2 < min_sep ** 2 for tx, ty in taken):
            continue                                           # already counted
        sl = (slice(y, y + bhh), slice(x, x + bw))
        region = lab[sl] == i
        green_px = int((exg[sl][region] > exg_lo).sum())
        if require_green and green_px == 0:
            continue
        nv = float(ndvi_nat[sl][region].mean())
        nr = float(ndre_nat[sl][region].mean())
        if use_gate and (nv < ndvi_lo or nr < ndre_lo):
            continue
        taken.append((cx, cy))
        out.append({"centroid_x": cx, "centroid_y": cy, "area_px": int(area),
                    "ndvi_mean": nv, "ndre_mean": nr, "method": "dark_spot",
                    "score": float(bh[sl][region].mean())})
    return out


# ----------------------------------------------------------------------------
# row-geometry constraints
# ----------------------------------------------------------------------------
def _row_angle(xs, ys):
    """Direction of the rows: the angle whose perpendicular profile is sharpest."""
    def sharp(th):
        v = -xs * math.sin(th) + ys * math.cos(th)
        hst = np.bincount((v - v.min()).astype(np.int64))
        return float((hst.astype(np.float64) ** 2).sum())
    coarse = np.radians(np.arange(-90, 90, 1.0))
    best = coarse[int(np.argmax([sharp(t) for t in coarse]))]
    fine = best + np.radians(np.arange(-1.0, 1.01, 0.1))
    return float(fine[int(np.argmax([sharp(t) for t in fine]))])


def _spacing_from_rows(rows_u, min_lag, max_lag):
    """Plant spacing (working px) from the summed autocorrelation of all rows."""
    max_lag = int(max_lag)
    if max_lag <= min_lag + 2:
        return None
    acc = np.zeros(max_lag + 1)
    n = 0
    for u in rows_u:
        if len(u) < 4:
            continue
        sig = np.bincount((u - u.min()).astype(np.int64)).astype(np.float64)
        sig = np.convolve(sig, np.array([0.25, 0.5, 1.0, 0.5, 0.25]), mode="same")
        f = np.fft.rfft(sig, 2 * len(sig) + max_lag)
        ac = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
        acc[: len(ac)] += ac
        n += len(u)
    if n < 12:
        return None
    seg = acc[int(min_lag):]
    if seg.max() <= 0:
        return None
    peaks = [i for i in range(1, len(seg) - 1)
             if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] >= 0.5 * seg.max()]
    if not peaks:
        return None
    return float(peaks[0] + int(min_lag))       # first strong peak = base spacing


def apply_row_constraints(instances, plast_core, gsd_nat, cfg):
    """
    Keep only plants that sit like real crop on mulch:
      * inside a proper plastic row (long, straight, normal width) - short strips
        and plastic bits outside the field are discarded
      * near the row centre line (row_center_band_frac of the row width)
      * at most one plant per planting position (min gap = fraction of the
        measured plant spacing) - removes crowded double / triple circles
    Returns (kept_instances, valid_row_mask_native, info).
    """
    info = {"rows_found": 0, "plant_spacing_cm": "", "rejected_outside_rows": 0,
            "rejected_off_center": 0, "rejected_crowded": 0}
    h, w = plast_core.shape[:2]
    if not instances:
        return instances, plast_core, info
    sc = min(1.0, float(cfg.get("row_model_width", 1600)) / w)
    m = cv2.resize(plast_core.astype(np.uint8), (max(1, int(w * sc)), max(1, int(h * sc))),
                   interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(m)
    if len(xs) < 500:
        return instances, plast_core, info
    xs = xs.astype(np.float64); ys = ys.astype(np.float64)
    pick = np.random.default_rng(0).choice(len(xs), min(len(xs), 60000), replace=False)
    th = _row_angle(xs[pick], ys[pick])
    ct, st = math.cos(th), math.sin(th)
    U = xs * ct + ys * st                     # along the row
    V = -xs * st + ys * ct                    # across the row
    v0 = V.min()

    # --- row bands from the across-row profile
    prof = np.bincount((V - v0).astype(np.int64)).astype(np.float64)
    prof = np.convolve(prof, np.ones(3) / 3, mode="same")
    thr = float(cfg.get("row_profile_threshold", 0.3)) * np.percentile(prof[prof > 0], 95)
    on = prof > thr
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and (on[j + 1] or (j + 2 < len(on) and on[j + 2])):
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    if not runs:
        return instances, plast_core, info
    widths = np.array([b - a + 1 for a, b in runs])
    med_w = float(np.median(widths))
    bands = []
    for (a, b), wd in zip(runs, widths):
        if wd < 0.4 * med_w:
            continue                               # slivers
        k = int(round(wd / med_w)) if wd > 1.8 * med_w else 1
        step = (b - a + 1) / k                     # touching rows -> split
        for q in range(k):
            bands.append((a + q * step, a + (q + 1) * step - 1))

    # --- per row: straight centre line, width, continuous segments along the row
    gap_split = float(cfg.get("row_gap_split_frac", 0.04)) * max(m.shape)
    rows = []
    for a, b in bands:
        sel = (V - v0 >= a) & (V - v0 <= b)
        if sel.sum() < 50:
            continue
        u, v = U[sel], V[sel]
        nb = max(4, int((u.max() - u.min()) / 20))
        edges = np.linspace(u.min(), u.max() + 1e-6, nb + 1)
        bi = np.digitize(u, edges) - 1
        cu, cv, wd = [], [], []
        for k in range(nb):
            vv = v[bi == k]
            if len(vv) >= 5:
                cu.append(u[bi == k].mean())
                cv.append((np.percentile(vv, 3) + np.percentile(vv, 97)) / 2)
                wd.append(np.percentile(vv, 97) - np.percentile(vv, 3))
        if len(cu) < 2:
            continue
        coef = np.polyfit(cu, cv, 1)               # rows are straight lines
        width = float(np.median(wd))
        occ = np.unique(np.round(u).astype(np.int64))
        segs, s0 = [], occ[0]
        for p0, p1 in zip(occ[:-1], occ[1:]):
            if p1 - p0 > gap_split:
                segs.append((s0, p0)); s0 = p1
        segs.append((s0, occ[-1]))
        rows.append({"coef": coef, "width": width, "segs": segs})
    if not rows:
        return instances, plast_core, info
    longest = max(s1 - s0 for r in rows for s0, s1 in r["segs"])
    min_len = float(cfg.get("row_min_length_frac", 0.25)) * longest
    ok_w = float(np.median([r["width"] for r in rows]))
    rows = [r for r in rows if 0.5 * ok_w <= r["width"] <= 1.8 * ok_w]
    for r in rows:
        r["segs"] = [sg for sg in r["segs"] if sg[1] - sg[0] >= min_len]
    rows = [r for r in rows if r["segs"]]
    info["rows_found"] = len(rows)
    if not rows:
        return [], np.zeros_like(plast_core), info

    # --- assign plants to rows, centre-band test
    band = float(cfg.get("row_center_band_frac", 0.25))
    per_row = defaultdict(list)
    for inst in instances:
        x, y = inst["centroid_x"] * sc, inst["centroid_y"] * sc
        u, v = x * ct + y * st, -x * st + y * ct
        best = None
        for ri, r in enumerate(rows):
            if not any(s0 - 0.5 * r["width"] <= u <= s1 + 0.5 * r["width"] for s0, s1 in r["segs"]):
                continue
            off = v - np.polyval(r["coef"], u)
            if best is None or abs(off) < abs(best[1]):
                best = (ri, off)
        if best is None or abs(best[1]) > rows[best[0]]["width"]:
            info["rejected_outside_rows"] += 1
            continue
        ri, off = best
        rel = off / rows[ri]["width"]
        if abs(rel) > band:
            info["rejected_off_center"] += 1
            continue
        inst["row_id"] = ri
        inst["row_offset_frac"] = round(float(rel), 3)
        inst["_u"] = u
        per_row[ri].append(inst)

    # --- plant spacing + one plant per planting position
    gsd_w = gsd_nat / sc
    sp_cfg = cfg.get("plant_spacing_cm")
    if sp_cfg:
        spacing = float(sp_cfg) / 100.0 / gsd_w
    else:
        spacing = _spacing_from_rows(
            [np.array([i["_u"] for i in v]) for v in per_row.values()],
            float(cfg.get("plant_spacing_min_cm", 8)) / 100.0 / gsd_w,
            float(cfg.get("plant_spacing_max_cm", 80)) / 100.0 / gsd_w)
    if spacing:
        info["plant_spacing_cm"] = round(spacing * gsd_w * 100, 1)
    min_gap = float(cfg.get("crowd_min_gap_frac", 0.6)) * spacing if spacing else 0.0
    kept = []
    for ri, lst in per_row.items():
        # strongest + most central first
        lst.sort(key=lambda i: -i.get("score", 100.0) * (1 - abs(i["row_offset_frac"]) / max(band, 1e-6) * 0.5))
        acc_u = []
        for inst in lst:
            if min_gap and any(abs(inst["_u"] - au) < min_gap for au in acc_u):
                info["rejected_crowded"] += 1
                continue
            acc_u.append(inst["_u"])
            kept.append(inst)
    for inst in kept:
        inst.pop("_u", None)

    # --- mask of valid rows (for the overlay tint)
    gy, gx = np.mgrid[0:m.shape[0], 0:m.shape[1]].astype(np.float32)
    GU, GV = gx * ct + gy * st, -gx * st + gy * ct
    valid = np.zeros(m.shape, bool)
    for r in rows:
        near = np.abs(GV - np.polyval(r["coef"], GU)) <= 0.75 * r["width"]
        inseg = np.zeros(m.shape, bool)
        for s0, s1 in r["segs"]:
            inseg |= (GU >= s0) & (GU <= s1)
        valid |= near & inseg
    valid = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return kept, plast_core & valid, info


def count_plants_native(rgb, plastic_ms, H, ndvi, ndre, gsd_ms, cfg):
    """Plant detection on native RGB inside row mask. Returns (instances, plast_nat, meta)."""
    h, w = rgb.shape[:2]
    gsd_nat = gsd_ms / (w / ndvi.shape[1])

    plast = cv2.warpPerspective(plastic_ms, H, (w, h),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    plast_core = plast.copy()
    d = int(cfg.get("native_dilate_px", 9))
    if d > 0:
        plast = cv2.dilate(plast.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (2 * d + 1, 2 * d + 1))) > 0
    b, g, r = cv2.split(rgb)
    exg = excess_green(rgb)
    exg_min = int(cfg.get("native_exg_min", 20))
    green = ((exg > exg_min)
             & (g.astype(np.int16) > r.astype(np.int16))
             & (g.astype(np.int16) > b.astype(np.int16) + 5))   # kills yellow tags
    mask = (plast & green).astype(np.uint8) * 255               # NO morph-open
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # MS and RGB have different fields of view, so a plain resize samples the
    # wrong pixels (plastic instead of plant) -> use the same homography as the mask
    ndvi_nat = cv2.warpPerspective(ndvi.astype(np.float32), H, (w, h),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    ndre_nat = cv2.warpPerspective(ndre.astype(np.float32), H, (w, h),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    min_a, max_a = plant_area_bounds(gsd_nat, cfg)
    fg_frac = float(cfg.get("watershed_fg_fraction", 0.45))
    ndvi_lo = float(cfg.get("native_ndvi_min", 0.10))
    ndre_lo = float(cfg.get("native_ndre_min", 0.03))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    instances = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_a:
            continue
        comp = labels == i
        if area > max_a:
            instances.extend(split_large_blob(comp, ndvi_nat, ndre_nat, min_a, max_a,
                                              fg_frac, ndvi_lo, ndre_lo))
            continue
        # merged-crown check even below max_a (>=2 distance-transform peaks)
        dist = cv2.distanceTransform(comp.astype(np.uint8) * 255, cv2.DIST_L2, 5)
        _, sfg = cv2.threshold(dist, fg_frac * max(dist.max(), 1e-6), 255, 0)
        peaks = cv2.connectedComponents(sfg.astype(np.uint8))[0] - 1
        if peaks >= 2:
            instances.extend(split_large_blob(comp, ndvi_nat, ndre_nat, min_a, max_a,
                                              fg_frac, ndvi_lo, ndre_lo))
        elif blob_passes(comp, ndvi_nat, ndre_nat, ndvi_lo, ndre_lo):
            instances.append(instance_from_region(comp, ndvi_nat, ndre_nat, "blob"))

    if cfg.get("dark_spot_detection", True):
        instances.extend(detect_dark_spots(rgb, plast_core, exg, ndvi_nat, ndre_nat,
                                           gsd_nat, instances, cfg))

    row_info = {}
    if cfg.get("row_constraints", True):
        instances, row_mask, row_info = apply_row_constraints(instances, plast_core,
                                                              gsd_nat, cfg)
        plast = row_mask        # overlay tint = valid rows only

    for inst in instances:                     # native -> MS px for georeferencing
        p = H @ np.array([inst["centroid_x"], inst["centroid_y"], 1.0])
        inst["ms_x"] = float(p[0] / p[2])
        inst["ms_y"] = float(p[1] / p[2])

    meta = {
        "candidate_pixels": int((mask > 0).sum()),
        "ndvi_min": ndvi_lo,
        "ndre_min": ndre_lo,
        "note": f"native exg>{exg_min} ndvi>{ndvi_lo:.2f} ndre>{ndre_lo:.2f}",
        **row_info,
    }
    return instances, plast.astype(np.uint8) * 255, meta


# ----------------------------------------------------------------------------
# overlay / per-image processing
# ----------------------------------------------------------------------------
def draw_overlay(rgb, plastic_u8, instances, plant_count, radius=10):
    vis = rgb.copy()
    tint = vis.copy()
    tint[plastic_u8 > 0] = (0, 255, 0)
    cv2.addWeighted(tint, 0.25, vis, 0.75, 0, vis)
    for idx, inst in enumerate(instances, 1):
        x = int(round(inst["centroid_x"]))
        y = int(round(inst["centroid_y"]))
        color = (0, 255, 255) if inst["method"] == "dark_spot" else (0, 0, 255)
        cv2.circle(vis, (x, y), radius, color, 2)
        cv2.putText(vis, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    if plant_count is not None:
        cv2.putText(vis, f"plants={plant_count}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def process_one(stem, jpg_path, bands, out_dir, cfg, write_images=True):
    rgb = cv2.imread(str(jpg_path))
    nir = cv2.imread(str(bands["NIR"]), cv2.IMREAD_UNCHANGED)
    red = cv2.imread(str(bands["R"]), cv2.IMREAD_UNCHANGED)
    red_edge = cv2.imread(str(bands["RE"]), cv2.IMREAD_UNCHANGED)
    if rgb is None or nir is None or red is None or red_edge is None:
        raise RuntimeError(f"Failed to read images for {stem}")

    geo = ip.georef_from_ms(bands["NIR"], nir.shape)
    for w_ in geo.get("warnings", []):
        print(f"  GPS WARNING: {w_}")
    rgb_reg, aux = ip.register_rgb_to_ms(rgb, nir, refine_band=red)
    if isinstance(aux, np.ndarray) and aux.shape == (3, 3):
        # register_rgb_to_ms returns H for the RGB *resized to MS size*;
        # prepend the native->MS-size scale so H maps native RGB px -> MS px
        sx = nir.shape[1] / rgb.shape[1]
        sy = nir.shape[0] / rgb.shape[0]
        H = aux @ np.diag([sx, sy, 1.0])
    else:
        H = estimate_h_rgb_to_ms(rgb, red)

    plastic_u8, plastic_snr = segment_plastic_rows(rgb_reg, cfg)
    ndvi = ip.compute_ndvi(red, nir)
    ndre = ip.compute_ndre(red_edge, nir)

    status = "ok"
    if int((plastic_u8 > 0).sum()) < 100:
        instances, plast_nat = [], np.zeros(rgb.shape[:2], np.uint8)
        meta = {"candidate_pixels": 0, "ndvi_min": 0.0, "ndre_min": 0.0, "note": ""}
        status = "no_plastic"
    else:
        instances, plast_nat, meta = count_plants_native(
            rgb, plastic_u8, H, ndvi, ndre, geo["gsd_m"], cfg)

    plant_rows = []
    for idx, inst in enumerate(instances, 1):
        lat, lon = ip.pixel_to_latlon(
            inst["ms_x"], inst["ms_y"], geo["lat"], geo["lon"],
            geo["yaw_deg"], geo["gsd_m"], geo["cx"], geo["cy"])
        plant_rows.append({
            "image_id": stem,
            "plant_id": idx,
            "lat": lat,
            "lon": lon,
            "centroid_x_px": round(inst["ms_x"], 2),
            "centroid_y_px": round(inst["ms_y"], 2),
            "rgb_x_px": round(inst["centroid_x"], 1),
            "rgb_y_px": round(inst["centroid_y"], 1),
            "rgb_width_px": rgb.shape[1],
            "area_px": inst["area_px"],
            "ndvi_mean": round(inst["ndvi_mean"], 4),
            "ndre_mean": round(inst["ndre_mean"], 4),
            "detection_method": inst["method"],
            "row_id": inst.get("row_id", ""),
            "row_offset_frac": inst.get("row_offset_frac", ""),
            "image_lat": geo["lat"],
            "image_lon": geo["lon"],
            "gsd_m": geo["gsd_m"],
            "yaw_deg": geo["yaw_deg"],
        })

    summary = {
        "image_id": stem,
        "plant_count": len(instances),
        "status": status,
        "threshold_note": meta["note"],
        "plastic_snr": round(plastic_snr, 3),
        "plastic_pixels": int((plastic_u8 > 0).sum()),
        "candidate_pixels": meta.get("candidate_pixels", 0),
        "ndvi_threshold": meta.get("ndvi_min", cfg["native_ndvi_min"]),
        "ndre_threshold": meta.get("ndre_min", cfg["native_ndre_min"]),
        "count_blob": sum(1 for i in instances if i["method"] == "blob"),
        "count_watershed_split": sum(1 for i in instances if i["method"] == "watershed_split"),
        "count_area_estimate": sum(1 for i in instances if i["method"].startswith("area_estimate")),
        "count_dark_spot": sum(1 for i in instances if i["method"] == "dark_spot"),
        "rows_found": meta.get("rows_found", ""),
        "plant_spacing_cm": meta.get("plant_spacing_cm", ""),
        "rejected_outside_rows": meta.get("rejected_outside_rows", ""),
        "rejected_off_center": meta.get("rejected_off_center", ""),
        "rejected_crowded": meta.get("rejected_crowded", ""),
        "image_lat": geo["lat"],
        "image_lon": geo["lon"],
        "gsd_m": geo["gsd_m"],
        "yaw_deg": geo["yaw_deg"],
        "gimbal_pitch_deg": geo.get("pitch_deg", ""),
        "gimbal_yaw_raw_deg": geo.get("gimbal_yaw_deg", ""),
        "yaw_source": geo.get("yaw_source", ""),
        "rtk_flag": geo.get("rtk_flag", ""),
        "altitude_m": geo.get("alt_m", ""),
        "capture_utc": geo.get("capture_utc", ""),
    }

    if write_images and cfg.get("write_overlay", True):
        # clean copy (row tint only, no circles); the filtered circles are drawn
        # on it after de-duplication, then this temporary copy is deleted
        base = draw_overlay(rgb, plast_nat, [], None)
        if base.shape[1] > 2600:
            base = cv2.resize(base, (2600, int(base.shape[0] * 2600 / base.shape[1])),
                              interpolation=cv2.INTER_AREA)
        (out_dir / "_base").mkdir(exist_ok=True)
        cv2.imwrite(str(out_dir / "_base" / f"{stem}.jpg"), base)
    # pixel -> GPS samples on a grid (exact georef for de-duplication alignment)
    summary["_px2ll"] = []
    for gx in np.linspace(0, rgb.shape[1] - 1, 5):
        for gy in np.linspace(0, rgb.shape[0] - 1, 5):
            q = H @ np.array([gx, gy, 1.0])
            la, lo = ip.pixel_to_latlon(q[0] / q[2], q[1] / q[2], geo["lat"], geo["lon"],
                                        geo["yaw_deg"], geo["gsd_m"], geo["cx"], geo["cy"])
            summary["_px2ll"].append((float(gx), float(gy), la, lo))

    # boundary points GPS -> native RGB px (drawn on the final image)
    summary["_boundary_px"] = []
    if cfg.get("_boundary"):
        try:
            Hi = np.linalg.inv(H)
            for (lat, lon), name in zip(cfg["_boundary"], cfg["_boundary_names"]):
                col, row = ip.latlon_to_pixel(lat, lon, geo["lat"], geo["lon"], geo["yaw_deg"],
                                              geo["gsd_m"], geo["cx"], geo["cy"])
                q = Hi @ np.array([col, row, 1.0])
                summary["_boundary_px"].append((name, q[0] / q[2], q[1] / q[2]))
        except Exception as exc:
            print(f"  boundary projection failed: {exc}")
    return summary, plant_rows


# ----------------------------------------------------------------------------
# batch runner
# ----------------------------------------------------------------------------
SUMMARY_FIELDS = ["image_id", "detections_raw", "new_plants", "repeat_plants",
                  "outside_farm", "running_total", "status", "threshold_note", "plastic_snr",
                  "plastic_pixels", "candidate_pixels", "ndvi_threshold", "ndre_threshold",
                  "count_blob", "count_watershed_split", "count_area_estimate", "count_dark_spot",
                  "rows_found", "plant_spacing_cm", "rejected_outside_rows",
                  "rejected_off_center", "rejected_crowded",
                  "image_lat", "image_lon", "gsd_m", "yaw_deg",
                  "gimbal_pitch_deg", "gimbal_yaw_raw_deg", "yaw_source", "rtk_flag", "altitude_m", "capture_utc"]
INSTANCE_FIELDS = ["image_id", "plant_id", "lat", "lon", "centroid_x_px", "centroid_y_px",
                   "rgb_x_px", "rgb_y_px", "rgb_width_px",
                   "area_px", "ndvi_mean", "ndre_mean", "detection_method",
                   "row_id", "row_offset_frac",
                   "image_lat", "image_lon", "gsd_m", "yaw_deg"]


def discover_folders(root: Path, names: list | None) -> list:
    if names:
        return [root / n for n in names]
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.lower() in {"output", "html", "__pycache__"}:
            continue
        if (p / "clean").exists() or any(p.glob("*_D.JPG")) or any(p.glob("*_D.jpg")):
            out.append(p)
    return out


def load_boundary(cfg, in_dir):
    """Farm boundary from <folder>/farm_boundary.json or the config.
    Accepts [[lat, lon], ...] or [[lat, lon, "name"], ...]."""
    raw = cfg.get("farm_boundary")
    bfile = in_dir / "farm_boundary.json"
    if bfile.exists():
        try:
            raw = json.loads(bfile.read_text(encoding="utf-8"))
            print(f"  farm boundary from {bfile.name}")
        except Exception as exc:
            print(f"  could not read {bfile.name}: {exc}")
    if not raw or len(raw) < 3:
        return None, None
    coords = [[float(p[0]), float(p[1])] for p in raw]
    names = [str(p[2]) if len(p) > 2 else f"P{i + 1}" for i, p in enumerate(raw)]
    return coords, names


def run_folder(folder, cfg, filter_dir, all_images, no_images):
    in_dir = folder.resolve()
    out_dir = in_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    filter_stems, resolved_filter = None, None
    if not all_images:
        resolved_filter = ip.resolve_filter_dir(in_dir, filter_dir)
        if resolved_filter is not None:
            filter_stems = ip.collect_clean_stems(resolved_filter)
            if not filter_stems:
                print(f"  skip {in_dir.name}: no stems in {resolved_filter}")
                return
    sets = ip.find_set(in_dir, filter_stems=filter_stems)
    # macOS writes hidden "._<name>" metadata files next to every file copied to a
    # USB/exFAT drive or unzipped from Windows; they are not images -> ignore them
    n_all = len(sets)
    sets = [(stem, jpg, bands) for stem, jpg, bands in sets
            if not str(stem).startswith("._")
            and not Path(jpg).name.startswith("._")
            and not any(Path(b).name.startswith("._") for b in bands.values())]
    if len(sets) < n_all:
        print(f"  ignored {n_all - len(sets)} hidden macOS '._' files in {in_dir.name}")
    if not sets:
        print(f"  skip {in_dir.name}: no matching capture sets")
        return
    print(f"\n=== {in_dir.name} ({len(sets)} images) ===")
    if filter_stems is not None:
        print(f"  clean filter: {resolved_filter} ({len(filter_stems)} listed)")
    summary_path = out_dir / "plant_count_batch.csv"
    instance_path = out_dir / "plant_instances_batch.csv"
    base_dir = out_dir / "_base"
    if base_dir.exists():
        shutil.rmtree(base_dir, ignore_errors=True)
    boundary, boundary_names = load_boundary(cfg, in_dir)
    cfg = {**cfg, "_boundary": boundary, "_boundary_names": boundary_names}
    summaries, all_plants, marks, px2ll, img_paths = [], [], {}, {}, {}
    with open(instance_path, "w", encoding="utf-8", newline="") as inf:
        iw = csv.DictWriter(inf, fieldnames=INSTANCE_FIELDS)
        iw.writeheader()
        for i, (stem, jpg, bands) in enumerate(sets, 1):
            print(f"  [{i}/{len(sets)}] {stem}", end=" ... ")
            try:
                summary, plants = process_one(stem, jpg, bands, out_dir, cfg,
                                              write_images=not no_images)
                marks[stem] = summary.pop("_boundary_px", [])
                px2ll[stem] = summary.pop("_px2ll", [])
                img_paths[stem] = jpg
                summaries.append(summary)
                all_plants.extend(plants)
                if cfg.get("write_instance_csv", True):
                    iw.writerows(plants)
                    inf.flush()
                print(f"{summary['plant_count']} detections ({summary['status']})")
            except Exception as exc:
                print(f"FAILED: {exc}")

    # ---- remove duplicates + draw filtered evidence images
    print(f"\n  --- removing duplicates ({len(all_plants)} detections) ---")
    per_image, unique = {}, None
    try:
        res = dd.run_dedupe(
            all_plants, out_dir, base_dir=base_dir,
            spacing=cfg.get("dedupe_spacing_m"),
            search_radius=float(cfg.get("dedupe_search_radius_m", 1.0)),
            min_overlap=int(cfg.get("dedupe_min_overlap", 15)),
            no_align=bool(cfg.get("gps_rtk", False)),
            draw_images=not no_images and cfg.get("write_overlay", True),
            hide_repeats=bool(cfg.get("hide_repeats_in_images", False)),
            farm_boundary=boundary, boundary_marks=marks,
            px2ll=px2ll, image_paths=img_paths)
        if res is None:
            print("  no georeferenced detections - de-duplication skipped")
        else:
            per_image, unique = res["per_image"], res["unique"]
            html = pm.write_plant_map(res["uniq_path"], out_dir / "plant_map.html",
                                      boundary, title=f"Plant map - {in_dir.name}")
            print(f"  map: {html}")
    except Exception as exc:
        print(f"  de-duplication FAILED: {exc}")
    if not cfg.get("keep_base_images", False):
        shutil.rmtree(base_dir, ignore_errors=True)

    with open(summary_path, "w", encoding="utf-8", newline="") as sf:
        sw = csv.DictWriter(sf, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        sw.writeheader()
        for sm in summaries:
            a = per_image.get(sm["image_id"], {})
            sw.writerow({**sm, "detections_raw": sm["plant_count"],
                         "new_plants": a.get("new_plants", 0 if per_image else ""),
                         "repeat_plants": a.get("repeat_plants", 0 if per_image else ""),
                         "outside_farm": a.get("outside_farm", 0 if per_image else ""),
                         "running_total": a.get("running_total", "")})
    print(f"\n  {in_dir.name}: raw detections={len(all_plants)} | "
          f"UNIQUE PLANTS{' IN FARM' if boundary else ''}="
          f"{unique if unique is not None else 'n/a'}")
    print(f"  results in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Count plants on plastic mulch (clean captures)")
    ap.add_argument("--input", type=Path, nargs="*", default=None,
                    help="Folder(s) to process (default: all with clean/ or JPG sets)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON config path")
    ap.add_argument("--filter-dir", type=Path, default=None,
                    help="Override clean folder (default: <input>/clean)")
    ap.add_argument("--all-images", action="store_true", help="Ignore clean/ filter")
    ap.add_argument("--no-images", action="store_true", help="Skip the evidence images")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.no_images:
        cfg["write_overlay"] = False
    folders = discover_folders(ROOT, args.input)
    if not folders:
        raise SystemExit("No folders to process")
    print(f"Plant-count pipeline | config={args.config.name} | folders={len(folders)}")
    for folder in folders:
        run_folder(folder, cfg, args.filter_dir, args.all_images, args.no_images)
    print("\nDone.")


if __name__ == "__main__":
    main()