"""
Align each Mavic 3M band (vignetting, undistort, CalibratedHMatrix), register
RGB to the corrected NIR grid, segment plastic mulch, then compute NDVI and
NDRE on plant / soil / plastic regions.

Radiometry multiplies by SensorGainAdjustment. The correction lives in
maize/align_calibrate_m3m.py.

Writes per-image RGB / NDVI / NDRE images plus consolidated batch CSVs with lat/lon:
  - <stem>_rgb.jpg, <stem>_ndvi.png, <stem>_ndre.png
  - ndvi_regions_batch.csv / ndvi_tile_summary_100.csv
  - ndre_regions_batch.csv / ndre_tile_summary_100.csv

Also flags stressed hotspot tiles (same NDVI/NDRE rules as plot_stress_hotspots.py)
and draws them on RGB / NDVI / NDRE images in a separate folder:
  - <hotspot-dir>/<stem>_hotspots_rgb.jpg
  - <hotspot-dir>/<stem>_hotspots_ndvi.png
  - <hotspot-dir>/<stem>_hotspots_ndre.png
  - <hotspot-dir>/hotspot_tiles_batch.csv

Detects individual plants with plant-count1.py on the MS-aligned RGB, then
samples localised NDVI/NDRE in a canopy-sized window around each centroid:
  - <stem>_plants.csv / plant_ndvi_ndre_batch.csv
  - <stem>_plants.jpg  (plants coloured by NDVI/NDRE condition)

Client capture folders are resolved via --data-root or GRAI_DATA_ROOT (default: current directory).
Outputs stay inside each client folder: <client>/output, <client>/hotspot-tiles.
"""

import os
import sys
import argparse
import csv
import importlib.util
import json
import math
import re
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

try:
    import stress_rule  # only needed for hotspot/stress classification
except ModuleNotFoundError:
    stress_rule = None

import cv2
import numpy as np

DEFAULT_DATA_ROOT = Path(os.environ.get("GRAI_DATA_ROOT", "."))
DEFAULT_HOTSPOT_DIRNAME = "hotspot-tiles"
CLIENT_SKIP_DIRS = {
    "output",
    "html",
    "__pycache__",
    "hotspot-tiles",
    "plant-count-output",
    "plant-count-geo-output",
    "clean",
    "ravindra_agricultural_report",
}
NDVI_VEG_THRESH = 0.35
TILE_SIZE = 100
PLANT_CLUSTER_PX = 30
PLANT_MAX_DIM = 2000
PLANT_MIN_SAMPLE_PX = 3

_ROOT = Path(__file__).resolve().parent
_pc1_path = _ROOT / "plant-count1.py"
if _pc1_path.exists():  # only needed for this file's own plant detection
    _pc1_spec = importlib.util.spec_from_file_location("pc1", _pc1_path)
    pc1 = importlib.util.module_from_spec(_pc1_spec)
    _pc1_spec.loader.exec_module(pc1)
else:
    pc1 = None
HOTSPOT_CONFIG_NAME = "ndvi-ndre-config.json"
HOTSPOT_DEFAULT_CONFIG = {
    "ndvi": {
        "stressed_max": 0.425,
        "watch_min": 0.425,
        "watch_max": 0.456,
        "healthy_min": 0.456,
        "bare_max": 0.15,
    },
    "ndre": {
        "stressed_max": 0.20,
        "watch_min": 0.20,
        "watch_max": 0.32,
        "healthy_min": 0.32,
        "bare_max": 0.0,
    },
    "stress_rule": "both",
    "prefer_plant_values": True,
    "min_plant_pixels": 10,
    "exclude_bare": True,
    "require_vegetation": False,
    "min_veg_share_pct": 10.0,
    "require_plastic": True,
    "min_plastic_pixels": 50,
    "min_plastic_fraction": 0.05,
}


# ---------------------------------------------------------------------------
# Registration: RGB (5280x3956) -> MS grid (2592x1944)
# ---------------------------------------------------------------------------

def grad_mag(img):
    """Modality-invariant gradient magnitude (float32), then uint8 for ORB."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = g.astype(np.float32)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(cv2.magnitude(gx, gy), (5, 5), 0)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag, mag_u8


def central_mask(shape, frac=0.9):
    h, w = shape[:2]
    y0, y1 = int(h * (1 - frac) / 2), int(h * (1 + frac) / 2)
    x0, x1 = int(w * (1 - frac) / 2), int(w * (1 + frac) / 2)
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def to_u8(img):
    return cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )


def match_homography(tmpl_u8, inp_u8, mask=None, ransac_thresh=5.0):
    orb = cv2.ORB_create(nfeatures=20000, scaleFactor=1.2, nlevels=8)
    k1, d1 = orb.detectAndCompute(tmpl_u8, mask)
    k2, d2 = orb.detectAndCompute(inp_u8, mask)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, 0, 0

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
    good = [
        a
        for pair in pairs
        if len(pair) == 2
        for a, b in [pair]
        if a.distance < 0.75 * b.distance
    ]
    if len(good) < 8:
        return None, len(good), 0

    src = np.float32([k2[m.queryIdx].pt for m in good])
    dst = np.float32([k1[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    n_in = int(inliers.sum()) if inliers is not None else 0
    return H, len(good), n_in


def register_rgb_to_ms(rgb, ms_ref, refine_band=None):
    h, w = ms_ref.shape[:2]
    rgb_s = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    mask = central_mask((h, w), 0.85)

    candidates = []
    if refine_band is not None:
        candidates.append(("red-R", to_u8(refine_band), rgb_s[:, :, 2]))
        candidates.append(
            ("grad(red)-grad(R)", grad_mag(refine_band)[1], grad_mag(rgb_s[:, :, 2])[1])
        )
    candidates.append(("grad(MS)-grad(RGB)", grad_mag(ms_ref)[1], grad_mag(rgb_s)[1]))

    best = None
    for name, tmpl_u8, inp_u8 in candidates:
        H, n_good, n_in = match_homography(tmpl_u8, inp_u8, mask)
        print(f"  {name}: matches={n_good} inliers={n_in}")
        if H is None:
            continue
        if best is None or n_in > best[0]:
            best = (n_in, name, H)

    if best is None or best[0] < 10:
        raise RuntimeError("Homography estimation failed (too few inliers)")

    _, name, H = best
    print(f"  using {name}")

    if refine_band is not None:
        tmpl_raw = refine_band.astype(np.float32)
        inp_raw = cv2.warpPerspective(
            rgb_s[:, :, 2].astype(np.float32), H, (w, h), flags=cv2.INTER_LINEAR
        )
    else:
        tmpl_raw = ms_ref.astype(np.float32)
        inp_raw = cv2.warpPerspective(
            cv2.cvtColor(rgb_s, cv2.COLOR_BGR2GRAY).astype(np.float32),
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
        )

    tmpl_ecc = cv2.normalize(tmpl_raw, None, 0, 1, cv2.NORM_MINMAX)
    inp_ecc = cv2.normalize(inp_raw, None, 0, 1, cv2.NORM_MINMAX)

    shift, response = cv2.phaseCorrelate(
        tmpl_ecc.astype(np.float64), inp_ecc.astype(np.float64)
    )
    dx, dy = shift
    print(f"  phase shift dx={dx:.2f} dy={dy:.2f} resp={response:.4f}")
    # phaseCorrelate(tmpl, inp) returns how far inp is shifted FROM tmpl, so the
    # correction is the opposite shift (the old +dx/+dy doubled the error)
    T = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], dtype=np.float64)
    H = T @ H

    rgb_w = cv2.warpPerspective(rgb_s, H, (w, h), flags=cv2.INTER_AREA)
    src_ch = (
        rgb_s[:, :, 2]
        if refine_band is not None
        else cv2.cvtColor(rgb_s, cv2.COLOR_BGR2GRAY)
    )
    inp_ecc = cv2.normalize(
        cv2.warpPerspective(src_ch.astype(np.float32), H, (w, h), flags=cv2.INTER_LINEAR),
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    W = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    try:
        cc, W = cv2.findTransformECC(
            tmpl_ecc, inp_ecc, W, cv2.MOTION_AFFINE, criteria, None, 5
        )
        print(f"  ECC correlation={cc:.4f}")
    except cv2.error as e:
        msg = e.err if hasattr(e, "err") else str(e)
        print(f"  ECC skipped ({msg})")
        W = np.eye(2, 3, dtype=np.float32)

    rgb_reg = cv2.warpAffine(
        rgb_w, W, (w, h), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP
    )
    # ECC's W maps template (MS) coords -> rgb_w coords, so the full RGB -> MS
    # mapping is inv(W) @ H (matches rgb_reg, which uses WARP_INVERSE_MAP)
    W3 = np.vstack([W, [0, 0, 1]]).astype(np.float64)
    return rgb_reg, np.linalg.inv(W3) @ H


# ---------------------------------------------------------------------------
# Plastic mulch segmentation (B / S / V + morphology)
# ---------------------------------------------------------------------------

def segment_plastic(rgb, working_width=1600):
    h0, w0 = rgb.shape[:2]
    scale = working_width / w0
    img = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    sm = cv2.bilateralFilter(img, 9, 50, 50)
    B, G, R = cv2.split(sm)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)

    _, mB = cv2.threshold(B, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mS = (S < 60).astype(np.uint8) * 255
    mV = (V > 150).astype(np.uint8) * 255
    mask = mB & mS & mV

    exg = 2 * G.astype(np.int16) - R.astype(np.int16) - B.astype(np.int16)
    mask[exg > 30] = 0

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    )

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 300 or w / max(h, 1) < 1.5:
            continue
        clean[lab == i] = 255

    fg = gray[clean == 255].astype(float)
    bg = gray[clean == 0].astype(float)
    if len(fg) and len(bg):
        snr = (fg.mean() - bg.mean()) / np.hypot(fg.std() + 1e-6, bg.std() + 1e-6)
        print(
            f"  plastic mu={fg.mean():.0f} sig={fg.std():.0f} | "
            f"bg mu={bg.mean():.0f} sig={bg.std():.0f} | SNR={snr:.2f}"
        )

    return cv2.resize(clean, (w0, h0), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Georeferencing (DJI XMP, nadir MS)
# ---------------------------------------------------------------------------

def read_dji_xmp(path):
    raw = Path(path).read_bytes()
    i = raw.find(b"<x:xmpmeta")
    if i < 0:
        return {}
    j = raw.find(b"</x:xmpmeta>", i)
    if j < 0:
        return {}
    xmp = raw[i : j + 12].decode("utf-8", "replace")
    return dict(re.findall(r'drone-dji:([A-Za-z0-9]+)="([^"]+)"', xmp))


def meta_float(meta, key, default=None):
    v = meta.get(key)
    if v is None:
        return default
    return float(v)


def load_m3m_calibration():
    """Shared M3M alignment module (maize/align_calibrate_m3m.py)."""
    import sys

    name = "align_calibrate_m3m"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "maize" / "align_calibrate_m3m.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def radiometric_reflectance(img, tif_path):
    """
    Scalar DJI reflectance for one band.

    SensorGainAdjustment is multiplied. Band alignment is
    align_calibrate_m3m.calibrate_loaded_bands, which this pipeline uses
    before NDVI/NDRE.

    If BlackLevel is missing, returns float32 DNs unchanged.
    """
    meta = read_dji_xmp(tif_path)
    bl = meta_float(meta, "BlackLevel")
    if bl is None:
        return img.astype(np.float32)
    gain = meta_float(meta, "SensorGain", 1.0) or 1.0
    adj = meta_float(meta, "SensorGainAdjustment", 1.0) or 1.0
    exp_us = meta_float(meta, "ExposureTime", 1.0) or 1.0
    irr = meta_float(meta, "Irradiance", 1.0) or 1.0
    dn = np.clip(img.astype(np.float32) - bl, 0, None)
    return dn * adj / (gain * (exp_us / 1e6) * irr + 1e-12)


def pixel_to_latlon(col, row, lat0, lon0, yaw_deg, gsd_m, cx, cy):
    """
    Map MS pixel (col,row) to WGS84 using nadir pinhole + gimbal yaw.
    Yaw = heading of the image top, clockwise from north (DJI convention).
    Image right = heading + 90 deg, image down = heading + 180 deg:
      east  =  dx*cos(yaw) - dy*sin(yaw)
      north = -dx*sin(yaw) - dy*cos(yaw)
    (The previous version had +dy*sin(yaw) in east, which is only correct for
    yaw = 0 / 180 and collapses positions at diagonal headings.)
    """
    dx = (col - cx) * gsd_m
    dy = (row - cy) * gsd_m
    yaw = math.radians(yaw_deg)
    east = dx * math.cos(yaw) - dy * math.sin(yaw)
    north = -dx * math.sin(yaw) - dy * math.cos(yaw)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    return lat0 + north / m_per_deg_lat, lon0 + east / m_per_deg_lon


def latlon_to_pixel(lat, lon, lat0, lon0, yaw_deg, gsd_m, cx, cy):
    """Exact inverse of pixel_to_latlon: WGS84 -> MS pixel (col, row)."""
    north = (lat - lat0) * 111320.0
    east = (lon - lon0) * 111320.0 * math.cos(math.radians(lat0))
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    dx = east * c - north * s
    dy = -east * s - north * c
    return cx + dx / gsd_m, cy + dy / gsd_m


def georef_from_ms(nir_path, shape):
    meta = read_dji_xmp(nir_path)
    lat0 = meta_float(meta, "GpsLatitude")
    lon0 = meta_float(meta, "GpsLongitude")
    if lat0 is None or lon0 is None:
        raise RuntimeError(f"Missing GPS in XMP: {nir_path}")
    warnings = []
    alt = meta_float(meta, "RelativeAltitude", 0.0)
    if not alt or alt <= 0:
        warnings.append("RelativeAltitude missing/<=0 - ground size unknown, positions invalid")
    # Heading of the image top. FlightYawDegree (drone body) is used: when the
    # camera points exactly straight down (pitch -90.0) DJI's GimbalYawDegree
    # is ambiguous and is often reported 180 deg off (gimbal lock), while the
    # photo itself is not rotated. The camera turns with the body, so the
    # flight yaw is the reliable value.
    flight_yaw = meta_float(meta, "FlightYawDegree")
    gimbal_yaw = meta_float(meta, "GimbalYawDegree")
    if flight_yaw is not None:
        yaw, yaw_source = flight_yaw, "flight"
        if gimbal_yaw is not None:
            d = abs((gimbal_yaw - flight_yaw + 180.0) % 360.0 - 180.0)
            if 10.0 < d < 170.0:
                warnings.append(f"gimbal yaw {gimbal_yaw:.1f} and flight yaw {flight_yaw:.1f} "
                                f"differ by {d:.0f} deg (turning?) - using flight yaw")
    elif gimbal_yaw is not None:
        yaw, yaw_source = gimbal_yaw, "gimbal"
        warnings.append("FlightYawDegree missing - using GimbalYawDegree "
                        "(may be 180 deg off when pitch is exactly -90)")
    else:
        yaw, yaw_source = 0.0, "none"
        warnings.append("no yaw in metadata - assuming 0 (north up), positions unreliable")
    pitch = meta_float(meta, "GimbalPitchDegree")
    if pitch is not None and abs(pitch + 90.0) > 2.0:
        off = abs(alt * math.tan(math.radians(pitch + 90.0))) if alt else 0.0
        warnings.append(f"camera not pointing straight down (pitch {pitch:.1f} deg) - "
                        f"positions shifted ~{off:.1f} m")
    f_px = meta_float(meta, "CalibratedFocalLength")
    if not f_px:
        f_px = 2170.0
        warnings.append("CalibratedFocalLength missing - using 2170 px (approximate)")
    cx = meta_float(meta, "CalibratedOpticalCenterX", shape[1] / 2.0)
    cy = meta_float(meta, "CalibratedOpticalCenterY", shape[0] / 2.0)
    gsd = alt / f_px if alt else 0.0
    rtk = meta.get("RtkFlag", "")
    return {
        "pitch_deg": pitch if pitch is not None else "",
        "yaw_source": yaw_source,
        "gimbal_yaw_deg": gimbal_yaw if gimbal_yaw is not None else "",
        "flight_yaw_deg": flight_yaw if flight_yaw is not None else "",
        "rtk_flag": rtk,          # 50 = RTK fixed (cm accuracy); 0/16/34 = standard GPS / float
        "warnings": warnings,
        "lat": lat0,
        "lon": lon0,
        "alt_m": alt,
        "yaw_deg": yaw,
        "gsd_m": gsd,
        "cx": cx,
        "cy": cy,
        "capture_utc": meta.get("UTCAtExposure", ""),
    }


def _valid_mean(val):
    return not (
        val is None
        or val == ""
        or (isinstance(val, float) and math.isnan(val))
    )


def vigor_estimate(mean_ndvi):
    if not _valid_mean(mean_ndvi):
        return "No Data"
    if mean_ndvi < 0.2:
        return "Sparse Vegetation"
    if mean_ndvi < 0.5:
        return "Moderate Vegetation"
    return "Dense Vegetation"


def vigor_estimate_ndre(mean_ndre):
    """NDRE vigor bins (lower dynamic range than NDVI)."""
    if not _valid_mean(mean_ndre):
        return "No Data"
    if mean_ndre < 0.15:
        return "Sparse Vegetation"
    if mean_ndre < 0.3:
        return "Moderate Vegetation"
    return "Dense Vegetation"


# ---------------------------------------------------------------------------
# NDVI / NDRE + regional / tile analysis
# ---------------------------------------------------------------------------

def compute_ndvi(red, nir):
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    return (nir - red) / (nir + red + 1e-6)


def compute_ndre(red_edge, nir):
    """NDRE = (NIR - RedEdge) / (NIR + RedEdge)."""
    re = red_edge.astype(np.float32)
    nir = nir.astype(np.float32)
    return (nir - re) / (nir + re + 1e-6)


def region_stats(index_arr, mask, name):
    vals = index_arr[mask]
    if vals.size == 0:
        print(f"  {name}: 0 px")
        return {"region": name, "pixels": 0}
    stats = {
        "region": name,
        "pixels": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "p25": float(np.percentile(vals, 25)),
        "median": float(np.median(vals)),
        "p75": float(np.percentile(vals, 75)),
    }
    print(
        f"  {name}: n={stats['pixels']}  "
        f"mean={stats['mean']:.3f}  med={stats['median']:.3f}  "
        f"p25={stats['p25']:.3f}  p75={stats['p75']:.3f}"
    )
    return stats


def colorize_index(arr, vmin=-0.2, vmax=0.8, colorbar=False):
    clipped = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    vis = cv2.applyColorMap((clipped * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if not colorbar:
        return vis
    h, w = vis.shape[:2]
    bar_w = 36
    pad = 8
    canvas = np.zeros((h, w + bar_w + 52, 3), np.uint8)
    canvas[:, :w] = vis
    ramp = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    canvas[:, w + pad : w + pad + bar_w] = cv2.applyColorMap(ramp, cv2.COLORMAP_JET)
    ticks = [
        (0.0, f"{vmax:.1f}"),
        (0.5, f"{0.5 * (vmin + vmax):.2f}"),
        (1.0, f"{vmin:.1f}"),
    ]
    for frac, label in ticks:
        y = int(frac * (h - 1))
        cv2.putText(
            canvas,
            label,
            (w + pad + bar_w + 4, max(12, min(h - 4, y + 4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def make_class_map(plastic, veg):
    plants = (plastic > 0) & veg
    plastic_only = (plastic > 0) & ~veg
    weeds = (plastic == 0) & veg
    soil = (plastic == 0) & ~veg
    out = np.zeros((*plastic.shape, 3), np.uint8)
    out[soil] = (40, 70, 120)
    out[plastic_only] = (220, 220, 40)
    out[weeds] = (0, 200, 255)
    out[plants] = (0, 220, 0)
    return out, {
        "soil": soil,
        "plastic_only": plastic_only,
        "weeds": weeds,
        "plants": plants,
    }


def _tile_geo_base(stem, geo, tile_row, tile_col, x0, y0, x1, y1):
    cx = 0.5 * (x0 + x1 - 1)
    cy = 0.5 * (y0 + y1 - 1)
    lat, lon = pixel_to_latlon(
        cx, cy, geo["lat"], geo["lon"], geo["yaw_deg"], geo["gsd_m"], geo["cx"], geo["cy"]
    )
    return {
        "image_id": stem,
        "tile_row": tile_row,
        "tile_col": tile_col,
        "x_start": x0,
        "y_start": y0,
        "x_end": x1,
        "y_end": y1,
        "tile_width": x1 - x0,
        "tile_height": y1 - y0,
        "lat": lat,
        "lon": lon,
        "tile_center_lat": lat,
        "tile_center_lon": lon,
        "image_lat": geo["lat"],
        "image_lon": geo["lon"],
        "relative_altitude_m": geo["alt_m"],
        "gsd_m": geo["gsd_m"],
        "gimbal_yaw_deg": geo["yaw_deg"],
    }


def tile_ndvi_rows(stem, ndvi, plant_mask, geo, bands, tile_size=TILE_SIZE, plastic_mask=None):
    h, w = ndvi.shape
    if plastic_mask is None:
        plastic_mask = np.zeros((h, w), dtype=bool)
    else:
        plastic_mask = plastic_mask > 0
    rows = []
    tile_row = 0
    for y0 in range(0, h, tile_size):
        tile_row += 1
        tile_col = 0
        y1 = min(y0 + tile_size, h)
        for x0 in range(0, w, tile_size):
            tile_col += 1
            x1 = min(x0 + tile_size, w)
            patch = ndvi[y0:y1, x0:x1]
            plast = plastic_mask[y0:y1, x0:x1]
            plant_vals = patch[plant_mask[y0:y1, x0:x1]]
            vals = patch.ravel()
            mean = float(vals.mean())
            plast_count = int(plast.sum())
            row = _tile_geo_base(stem, geo, tile_row, tile_col, x0, y0, x1, y1)
            row.update(
                {
                    "nir_file": bands["NIR"].name,
                    "red_file": bands["R"].name,
                    "pixel_count": int(vals.size),
                    "ndvi_min": float(vals.min()),
                    "ndvi_max": float(vals.max()),
                    "ndvi_mean": mean,
                    "ndvi_median": float(np.median(vals)),
                    "ndvi_std": float(vals.std()),
                    "pct_negative": float(100.0 * np.mean(vals < 0)),
                    "pct_low_0_to_0_2": float(100.0 * np.mean((vals >= 0) & (vals < 0.2))),
                    "pct_moderate_0_2_to_0_5": float(
                        100.0 * np.mean((vals >= 0.2) & (vals < 0.5))
                    ),
                    "pct_high_ge_0_5": float(100.0 * np.mean(vals >= 0.5)),
                    "vigor_estimate": vigor_estimate(mean),
                    "plastic_pixel_count": plast_count,
                    "plastic_fraction": float(plast_count / max(vals.size, 1)),
                    "plant_pixel_count": int(plant_vals.size),
                    "plant_ndvi_mean": float(plant_vals.mean()) if plant_vals.size else "",
                    "plant_ndvi_median": (
                        float(np.median(plant_vals)) if plant_vals.size else ""
                    ),
                }
            )
            rows.append(row)
    return rows


def tile_ndre_rows(stem, ndre, plant_mask, geo, bands, tile_size=TILE_SIZE, plastic_mask=None):
    h, w = ndre.shape
    if plastic_mask is None:
        plastic_mask = np.zeros((h, w), dtype=bool)
    else:
        plastic_mask = plastic_mask > 0
    rows = []
    tile_row = 0
    for y0 in range(0, h, tile_size):
        tile_row += 1
        tile_col = 0
        y1 = min(y0 + tile_size, h)
        for x0 in range(0, w, tile_size):
            tile_col += 1
            x1 = min(x0 + tile_size, w)
            patch = ndre[y0:y1, x0:x1]
            plast = plastic_mask[y0:y1, x0:x1]
            plant_vals = patch[plant_mask[y0:y1, x0:x1]]
            vals = patch.ravel()
            mean = float(vals.mean())
            plast_count = int(plast.sum())
            row = _tile_geo_base(stem, geo, tile_row, tile_col, x0, y0, x1, y1)
            row.update(
                {
                    "nir_file": bands["NIR"].name,
                    "red_edge_file": bands["RE"].name,
                    "pixel_count": int(vals.size),
                    "ndre_min": float(vals.min()),
                    "ndre_max": float(vals.max()),
                    "ndre_mean": mean,
                    "ndre_median": float(np.median(vals)),
                    "ndre_std": float(vals.std()),
                    "pct_negative": float(100.0 * np.mean(vals < 0)),
                    "pct_low_0_to_0_15": float(
                        100.0 * np.mean((vals >= 0) & (vals < 0.15))
                    ),
                    "pct_moderate_0_15_to_0_3": float(
                        100.0 * np.mean((vals >= 0.15) & (vals < 0.3))
                    ),
                    "pct_high_ge_0_3": float(100.0 * np.mean(vals >= 0.3)),
                    "vigor_estimate": vigor_estimate_ndre(mean),
                    "plastic_pixel_count": plast_count,
                    "plastic_fraction": float(plast_count / max(vals.size, 1)),
                    "plant_pixel_count": int(plant_vals.size),
                    "plant_ndre_mean": float(plant_vals.mean()) if plant_vals.size else "",
                    "plant_ndre_median": (
                        float(np.median(plant_vals)) if plant_vals.size else ""
                    ),
                }
            )
            rows.append(row)
    return rows


def build_region_rows(stem, all_stats, regions, geo, index_prefix, vigor_fn):
    """Build region CSV rows for NDVI or NDRE."""
    region_rows = []
    for s in all_stats:
        lat, lon = geo["lat"], geo["lon"]
        if s["region"].startswith("plants") and s.get("pixels", 0) > 0:
            ys, xs = np.where(regions["plants"])
            lat, lon = pixel_to_latlon(
                float(xs.mean()),
                float(ys.mean()),
                geo["lat"],
                geo["lon"],
                geo["yaw_deg"],
                geo["gsd_m"],
                geo["cx"],
                geo["cy"],
            )
        region_rows.append(
            {
                "image_id": stem,
                "region": s["region"],
                "lat": lat,
                "lon": lon,
                "image_lat": geo["lat"],
                "image_lon": geo["lon"],
                "relative_altitude_m": geo["alt_m"],
                "gsd_m": geo["gsd_m"],
                "gimbal_yaw_deg": geo["yaw_deg"],
                "capture_utc": geo.get("capture_utc", ""),
                "pixels": s.get("pixels", 0),
                f"{index_prefix}_mean": s.get("mean", ""),
                f"{index_prefix}_std": s.get("std", ""),
                f"{index_prefix}_min": s.get("min", ""),
                f"{index_prefix}_max": s.get("max", ""),
                f"{index_prefix}_p25": s.get("p25", ""),
                f"{index_prefix}_median": s.get("median", ""),
                f"{index_prefix}_p75": s.get("p75", ""),
                "vigor_estimate": vigor_fn(s.get("mean"))
                if s.get("pixels", 0)
                else "No Data",
            }
        )
    return region_rows


def find_set(folder: Path, filter_stems=None):
    seen = set()
    jpgs = []
    for pat in ("*_D.JPG", "*_D.jpg"):
        for p in folder.glob(pat):
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                jpgs.append(p)
    jpgs.sort(key=lambda p: p.name.lower())
    sets = []
    for jpg in jpgs:
        stem = jpg.name.rsplit("_D.", 1)[0]
        if filter_stems is not None and stem not in filter_stems:
            continue
        bands = {
            "G": folder / f"{stem}_MS_G.TIF",
            "NIR": folder / f"{stem}_MS_NIR.TIF",
            "R": folder / f"{stem}_MS_R.TIF",
            "RE": folder / f"{stem}_MS_RE.TIF",
        }
        if all(p.exists() for p in bands.values()):
            sets.append((stem, jpg, bands))
    return sets


def stem_from_clean_filename(path: Path):
    """Extract capture stem from files in clean/ (e.g. *_rgb.jpg, *_D.JPG)."""
    name = path.stem
    if name.endswith("_rgb"):
        return name[:-4]
    if name.endswith("_D"):
        return name[:-2]
    return name


def collect_clean_stems(clean_dir: Path):
    """
    Return capture stems listed under a clean/ folder.
    Scans recursively for image files (*_rgb.jpg, *_D.JPG, etc.).
    """
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    stems = set()
    if not clean_dir.exists():
        return stems
    for path in clean_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            stems.add(stem_from_clean_filename(path))
    return stems


def resolve_filter_dir(input_dir: Path, filter_dir=None):
    """Resolve --filter-dir; default to <input>/clean when present."""
    if filter_dir is not None:
        return filter_dir.resolve()
    default = (input_dir / "clean").resolve()
    if default.exists():
        return default
    return None


def is_client_folder(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("."):
        return False
    if path.name.lower() in CLIENT_SKIP_DIRS:
        return False
    if (path / "clean").exists():
        return True
    return any(path.glob("*_D.JPG")) or any(path.glob("*_D.jpg"))


def resolve_client_folder(data_root: Path, name) -> Path:
    """Resolve a client folder name or path against the drone-data root."""
    raw = Path(name)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(data_root / raw)
        candidates.append(Path(raw))
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise SystemExit(
        f"Client folder not found: {name}\n  looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def discover_client_folders(data_root: Path, names=None):
    """Return client capture folders under data_root (or the named ones)."""
    if names:
        return [resolve_client_folder(data_root, n) for n in names]
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")
    folders = [p.resolve() for p in sorted(data_root.iterdir()) if is_client_folder(p)]
    if not folders:
        raise SystemExit(f"No client folders found in {data_root}")
    return folders


REGION_CSV_FIELDS = [
    "image_id",
    "region",
    "lat",
    "lon",
    "image_lat",
    "image_lon",
    "relative_altitude_m",
    "gsd_m",
    "gimbal_yaw_deg",
    "capture_utc",
    "pixels",
    "ndvi_mean",
    "ndvi_std",
    "ndvi_min",
    "ndvi_max",
    "ndvi_p25",
    "ndvi_median",
    "ndvi_p75",
    "vigor_estimate",
]

NDRE_REGION_CSV_FIELDS = [
    "image_id",
    "region",
    "lat",
    "lon",
    "image_lat",
    "image_lon",
    "relative_altitude_m",
    "gsd_m",
    "gimbal_yaw_deg",
    "capture_utc",
    "pixels",
    "ndre_mean",
    "ndre_std",
    "ndre_min",
    "ndre_max",
    "ndre_p25",
    "ndre_median",
    "ndre_p75",
    "vigor_estimate",
]

TILE_CSV_FIELDS = [
    "image_id",
    "tile_row",
    "tile_col",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "tile_width",
    "tile_height",
    "lat",
    "lon",
    "tile_center_lat",
    "tile_center_lon",
    "image_lat",
    "image_lon",
    "relative_altitude_m",
    "gsd_m",
    "gimbal_yaw_deg",
    "nir_file",
    "red_file",
    "pixel_count",
    "ndvi_min",
    "ndvi_max",
    "ndvi_mean",
    "ndvi_median",
    "ndvi_std",
    "pct_negative",
    "pct_low_0_to_0_2",
    "pct_moderate_0_2_to_0_5",
    "pct_high_ge_0_5",
    "vigor_estimate",
    "plastic_pixel_count",
    "plastic_fraction",
    "plant_pixel_count",
    "plant_ndvi_mean",
    "plant_ndvi_median",
]

NDRE_TILE_CSV_FIELDS = [
    "image_id",
    "tile_row",
    "tile_col",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "tile_width",
    "tile_height",
    "lat",
    "lon",
    "tile_center_lat",
    "tile_center_lon",
    "image_lat",
    "image_lon",
    "relative_altitude_m",
    "gsd_m",
    "gimbal_yaw_deg",
    "nir_file",
    "red_edge_file",
    "pixel_count",
    "ndre_min",
    "ndre_max",
    "ndre_mean",
    "ndre_median",
    "ndre_std",
    "pct_negative",
    "pct_low_0_to_0_15",
    "pct_moderate_0_15_to_0_3",
    "pct_high_ge_0_3",
    "vigor_estimate",
    "plastic_pixel_count",
    "plastic_fraction",
    "plant_pixel_count",
    "plant_ndre_mean",
    "plant_ndre_median",
]

HOTSPOT_CSV_FIELDS = [
    "image_id",
    "tile_row",
    "tile_col",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "lat",
    "lon",
    "ndvi",
    "ndre",
    "source",
    "plant_pixel_count",
    "plastic_pixel_count",
    "plastic_fraction",
    "vigor_estimate",
]

PLANT_CSV_FIELDS = [
    "image_id",
    "plant_id",
    "x_px",
    "y_px",
    "lat",
    "lon",
    "radius_px",
    "sample_pixels",
    "sample_source",
    "ndvi_center",
    "ndvi_mean",
    "ndvi_median",
    "ndvi_std",
    "ndre_center",
    "ndre_mean",
    "ndre_median",
    "ndre_std",
    "condition",
    "vigor_estimate",
    "on_plastic",
    "image_lat",
    "image_lon",
    "gsd_m",
    "gimbal_yaw_deg",
]


# ---------------------------------------------------------------------------
# Stressed hotspot tiles (NDVI + NDRE) drawn on images
# ---------------------------------------------------------------------------

def _deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_hotspot_config(folder: Path):
    """Client ndvi-ndre-config.json if present, else this folder's JSON."""
    template = _ROOT / HOTSPOT_CONFIG_NAME
    packaged = dict(HOTSPOT_DEFAULT_CONFIG)
    packaged_path = None
    if template.exists():
        with open(template, encoding="utf-8") as f:
            packaged = _deep_merge(packaged, json.load(f))
        packaged_path = template
    client = folder / HOTSPOT_CONFIG_NAME
    if client.exists():
        with open(client, encoding="utf-8") as f:
            cfg = _deep_merge(packaged, json.load(f))
        print(f"  hotspot config: {client}")
        return cfg, client
    if packaged_path is not None:
        print(f"  hotspot config: {packaged_path} (default)")
        return packaged, packaged_path
    print(f"  hotspot config: defaults (no {HOTSPOT_CONFIG_NAME})")
    return packaged, client


def _fnum(val, default=None):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _pick_index_values(row, cfg):
    min_px = int(cfg.get("min_plant_pixels", 10))
    plant_px = int(float(row.get("plant_pixel_count") or 0))
    prefer = bool(cfg.get("prefer_plant_values", True))

    ndvi_tile = _fnum(row.get("ndvi_mean"))
    ndre_tile = _fnum(row.get("ndre_mean"))
    plant_ndvi = _fnum(row.get("plant_ndvi_mean"))
    plant_ndre = _fnum(row.get("plant_ndre_mean"))

    if prefer and plant_px >= min_px and plant_ndvi is not None:
        ndvi = plant_ndvi
        source = "plant"
    else:
        ndvi = ndvi_tile
        source = "tile"

    if prefer and plant_px >= min_px and plant_ndre is not None:
        ndre = plant_ndre
    else:
        ndre = ndre_tile

    return ndvi, ndre, source, plant_px


def _classify_condition(ndvi, ndre, cfg):
    return stress_rule.classify(ndvi, ndre, cfg)


def _passes_vegetation_filter(row, cfg):
    if not cfg.get("require_vegetation", False):
        return True
    pct_mod = _fnum(row.get("pct_moderate_0_2_to_0_5"), 0.0)
    pct_high = _fnum(row.get("pct_high_ge_0_5"), 0.0)
    return (pct_mod + pct_high) >= float(cfg.get("min_veg_share_pct", 10.0))


def _passes_plastic_filter(row, cfg):
    if not cfg.get("require_plastic", True):
        return True
    plastic_px = _fnum(row.get("plastic_pixel_count"), None)
    plastic_frac = _fnum(row.get("plastic_fraction"), None)
    min_px = float(cfg.get("min_plastic_pixels", 50))
    min_frac = float(cfg.get("min_plastic_fraction", 0.05))
    if plastic_px is None:
        return False
    pixel_count = _fnum(row.get("pixel_count"), 0) or 0
    if plastic_frac is None and pixel_count > 0:
        plastic_frac = plastic_px / pixel_count
    return plastic_px >= min_px and (plastic_frac or 0) >= min_frac


def select_hotspot_tiles(ndvi_rows, ndre_rows, cfg):
    """Return stressed plastic-row tiles with pixel bounds for overlay."""
    ndre_map = {
        (str(r.get("tile_row", "")), str(r.get("tile_col", ""))): r for r in ndre_rows
    }
    hotspots = []
    for n in ndvi_rows:
        e = ndre_map.get((str(n.get("tile_row", "")), str(n.get("tile_col", ""))), {})
        row = dict(n)
        row["ndre_mean"] = e.get("ndre_mean", "")
        row["ndre_median"] = e.get("ndre_median", "")
        row["plant_ndre_mean"] = e.get("plant_ndre_mean", "")
        row["plant_ndre_median"] = e.get("plant_ndre_median", "")

        if not _passes_vegetation_filter(row, cfg):
            continue
        if not _passes_plastic_filter(row, cfg):
            continue
        ndvi, ndre, source, plant_px = _pick_index_values(row, cfg)
        if _classify_condition(ndvi, ndre, cfg) != "stressed":
            continue

        hotspots.append(
            {
                "image_id": row.get("image_id", ""),
                "tile_row": row.get("tile_row", ""),
                "tile_col": row.get("tile_col", ""),
                "x_start": int(float(row.get("x_start", 0))),
                "y_start": int(float(row.get("y_start", 0))),
                "x_end": int(float(row.get("x_end", 0))),
                "y_end": int(float(row.get("y_end", 0))),
                "lat": row.get("lat", ""),
                "lon": row.get("lon", ""),
                "ndvi": ndvi,
                "ndre": ndre,
                "source": source,
                "plant_pixel_count": plant_px,
                "plastic_pixel_count": _fnum(row.get("plastic_pixel_count"), 0) or 0,
                "plastic_fraction": _fnum(row.get("plastic_fraction"), 0) or 0,
                "vigor_estimate": row.get("vigor_estimate", ""),
            }
        )
    return hotspots


def draw_hotspot_overlay(base_bgr, hotspots, title="Stressed hotspot tiles"):
    """Draw filled + outlined hotspot rectangles on a BGR image."""
    vis = base_bgr.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    fill = vis.copy()
    color = (0, 0, 220)
    for h in hotspots:
        pt1 = (int(h["x_start"]), int(h["y_start"]))
        pt2 = (max(int(h["x_end"]) - 1, pt1[0]), max(int(h["y_end"]) - 1, pt1[1]))
        cv2.rectangle(fill, pt1, pt2, color, thickness=-1)
    vis = cv2.addWeighted(fill, 0.28, vis, 0.72, 0)
    for h in hotspots:
        pt1 = (int(h["x_start"]), int(h["y_start"]))
        pt2 = (max(int(h["x_end"]) - 1, pt1[0]), max(int(h["y_end"]) - 1, pt1[1]))
        cv2.rectangle(vis, pt1, pt2, color, thickness=2)
        ndvi = h.get("ndvi")
        ndre = h.get("ndre")
        if ndvi is None or ndre is None:
            continue
        label = f"{ndvi:.2f}/{ndre:.2f}"
        tx, ty = pt1[0] + 3, min(pt1[1] + 14, vis.shape[0] - 4)
        cv2.putText(
            vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            vis,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    bar_h = 36
    cv2.rectangle(vis, (0, 0), (vis.shape[1], bar_h), (20, 20, 20), thickness=-1)
    legend = f"{title}  |  {len(hotspots)} tile(s)  |  red = stressed (NDVI/NDRE)"
    cv2.putText(
        vis,
        legend,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def write_hotspot_outputs(
    stem, rgb, ndvi, ndre, hotspots, hotspot_dir: Path, write_images=True
):
    hotspot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = hotspot_dir / f"{stem}_hotspots.csv"

    if write_images:
        cv2.imwrite(
            str(hotspot_dir / f"{stem}_hotspots_rgb.jpg"),
            draw_hotspot_overlay(rgb, hotspots, f"{stem} RGB hotspots"),
        )
        cv2.imwrite(
            str(hotspot_dir / f"{stem}_hotspots_ndvi.png"),
            draw_hotspot_overlay(colorize_index(ndvi), hotspots, f"{stem} NDVI hotspots"),
        )
        cv2.imwrite(
            str(hotspot_dir / f"{stem}_hotspots_ndre.png"),
            draw_hotspot_overlay(
                colorize_index(ndre, vmin=-0.1, vmax=0.5),
                hotspots,
                f"{stem} NDRE hotspots",
            ),
        )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HOTSPOT_CSV_FIELDS)
        w.writeheader()
        w.writerows(hotspots)

    dest = f"{hotspot_dir.name}/" if write_images else csv_path.name
    print(f"  hotspot tiles: {len(hotspots)} -> {dest}")
    return hotspots


# ---------------------------------------------------------------------------
# Precise plants (plant-count1.py) + localised NDVI / NDRE
# ---------------------------------------------------------------------------

def make_pc1_args(cluster_px=PLANT_CLUSTER_PX, max_dim=PLANT_MAX_DIM):
    return argparse.Namespace(
        max_dim=max_dim,
        plastic_s_max=100,
        plastic_v_min=135,
        min_plastic_area=1500,
        roi_dilate=3,
        exg_thresh=0.010,
        min_area=25,
        cluster_px=cluster_px,
        show_roi=False,
        save_masks=False,
        merge=False,
    )


def find_plant_json(input_dir: Path, stem: str):
    out_dir = input_dir / "plant-count-output"
    for name in (f"{stem}_rgb.json", f"{stem}.json"):
        path = out_dir / name
        if path.exists():
            return path
    return None


def load_plant_points_json(path: Path, shape):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    h, w = shape[:2]
    points = []
    for p in data.get("points", []):
        if isinstance(p, dict):
            x, y = int(p["x"]), int(p["y"])
        else:
            x, y = int(p[0]), int(p[1])
        if 0 <= x < w and 0 <= y < h:
            points.append((x, y))
    if not points:
        return None
    typical = float(data.get("typical_canopy_px", 80.0))
    return points, typical


def detect_plants_on_rgb(rgb_reg, pc1_args):
    """Run plant-count1 detection on MS-aligned RGB. Points are in MS pixels."""
    h, w = rgb_reg.shape[:2]
    scale = min(1.0, pc1_args.max_dim / float(max(h, w)))
    work = (
        rgb_reg
        if scale == 1.0
        else cv2.resize(rgb_reg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    )
    _roi, plants, pts_w, typical = pc1.detect(work, pc1_args)
    points = [(int(round(x / scale)), int(round(y / scale))) for x, y in pts_w]
    points = pc1.dedupe_points(points, pc1_args.cluster_px)
    plants_full = (
        plants
        if scale == 1.0
        else cv2.resize(plants, (w, h), interpolation=cv2.INTER_NEAREST)
    )
    return points, typical, scale, plants_full


def canopy_radius_px(typical, scale, override=0):
    if override and override > 0:
        return int(override)
    r_work = max(2.0, math.sqrt(max(typical, 1.0) / math.pi))
    return max(6, int(round(1.2 * r_work / max(scale, 1e-6))))


def local_plant_indices(ndvi, ndre, x, y, radius, plant_mask=None, veg_mask=None, min_px=PLANT_MIN_SAMPLE_PX):
    """
    Sample NDVI/NDRE in a disk around (x, y).
    Prefer plant-count1 canopy pixels, then vegetation, then the full disk.
    """
    h, w = ndvi.shape
    x, y = int(x), int(y)
    if not (0 <= x < w and 0 <= y < h):
        return None
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
    ndvi_p = ndvi[y0:y1, x0:x1]
    ndre_p = ndre[y0:y1, x0:x1]

    candidates = []
    if plant_mask is not None:
        candidates.append(("plant_mask", disk & (plant_mask[y0:y1, x0:x1] > 0)))
    if veg_mask is not None:
        candidates.append(("veg_disk", disk & veg_mask[y0:y1, x0:x1]))
    candidates.append(("disk", disk))

    sel, source = None, "disk"
    for name, mask in candidates:
        if int(mask.sum()) >= min_px:
            sel, source = mask, name
            break
    if sel is None:
        sel, source = disk, "disk"
        if int(sel.sum()) == 0:
            return None

    ndvi_vals = ndvi_p[sel]
    ndre_vals = ndre_p[sel]
    return {
        "pixels": int(ndvi_vals.size),
        "source": source,
        "ndvi_mean": float(ndvi_vals.mean()),
        "ndvi_median": float(np.median(ndvi_vals)),
        "ndvi_std": float(ndvi_vals.std()),
        "ndre_mean": float(ndre_vals.mean()),
        "ndre_median": float(np.median(ndre_vals)),
        "ndre_std": float(ndre_vals.std()),
        "ndvi_center": float(ndvi[y, x]),
        "ndre_center": float(ndre[y, x]),
    }


def plant_index_rows(stem, points, ndvi, ndre, geo, radius, plant_mask, veg_mask, plastic, hotspot_cfg=None):
    rows = []
    for i, (x, y) in enumerate(points, 1):
        stats = local_plant_indices(ndvi, ndre, x, y, radius, plant_mask, veg_mask)
        if stats is None:
            continue
        lat, lon = pixel_to_latlon(
            x, y, geo["lat"], geo["lon"], geo["yaw_deg"], geo["gsd_m"], geo["cx"], geo["cy"]
        )
        ndvi_m, ndre_m = stats["ndvi_mean"], stats["ndre_mean"]
        if hotspot_cfg is not None:
            condition = _classify_condition(ndvi_m, ndre_m, hotspot_cfg)
        else:
            condition = vigor_estimate(ndvi_m)
        on_plastic = bool(plastic[int(y), int(x)] > 0) if plastic is not None else ""
        rows.append(
            {
                "image_id": stem,
                "plant_id": i,
                "x_px": int(x),
                "y_px": int(y),
                "lat": lat,
                "lon": lon,
                "radius_px": radius,
                "sample_pixels": stats["pixels"],
                "sample_source": stats["source"],
                "ndvi_center": round(stats["ndvi_center"], 4),
                "ndvi_mean": round(ndvi_m, 4),
                "ndvi_median": round(stats["ndvi_median"], 4),
                "ndvi_std": round(stats["ndvi_std"], 4),
                "ndre_center": round(stats["ndre_center"], 4),
                "ndre_mean": round(ndre_m, 4),
                "ndre_median": round(stats["ndre_median"], 4),
                "ndre_std": round(stats["ndre_std"], 4),
                "condition": condition,
                "vigor_estimate": vigor_estimate(ndvi_m),
                "on_plastic": int(on_plastic) if on_plastic != "" else "",
                "image_lat": geo["lat"],
                "image_lon": geo["lon"],
                "gsd_m": geo["gsd_m"],
                "gimbal_yaw_deg": geo["yaw_deg"],
            }
        )
    return rows


def draw_plant_overlay(rgb, plant_rows, radius):
    vis = rgb.copy()
    colors = {
        "stressed": (0, 0, 220),
        "watch": (0, 180, 255),
        "healthy": (0, 200, 0),
        "bare": (160, 160, 160),
        "unknown": (180, 180, 180),
        "Sparse Vegetation": (0, 0, 220),
        "Moderate Vegetation": (0, 180, 255),
        "Dense Vegetation": (0, 200, 0),
        "No Data": (180, 180, 180),
    }
    r = max(5, int(radius))
    counts = {"stressed": 0, "watch": 0, "healthy": 0, "other": 0}
    for row in plant_rows:
        cond = str(row.get("condition") or "unknown")
        color = colors.get(cond, colors["unknown"])
        x, y = int(row["x_px"]), int(row["y_px"])
        cv2.circle(vis, (x, y), r, color, 2)
        cv2.circle(vis, (x, y), 2, color, -1)
        if cond in counts:
            counts[cond] += 1
        else:
            counts["other"] += 1

    bar_h = 36
    cv2.rectangle(vis, (0, 0), (vis.shape[1], bar_h), (20, 20, 20), thickness=-1)
    legend = (
        f"plants={len(plant_rows)}  stressed={counts['stressed']}  "
        f"watch={counts['watch']}  healthy={counts['healthy']}  "
        f"r={r}px  (red/yellow/green = NDVI+NDRE condition)"
    )
    cv2.putText(
        vis, legend, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return vis


def resolve_plant_detections(stem, rgb_reg, input_dir, pc1_args, reuse_plants):
    """Return (points, typical, scale, plant_mask, source)."""
    h, w = rgb_reg.shape[:2]
    if reuse_plants and input_dir is not None:
        json_path = find_plant_json(input_dir, stem)
        if json_path is not None:
            loaded = load_plant_points_json(json_path, (h, w))
            if loaded is not None:
                points, typical = loaded
                scale = min(1.0, pc1_args.max_dim / float(max(h, w)))
                print(f"  plants: {len(points)} from {json_path.name}")
                return points, typical, scale, None, "json"

    points, typical, scale, plant_mask = detect_plants_on_rgb(rgb_reg, pc1_args)
    print(
        f"  plants: {len(points)} detected (plant-count1, "
        f"typical={typical:.0f}px work, cluster={pc1_args.cluster_px})"
    )
    return points, typical, scale, plant_mask, "detect"


def process_one(
    stem,
    jpg_path,
    bands,
    out_dir: Path,
    write_images=True,
    hotspot_dir=None,
    hotspot_cfg=None,
    detect_plants=True,
    pc1_args=None,
    plant_radius=0,
    reuse_plants=False,
    input_dir=None,
):
    print(f"\n=== {stem} ===")
    rgb = cv2.imread(str(jpg_path))
    nir = cv2.imread(str(bands["NIR"]), cv2.IMREAD_UNCHANGED)
    red = cv2.imread(str(bands["R"]), cv2.IMREAD_UNCHANGED)
    red_edge = cv2.imread(str(bands["RE"]), cv2.IMREAD_UNCHANGED)
    if rgb is None or nir is None or red is None or red_edge is None:
        raise RuntimeError(f"Failed to read images for {stem}")

    geo = georef_from_ms(bands["NIR"], nir.shape)
    print(
        f"  RGB {rgb.shape} -> MS {nir.shape} {nir.dtype} | "
        f"GPS {geo['lat']:.6f},{geo['lon']:.6f}  "
        f"GSD={geo['gsd_m']*100:.1f}cm  yaw={geo['yaw_deg']:.1f}"
    )

    band_images = {"NIR": nir, "R": red, "RE": red_edge}
    band_paths = {"NIR": bands["NIR"], "R": bands["R"], "RE": bands["RE"]}
    green_path = bands.get("G")
    if green_path is not None and Path(green_path).exists():
        green = cv2.imread(str(green_path), cv2.IMREAD_UNCHANGED)
        if green is not None:
            band_images["G"] = green
            band_paths["G"] = green_path
    calibrated = load_m3m_calibration().calibrate_loaded_bands(
        band_images, band_paths, rgb=rgb, rgb_path=jpg_path, verbose=True
    )
    rgb_reg = calibrated["rgb"]
    red_c = calibrated["refl"]["R"]
    nir_c = calibrated["refl"]["NIR"]
    re_c = calibrated["refl"]["RE"]
    plastic = segment_plastic(rgb_reg)
    ndvi = compute_ndvi(red_c, nir_c)
    ndre = compute_ndre(re_c, nir_c)
    veg = ndvi > NDVI_VEG_THRESH
    _, regions = make_class_map(plastic, veg)

    print("  NDVI by region:")
    ndvi_stats = [
        region_stats(ndvi, np.ones_like(ndvi, bool), "all"),
        region_stats(ndvi, regions["plants"], "plants (plastic AND veg)"),
        region_stats(ndvi, regions["weeds"], "weeds (veg off plastic)"),
        region_stats(ndvi, regions["soil"], "soil"),
        region_stats(ndvi, regions["plastic_only"], "bare plastic"),
        region_stats(ndvi, plastic > 0, "all plastic rows"),
        region_stats(ndvi, veg, "all vegetation"),
    ]

    print("  NDRE by region:")
    ndre_stats = [
        region_stats(ndre, np.ones_like(ndre, bool), "all"),
        region_stats(ndre, regions["plants"], "plants (plastic AND veg)"),
        region_stats(ndre, regions["weeds"], "weeds (veg off plastic)"),
        region_stats(ndre, regions["soil"], "soil"),
        region_stats(ndre, regions["plastic_only"], "bare plastic"),
        region_stats(ndre, plastic > 0, "all plastic rows"),
        region_stats(ndre, veg, "all vegetation"),
    ]

    if write_images:
        cv2.imwrite(str(out_dir / f"{stem}_rgb.jpg"), rgb_reg)
        marker = load_m3m_calibration().CALIBRATION_ID
        (out_dir / f"{stem}_rgb.calibration").write_text(marker + "\n", encoding="utf-8")
        cv2.imwrite(str(out_dir / f"{stem}_ndvi.png"), colorize_index(ndvi, colorbar=True))
        cv2.imwrite(
            str(out_dir / f"{stem}_ndre.png"),
            colorize_index(ndre, vmin=-0.1, vmax=0.6, colorbar=True),
        )

    region_rows = build_region_rows(
        stem, ndvi_stats, regions, geo, "ndvi", vigor_estimate
    )
    ndre_region_rows = build_region_rows(
        stem, ndre_stats, regions, geo, "ndre", vigor_estimate_ndre
    )

    with open(out_dir / f"{stem}_ndvi_regions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGION_CSV_FIELDS)
        w.writeheader()
        w.writerows(region_rows)

    with open(out_dir / f"{stem}_ndre_regions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NDRE_REGION_CSV_FIELDS)
        w.writeheader()
        w.writerows(ndre_region_rows)

    tile_rows = tile_ndvi_rows(
        stem, ndvi, regions["plants"], geo, bands, TILE_SIZE, plastic_mask=plastic
    )
    with open(out_dir / f"{stem}_ndvi_tiles.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TILE_CSV_FIELDS)
        w.writeheader()
        w.writerows(tile_rows)

    ndre_tile_rows = tile_ndre_rows(
        stem, ndre, regions["plants"], geo, bands, TILE_SIZE, plastic_mask=plastic
    )
    with open(out_dir / f"{stem}_ndre_tiles.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NDRE_TILE_CSV_FIELDS)
        w.writeheader()
        w.writerows(ndre_tile_rows)

    hotspot_rows = []
    if hotspot_dir is not None and hotspot_cfg is not None:
        hotspot_rows = select_hotspot_tiles(tile_rows, ndre_tile_rows, hotspot_cfg)
        write_hotspot_outputs(
            stem,
            rgb_reg,
            ndvi,
            ndre,
            hotspot_rows,
            hotspot_dir,
            write_images=True,
        )

    plant_rows = []
    if detect_plants:
        args_pc1 = pc1_args or make_pc1_args()
        points, typical, scale, plant_mask, _src = resolve_plant_detections(
            stem, rgb_reg, input_dir, args_pc1, reuse_plants
        )
        radius = canopy_radius_px(typical, scale, plant_radius)
        plant_rows = plant_index_rows(
            stem,
            points,
            ndvi,
            ndre,
            geo,
            radius,
            plant_mask,
            veg,
            plastic,
            hotspot_cfg=hotspot_cfg,
        )
        with open(out_dir / f"{stem}_plants.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PLANT_CSV_FIELDS)
            w.writeheader()
            w.writerows(plant_rows)
        n_stress = sum(1 for r in plant_rows if r.get("condition") == "stressed")
        print(
            f"  localised NDVI/NDRE: {len(plant_rows)} plants  "
            f"radius={radius}px  stressed={n_stress}"
        )
        if write_images:
            cv2.imwrite(
                str(out_dir / f"{stem}_plants.jpg"),
                draw_plant_overlay(rgb_reg, plant_rows, radius),
            )

    return region_rows, tile_rows, ndre_region_rows, ndre_tile_rows, hotspot_rows, plant_rows


def process_folder(in_dir: Path, args, pc1_args, output_override=None, hotspot_override=None, custom_filter=None):
    out_dir = (output_override or (in_dir / "output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hotspot_dir = None
    hotspot_cfg, _cfg_path = load_hotspot_config(in_dir)
    if not args.no_hotspots:
        hotspot_dir = (
            hotspot_override or (in_dir / DEFAULT_HOTSPOT_DIRNAME)
        ).resolve()
        hotspot_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"  hotspot overlays -> {hotspot_dir}  "
            f"(NDVI < {hotspot_cfg['ndvi']['stressed_max']}, "
            f"NDRE < {hotspot_cfg['ndre']['stressed_max']}, "
            f"rule={hotspot_cfg.get('stress_rule')})"
        )

    filter_stems = None
    filter_dir = None
    if not args.all_images:
        filter_dir = resolve_filter_dir(in_dir, custom_filter)
        if filter_dir is not None:
            filter_stems = collect_clean_stems(filter_dir)
            if not filter_stems:
                print(f"  skip {in_dir.name}: no image stems under {filter_dir}")
                return

    sets = find_set(in_dir, filter_stems=filter_stems)
    if not sets:
        if filter_stems is not None:
            print(
                f"  skip {in_dir.name}: no matching JPG+MS sets for "
                f"{len(filter_stems)} stem(s) from {filter_dir}"
            )
            return
        print(f"  skip {in_dir.name}: no JPG+MS sets found")
        return
    if args.limit > 0:
        sets = sets[: args.limit]

    print(f"Processing {len(sets)} captures from {in_dir}")
    if filter_stems is not None:
        print(f"  filtered by clean folder: {filter_dir} ({len(filter_stems)} listed)")

    ndvi_region_batch = out_dir / "ndvi_regions_batch.csv"
    ndvi_tile_batch = out_dir / "ndvi_tile_summary_100.csv"
    ndre_region_batch = out_dir / "ndre_regions_batch.csv"
    ndre_tile_batch = out_dir / "ndre_tile_summary_100.csv"
    hotspot_batch = hotspot_dir / "hotspot_tiles_batch.csv" if hotspot_dir else None
    plant_batch = out_dir / "plant_ndvi_ndre_batch.csv" if not args.no_plants else None

    extra_files = []
    if hotspot_batch is not None:
        extra_files.append(open(hotspot_batch, "w", encoding="utf-8", newline=""))
    else:
        extra_files.append(None)
    if plant_batch is not None:
        extra_files.append(open(plant_batch, "w", encoding="utf-8", newline=""))
    else:
        extra_files.append(None)
    hotspot_file, plant_file = extra_files

    try:
        with open(ndvi_region_batch, "w", encoding="utf-8", newline="") as nrf, open(
            ndvi_tile_batch, "w", encoding="utf-8", newline=""
        ) as ntf, open(ndre_region_batch, "w", encoding="utf-8", newline="") as erf, open(
            ndre_tile_batch, "w", encoding="utf-8", newline=""
        ) as etf:
            ndvi_region_writer = csv.DictWriter(nrf, fieldnames=REGION_CSV_FIELDS)
            ndvi_tile_writer = csv.DictWriter(ntf, fieldnames=TILE_CSV_FIELDS)
            ndre_region_writer = csv.DictWriter(erf, fieldnames=NDRE_REGION_CSV_FIELDS)
            ndre_tile_writer = csv.DictWriter(etf, fieldnames=NDRE_TILE_CSV_FIELDS)
            ndvi_region_writer.writeheader()
            ndvi_tile_writer.writeheader()
            ndre_region_writer.writeheader()
            ndre_tile_writer.writeheader()
            hotspot_writer = None
            if hotspot_file is not None:
                hotspot_writer = csv.DictWriter(
                    hotspot_file, fieldnames=HOTSPOT_CSV_FIELDS
                )
                hotspot_writer.writeheader()
            plant_writer = None
            if plant_file is not None:
                plant_writer = csv.DictWriter(plant_file, fieldnames=PLANT_CSV_FIELDS)
                plant_writer.writeheader()

            for i, (stem, jpg, bands) in enumerate(sets, 1):
                print(f"\n[{i}/{len(sets)}]", end="")
                try:
                    (
                        region_rows,
                        tile_rows,
                        ndre_region_rows,
                        ndre_tile_rows,
                        hotspot_rows,
                        plant_rows,
                    ) = process_one(
                        stem,
                        jpg,
                        bands,
                        out_dir,
                        write_images=not args.no_images,
                        hotspot_dir=hotspot_dir,
                        hotspot_cfg=hotspot_cfg,
                        detect_plants=not args.no_plants,
                        pc1_args=pc1_args,
                        plant_radius=args.plant_radius,
                        reuse_plants=args.reuse_plants,
                        input_dir=in_dir,
                    )
                    ndvi_region_writer.writerows(region_rows)
                    ndvi_tile_writer.writerows(tile_rows)
                    ndre_region_writer.writerows(ndre_region_rows)
                    ndre_tile_writer.writerows(ndre_tile_rows)
                    if hotspot_writer is not None:
                        hotspot_writer.writerows(hotspot_rows)
                        hotspot_file.flush()
                    if plant_writer is not None:
                        plant_writer.writerows(plant_rows)
                        plant_file.flush()
                    nrf.flush()
                    ntf.flush()
                    erf.flush()
                    etf.flush()
                except Exception as e:
                    print(f"  FAILED {stem}: {e}")
                    continue
    finally:
        if hotspot_file is not None:
            hotspot_file.close()
        if plant_file is not None:
            plant_file.close()

    print(f"\nDone. Outputs in {out_dir.resolve()}")
    print(f"  NDVI regions: {ndvi_region_batch}")
    print(f"  NDVI tiles:   {ndvi_tile_batch}")
    print(f"  NDRE regions: {ndre_region_batch}")
    print(f"  NDRE tiles:   {ndre_tile_batch}")
    if plant_batch is not None:
        print(f"  Plant NDVI/NDRE: {plant_batch}")
    if hotspot_dir is not None:
        print(f"  Hotspot overlays: {hotspot_dir}")
        print(f"  Hotspot tiles:    {hotspot_batch}")


def main():
    ap = argparse.ArgumentParser(
        description="Batch RGB-MS NDVI/NDRE with lat/lon CSV export"
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root folder that holds client capture folders "
        f"(default: {DEFAULT_DATA_ROOT})",
    )
    ap.add_argument(
        "--input",
        type=Path,
        nargs="*",
        default=None,
        help="Client folder name(s) or path(s) under --data-root. "
        "Default: every client folder in --data-root.",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--hotspot-dir",
        type=Path,
        default=None,
        help="Folder for hotspot overlay images (default: <input>/hotspot-tiles)",
    )
    ap.add_argument(
        "--no-hotspots",
        action="store_true",
        help="Skip hotspot tile overlays",
    )
    ap.add_argument(
        "--filter-dir",
        type=Path,
        default=None,
        help="Only process captures listed under this folder (e.g. RAVINDRA/clean). "
        "Defaults to <input>/clean when that folder exists.",
    )
    ap.add_argument(
        "--all-images",
        action="store_true",
        help="Process every capture in --input, ignore clean/ filter",
    )
    ap.add_argument("--no-images", action="store_true", help="Skip overlay/preview images")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N sets (0=all)")
    ap.add_argument(
        "--no-plants",
        action="store_true",
        help="Skip plant-count1 detection and localised NDVI/NDRE",
    )
    ap.add_argument(
        "--reuse-plants",
        action="store_true",
        help="Reuse plant-count-output/<stem>_rgb.json points when they fit the MS grid",
    )
    ap.add_argument(
        "--cluster-px",
        type=int,
        default=PLANT_CLUSTER_PX,
        help="plant-count1 within-image NMS radius in MS pixels",
    )
    ap.add_argument(
        "--plant-radius",
        type=int,
        default=0,
        help="local NDVI/NDRE disk radius in MS pixels (0=auto from canopy size)",
    )
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    folders = discover_client_folders(data_root, args.input)
    print(f"Data root: {data_root}")
    print(f"Client folders ({len(folders)}): " + ", ".join(p.name for p in folders))

    pc1_args = None
    if not args.no_plants:
        pc1_args = make_pc1_args(cluster_px=args.cluster_px)
        print(
            f"  plant NDVI/NDRE: plant-count1  cluster={args.cluster_px}px  "
            f"radius={'auto' if args.plant_radius <= 0 else f'{args.plant_radius}px'}"
            f"{'  reuse-json' if args.reuse_plants else ''}"
        )

    single = len(folders) == 1
    for folder in folders:
        print(f"\n######## {folder.name} ########")
        process_folder(
            folder,
            args,
            pc1_args,
            output_override=args.output if single else None,
            hotspot_override=args.hotspot_dir if single else None,
            custom_filter=args.filter_dir if single else None,
        )


if __name__ == "__main__":
    main()
