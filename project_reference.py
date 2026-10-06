#!/usr/bin/env python3
"""
Put the plants of an earlier flight onto the photos of a new flight.

Rules (each reference plant ends up drawn at most ONCE, on its best photo):
  1. Only the multispectral (TIF) area of each photo is used, like plant_count.py.
  2. A point must fall on a plastic mulch row (near the row centre); points on
     soil, weeds or between rows are rejected.
  3. A plant usually falls in several photos. It is kept only in the photo where
     there is plant evidence at the point (green leaves, or a planting hole for
     seedlings) and where that evidence is strongest; points with no evidence in
     any photo are discarded.
  Before these checks, a point that is slightly off (within --snap-cm, default
  15 cm) is moved to the nearest visible plant on the row, if no other point
  has already taken that plant.
  4. Two kept plants can never be closer than a minimum gap (default 60% of the
     plant spacing measured along the rows); the weaker one is dropped.

Usage
  python project_reference.py REFERENCE.csv NEW_BATCH_FOLDER
  python project_reference.py REF.csv NEW --offset-from LAT,LON --offset-to LAT,LON
  (REFERENCE.csv = plants_unique.csv of the first batch)

Output in <new folder>/output_reference/
  overlays/<image>_ref.jpg   marked photos (only the plants assigned to that photo)
  reference_plants.csv       per reference plant: status, photo, pixel, evidence
  reference_frames.csv       per photo: points in TIF area / on rows / kept
  reference_map.html         map of the kept plants (double-click)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("plant_count", ROOT / "plant_count.py")
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)
ip, pm = pc.ip, pc.pm

KEPT = (0, 255, 255)          # yellow (BGR)

# ---------------------------------------------------------------------------
# RGB camera georeference, included here so this script works with any
# version of image-processing.py
# ---------------------------------------------------------------------------
import json
import re

def _xmp_dict(path):
    raw = Path(path).read_bytes()[:600_000]
    i = raw.find(b"<x:xmpmeta")
    j = raw.find(b"</x:xmpmeta>", i)
    if i < 0 or j < 0:
        return {}
    x = raw[i:j].decode("utf-8", "replace")
    d = dict(re.findall(r'drone-dji:([A-Za-z0-9]+)="([^"]*)"', x))
    d.update(re.findall(r"<drone-dji:([A-Za-z0-9]+)>([^<]*)</drone-dji:", x))
    return d


def rgb_georef(jpg_path, shape=None, yaw_offset_deg=0.0):
    """Camera model of one RGB photo. shape=(h, w) of the image (read if None)."""
    d = _xmp_dict(jpg_path)

    def f(k, default=None):
        try:
            return float(d[k])
        except (KeyError, ValueError):
            return default
    if shape is None:
        im = cv2.imread(str(jpg_path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        shape = (im.shape[0] * 8, im.shape[1] * 8) if im is not None else (3956, 5280)
    h, w = int(shape[0]), int(shape[1])
    lat, lon = f("GpsLatitude", f("Latitude")), f("GpsLongitude", f("GpsLongtitude", f("Longitude")))
    if lat is None or lon is None:
        raise RuntimeError(f"no GPS in {Path(jpg_path).name}")
    yaw = f("FlightYawDegree", f("GimbalYawDegree", 0.0))
    alt = f("RelativeAltitude", 0.0)
    K = D = None
    dw = d.get("DewarpData", "")
    if ";" in dw:
        try:
            fx, fy, dcx, dcy, k1, k2, p1, p2, k3 = (float(v) for v in dw.split(";")[1].split(",")[:9])
            K = np.array([[fx, 0, w / 2 + dcx], [0, fy, h / 2 + dcy], [0, 0, 1.0]])
            D = np.array([k1, k2, p1, p2, k3])
        except ValueError:
            K = None
    if K is None:
        fpx = f("CalibratedFocalLength", 0.7 * w)
        K = np.array([[fpx, 0, f("CalibratedOpticalCenterX", w / 2)],
                      [0, fpx, f("CalibratedOpticalCenterY", h / 2)], [0, 0, 1.0]])
        D = np.zeros(5)
    return {"lat": lat, "lon": lon, "alt_m": alt, "yaw_meta_deg": yaw,
            "yaw_deg": yaw + yaw_offset_deg, "yaw_offset_deg": yaw_offset_deg,
            "K": K, "D": D, "w": w, "h": h, "rtk_flag": d.get("RtkFlag", ""),
            "pitch_deg": f("GimbalPitchDegree"), "gsd_m": alt / K[0, 0] if alt else 0.0}


def rgb_px_to_latlon(g, xs, ys):
    """RGB pixel(s) -> (lat, lon) arrays. Flat ground, nadir camera."""
    pts = np.column_stack([np.atleast_1d(xs), np.atleast_1d(ys)]).astype(np.float64)
    n = cv2.undistortPoints(pts.reshape(-1, 1, 2), g["K"], g["D"]).reshape(-1, 2)
    dx, dy = n[:, 0] * g["alt_m"], n[:, 1] * g["alt_m"]
    yaw = math.radians(g["yaw_deg"])
    c, s = math.cos(yaw), math.sin(yaw)
    east = dx * c - dy * s
    north = -dx * s - dy * c
    return (g["lat"] + north / 111320.0,
            g["lon"] + east / (111320.0 * math.cos(math.radians(g["lat"]))))


def latlon_to_rgb_px(g, lats, lons):
    """Inverse of rgb_px_to_latlon: (lat, lon) -> RGB pixel (x, y) arrays."""
    lats, lons = np.atleast_1d(lats).astype(np.float64), np.atleast_1d(lons).astype(np.float64)
    north = (lats - g["lat"]) * 111320.0
    east = (lons - g["lon"]) * 111320.0 * math.cos(math.radians(g["lat"]))
    yaw = math.radians(g["yaw_deg"])
    c, s = math.cos(yaw), math.sin(yaw)
    dx = east * c - north * s
    dy = -east * s - north * c
    alt = g["alt_m"] or 1.0
    obj = np.column_stack([dx / alt, dy / alt, np.ones(len(dx))]).reshape(-1, 1, 3)
    px, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), g["K"], g["D"])
    px = px.reshape(-1, 2)
    return px[:, 0], px[:, 1]


def load_heading_calibration(folder):
    """Look for heading_calibration.json in <folder>/output/ or <folder>/."""
    folder = Path(folder)
    for p in (folder / "output" / "heading_calibration.json", folder / "heading_calibration.json"):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def heading_offset_for(cal, yaw_meta_deg, jpg_dir=None):
    """Offset for a photo: median of calibration pairs flown in a similar direction
    (within 20 deg), else the overall median. cal = dict or None (-> 0)."""
    if not cal or cal.get("offset_deg") is None:
        return 0.0
    near = []
    for p in cal.get("pairs", []):
        y = p.get("yaw_deg")
        if y is not None and abs((y - yaw_meta_deg + 180) % 360 - 180) <= 20:
            near.append(p["offset_deg"])
    if len(near) >= 3:
        return float(np.median(near))
    return float(cal["offset_deg"])


GEO = {n: globals()[n] for n in ("rgb_georef", "latlon_to_rgb_px",
                                 "load_heading_calibration", "heading_offset_for")}



def load_frame(jpg, bands, heading_cal, cfg):
    """RGB, camera model, RGB->MS homography, plastic row mask (native px)."""
    rgb = cv2.imread(str(jpg))
    nir = cv2.imread(str(bands["NIR"]), cv2.IMREAD_UNCHANGED)
    red = cv2.imread(str(bands["R"]), cv2.IMREAD_UNCHANGED)
    re_ = cv2.imread(str(bands["RE"]), cv2.IMREAD_UNCHANGED) if "RE" in bands else None
    if rgb is None or nir is None or red is None:
        raise RuntimeError("could not read the photo or its bands")
    g0 = GEO["rgb_georef"](jpg, rgb.shape[:2])
    g = GEO["rgb_georef"](jpg, rgb.shape[:2], GEO["heading_offset_for"](heading_cal, g0["yaw_meta_deg"]))
    with contextlib.redirect_stdout(io.StringIO()):
        rgb_reg, aux = ip.register_rgb_to_ms(rgb, nir, refine_band=red)
        if not (isinstance(aux, np.ndarray) and aux.shape == (3, 3)):
            raise RuntimeError("RGB / multispectral alignment failed")
        H = aux @ np.diag([nir.shape[1] / rgb.shape[1], nir.shape[0] / rgb.shape[0], 1.0])
        plastic_ms, _ = pc.segment_plastic_rows(rgb_reg, cfg)
    h, w = rgb.shape[:2]
    plast = cv2.warpPerspective(plastic_ms, H, (w, h),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    ms_area = cv2.warpPerspective(np.full(nir.shape[:2], 255, np.uint8), H, (w, h),
                                  flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    ndvi = ip.compute_ndvi(red, nir)
    ndre = ip.compute_ndre(re_, nir) if re_ is not None else None
    return rgb, g, H, plast, ms_area, ndvi, ndre


def evidence_maps(rgb, gsd, cfg, radius_px):
    """Per pixel: green fraction around it, and strongest dark spot around it."""
    b, gch, r = (c.astype(np.int16) for c in cv2.split(rgb))
    green = ((2 * gch - r - b) > float(cfg.get("native_exg_min", 20))).astype(np.float32)
    green_k = cv2.blur(green, (2 * radius_px + 1, 2 * radius_px + 1))
    hole_px = max(3.0, cfg.get("hole_diameter_cm", 8) / 100.0 / max(gsd, 1e-6))
    k = int(round(3 * hole_px)) | 1
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    dark_k = cv2.dilate(bh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (2 * radius_px + 1, 2 * radius_px + 1)))
    return green_k, dark_k, green > 0, bh, hole_px


def plant_candidates(green, bh, plast, hole_px, dark_min):
    """Centres of visible plants on the rows: dark planting holes (seedlings) and
    green leaf blobs. Used to snap reference points that are slightly off."""
    pts = []
    dark = ((bh > dark_min) & plast).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    a_min, a_max = math.pi * (0.3 * hole_px / 2) ** 2, math.pi * (3.0 * hole_px / 2) ** 2
    for i in range(1, n):
        if a_min <= st[i, 4] <= a_max:
            pts.append(cen[i])
    gm = (green & plast).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(gm, 8)
    for i in range(1, n):
        if st[i, 4] >= a_min:
            pts.append(cen[i])
    if not pts:
        return np.empty((0, 2))
    pts = np.array(pts, float)
    keep = []                                    # one candidate per plant
    for p in pts:
        if all(np.hypot(*(p - q)) >= hole_px for q in keep):
            keep.append(p)
    return np.array(keep)


def main():
    ap = argparse.ArgumentParser(description="Project reference plants onto a new batch")
    ap.add_argument("reference", type=Path, help="plants_unique.csv from the first batch")
    ap.add_argument("folder", type=Path, help="new batch folder with raw DJI files")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--radius-cm", type=float, default=6.0,
                    help="area checked for plant evidence around each point (default 6 cm)")
    ap.add_argument("--snap-cm", type=float, default=15.0,
                    help="move a point to the nearest visible plant within this distance "
                         "(default 15 cm, 0 = off)")
    ap.add_argument("--min-gap-m", type=float, default=None,
                    help="minimum distance between two kept plants (default: 0.6 x spacing)")
    ap.add_argument("--offset-from", default=None, metavar="LAT,LON")
    ap.add_argument("--offset-to", default=None, metavar="LAT,LON")
    ap.add_argument("--offset-m", default=None, metavar="EAST,NORTH")
    ap.add_argument("--no-frame-corrections", action="store_true")
    args = ap.parse_args()

    cfg = pc.load_config(args.config)
    green_min = 0.15
    dark_min = float(cfg.get("dark_spot_min_contrast", 30))

    ref = [r for r in csv.DictReader(open(args.reference, newline="", encoding="utf-8"))
           if r.get("lat") and r.get("lon")]
    if not ref:
        raise SystemExit("no lat/lon rows in the reference CSV")
    ref_ll = np.array([[float(r["lat"]), float(r["lon"])] for r in ref])
    uid = [r.get("plant_uid", str(i)) for i, r in enumerate(ref)]
    lat_c = float(ref_ll[:, 0].mean())
    kx = 111320.0 * math.cos(math.radians(lat_c))

    off_e = off_n = 0.0
    if args.offset_from and args.offset_to:
        (la1, lo1), (la2, lo2) = ([float(v) for v in t.replace(" ", "").split(",")]
                                  for t in (args.offset_from, args.offset_to))
        off_e, off_n = (lo2 - lo1) * kx, (la2 - la1) * 111320.0
    elif args.offset_m:
        off_e, off_n = (float(v) for v in args.offset_m.replace(" ", "").split(","))
    if off_e or off_n:
        ref_ll[:, 0] += off_n / 111320.0
        ref_ll[:, 1] += off_e / kx
        print(f"applying offset: {off_e:+.2f} m east, {off_n:+.2f} m north")
    print(f"reference: {len(ref)} plants from {args.reference.name}")

    # per-frame corrections of the new batch (undo them before using raw GPS)
    fcorr = {}
    fa = args.folder / "output" / "frame_alignment.csv"
    if fa.exists() and not args.no_frame_corrections:
        for r in csv.DictReader(open(fa, newline="", encoding="utf-8")):
            try:
                S = np.array([[float(r["corr_a"]), float(r["corr_b"]), float(r["corr_c"])],
                              [float(r["corr_d"]), float(r["corr_e"]), float(r["corr_f"])],
                              [0, 0, 1.0]])
                fcorr[r["image_id"]] = (np.linalg.inv(S), float(r["origin_lat"]),
                                        float(r["origin_lon"]))
            except (KeyError, ValueError):
                pass
    heading_cal = GEO["load_heading_calibration"](args.folder)

    sets = [s for s in ip.find_set(args.folder) if not str(s[0]).startswith("._")
            and not Path(s[1]).name.startswith("._")]
    print(f"new batch: {len(sets)} images in {args.folder.name}")
    out = args.folder / "output_reference"
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    # ---------------- pass 1: candidates per frame
    cand = defaultdict(list)        # plant index -> [(score, centrality, stem, x, y)]
    frame_rows, frame_info, spacings = [], {}, []
    stat = defaultdict(lambda: {"in_tif": 0, "on_rows": 0})
    for i, (stem, jpg, bands) in enumerate(sets, 1):
        print(f"  [{i}/{len(sets)}] {stem}", end=" ... ", flush=True)
        try:
            rgb, g, H, plast, ms_area, ndvi, ndre = load_frame(jpg, bands, heading_cal, cfg)
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue
        h, w = rgb.shape[:2]
        pts_ll = ref_ll
        if stem in fcorr:
            Si, la0, lo0 = fcorr[stem]
            k_ = 111320.0 * math.cos(math.radians(la0))
            e2, n2, _ = Si @ np.vstack([(ref_ll[:, 1] - lo0) * k_, (ref_ll[:, 0] - la0) * 111320.0,
                                        np.ones(len(ref_ll))])
            pts_ll = np.column_stack([la0 + n2 / 111320.0, lo0 + e2 / k_])
        X, Y = GEO["latlon_to_rgb_px"](g, pts_ll[:, 0], pts_ll[:, 1])
        X, Y = np.array(X, float), np.array(Y, float)
        X0, Y0 = X.copy(), Y.copy()
        xi = np.clip(np.round(X).astype(int), 0, w - 1)
        yi = np.clip(np.round(Y).astype(int), 0, h - 1)
        inside = (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
        idx = np.nonzero(inside & ms_area[yi, xi])[0]          # rule 1: TIF area only
        st = stat[stem]
        st["in_tif"] = len(idx)
        frame_info[stem] = (jpg, plast, ms_area)

        rad = max(2, int(round(args.radius_cm / 100.0 / max(g["gsd_m"], 1e-6))))
        green_k, dark_k, green_m, bh, hole_px = evidence_maps(rgb, g["gsd_m"], cfg, rad)
        snap = {}
        if args.snap_cm > 0 and len(idx):
            C = plant_candidates(green_m, bh, plast, hole_px, dark_min)
            snap_px = args.snap_cm / 100.0 / max(g["gsd_m"], 1e-6)
            if len(C):
                pairs = []                      # (distance, point, candidate)
                P = np.column_stack([X[idx], Y[idx]])
                for a, j in enumerate(idx):
                    d = np.hypot(*(C - P[a]).T)
                    for c in np.nonzero(d <= snap_px)[0]:
                        pairs.append((float(d[c]), int(j), int(c)))
                pairs.sort()
                used_p, used_c = set(), set()
                for d, j, c in pairs:           # closest first, one point per plant
                    if j in used_p or c in used_c:
                        continue
                    used_p.add(j)
                    used_c.add(c)
                    snap[j] = (C[c, 0], C[c, 1], d * g["gsd_m"])
        st["snapped"] = len(snap)
        inst = []
        for j in idx:
            if j in snap:
                X[j], Y[j] = snap[j][0], snap[j][1]
                xi[j], yi[j] = int(round(X[j])), int(round(Y[j]))
            gv, dv = float(green_k[yi[j], xi[j]]), float(dark_k[yi[j], xi[j]])
            inst.append({"centroid_x": float(X[j]), "centroid_y": float(Y[j]), "j": int(j),
                         "green": gv, "dark": dv,
                         "score": gv / green_min + dv / dark_min})
        # rule 2 (+ rule 4 within the photo): on a row, near its centre, one per position
        kept, _, info = pc.apply_row_constraints(inst, plast, g["gsd_m"], cfg)
        if info.get("plant_spacing_cm"):
            spacings.append(float(info["plant_spacing_cm"]))
        st["on_rows"] = len(kept)
        n_ev = 0
        for p in kept:
            if p["green"] >= green_min or p["dark"] >= dark_min:     # rule 3: evidence
                cen = math.hypot((p["centroid_x"] - w / 2) / (w / 2), (p["centroid_y"] - h / 2) / (h / 2))
                # NDVI / NDRE of this flight: mean over the evidence radius in the MS grid
                q = H @ np.array([p["centroid_x"], p["centroid_y"], 1.0])
                mc, mr = q[0] / q[2], q[1] / q[2]
                rm = max(1, int(round(rad * ndvi.shape[1] / w)))
                r0, r1 = int(mr) - rm, int(mr) + rm + 1
                c0, c1 = int(mc) - rm, int(mc) + rm + 1
                nv = nr = float("nan")
                if 0 <= r0 and r1 <= ndvi.shape[0] and 0 <= c0 and c1 <= ndvi.shape[1]:
                    nv = float(np.nanmean(ndvi[r0:r1, c0:c1]))
                    if ndre is not None:
                        nr = float(np.nanmean(ndre[r0:r1, c0:c1]))
                j = p["j"]
                sd = snap[j][2] if j in snap else 0.0
                la, lo = ref_ll[j]
                if j in snap:   # move the ground position by the same small shift
                    l1, o1 = rgb_px_to_latlon(g, [X0[j], p["centroid_x"]], [Y0[j], p["centroid_y"]])
                    la, lo = la + (l1[1] - l1[0]), lo + (o1[1] - o1[0])
                cand[j].append((p["score"] - sd, -cen, stem, p["centroid_x"], p["centroid_y"],
                                p["green"], p["dark"], nv, nr, sd, la, lo))
                n_ev += 1
        st["with_evidence"] = n_ev
        print(f"{len(idx)} in TIF area, {st['snapped']} snapped to a nearby plant, "
              f"{len(kept)} on rows, {n_ev} with plant evidence")

    # ---------------- rule 3: one photo per plant (strongest evidence)
    best = {j: max(c) for j, c in cand.items()}
    # ---------------- rule 4 across photos: minimum gap on the ground
    spacing_m = (np.median(spacings) / 100.0) if spacings else 0.3
    min_gap = args.min_gap_m if args.min_gap_m is not None else 0.6 * spacing_m
    pos_ll = ref_ll.copy()
    for j, b_ in best.items():
        pos_ll[j] = (b_[10], b_[11])
    E = (pos_ll[:, 1] - ref_ll[:, 1].mean()) * kx
    N = (pos_ll[:, 0] - ref_ll[:, 0].mean()) * 111320.0
    grid, keep = {}, []
    cell = max(min_gap, 0.05)
    too_close = set()
    for j in sorted(best, key=lambda j: -best[j][0]):
        cx_, cy_ = int(E[j] // cell), int(N[j] // cell)
        clash = any(math.hypot(E[j] - E[q], N[j] - N[q]) < min_gap
                    for a in (-1, 0, 1) for b in (-1, 0, 1) for q in grid.get((cx_ + a, cy_ + b), []))
        if clash:
            too_close.add(j)
            continue
        grid.setdefault((cx_, cy_), []).append(j)
        keep.append(j)
    keep_set = set(keep)
    print(f"\nplant spacing ~{spacing_m * 100:.0f} cm | minimum gap {min_gap * 100:.0f} cm")

    # ---------------- pass 2: draw each kept plant on its one photo
    by_frame = defaultdict(list)
    for j in keep:
        by_frame[best[j][2]].append(j)
    for stem, (jpg, plast, ms_area) in frame_info.items():
        rgb = cv2.imread(str(jpg))
        h, w = rgb.shape[:2]
        vis = rgb.copy()
        tint = vis.copy()
        tint[plast] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 0.8, tint, 0.2, 0)
        cnts, _ = cv2.findContours(ms_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 255), max(2, w // 1300))
        r = max(10, w // 300)
        for j in by_frame.get(stem, []):
            x, y = best[j][3], best[j][4]
            cv2.circle(vis, (int(round(x)), int(round(y))), r, KEPT, max(2, w // 1300), cv2.LINE_AA)
        st = stat[stem]
        lines = [stem, f"reference points in TIF area: {st['in_tif']}",
                 f"on plastic rows: {st['on_rows']}  with plant evidence: {st.get('with_evidence', 0)}",
                 f"kept in this photo: {len(by_frame.get(stem, []))}"]
        fs = max(0.8, w / 2600 * 1.3)
        lh = int(40 * fs)
        cv2.rectangle(vis, (0, 0), (int(w * 0.45), lh * len(lines) + 20), (0, 0, 0), -1)
        for k, t in enumerate(lines):
            cv2.putText(vis, t, (12, lh * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), max(1, int(2 * fs)), cv2.LINE_AA)
        if w > 2600:
            vis = cv2.resize(vis, (2600, int(h * 2600 / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "overlays" / f"{stem}_ref.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_rows.append({"image_id": stem, "in_tif_area": st["in_tif"], "on_rows": st["on_rows"],
                           "with_evidence": st.get("with_evidence", 0),
                           "kept_here": len(by_frame.get(stem, []))})

    # ---------------- CSVs + map
    def write(path, rows):
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)

    plant_rows = []
    for j, r in enumerate(ref):
        if j in keep_set:
            sc, _, stem, x, y, gv, dv, nv, nr, sd, la_s, lo_s = best[j]
            status = "found (snapped)" if sd > 0 else "found"
        else:
            stem, x, y, gv, dv, nv, nr, sd = "", "", "", "", "", float("nan"), float("nan"), ""
            la_s, lo_s = ref_ll[j]
            status = ("too close to another plant" if j in too_close else
                      "no plant evidence on a row (or not in any TIF area)")
        plant_rows.append({"plant_uid": uid[j], "lat": f"{la_s:.8f}", "lon": f"{lo_s:.8f}",
                           "ref_lat": f"{ref_ll[j, 0]:.8f}", "ref_lon": f"{ref_ll[j, 1]:.8f}",
                           "status": status, "photo": stem,
                           "snapped_cm": round(sd * 100, 1) if sd != "" else "",
                           "rgb_x_px": round(x, 1) if x != "" else "",
                           "rgb_y_px": round(y, 1) if y != "" else "",
                           "green_fraction": round(gv, 3) if gv != "" else "",
                           "dark_contrast": round(dv, 1) if dv != "" else "",
                           "ndvi": round(nv, 4) if np.isfinite(nv) else "",
                           "ndre": round(nr, 4) if np.isfinite(nr) else "",
                           # values of the same plant in the reference flight
                           "ref_ndvi": r.get("ndvi_mean", ""),
                           "ref_ndre": r.get("ndre_mean", "")})
    write(out / "reference_plants.csv", plant_rows)
    write(out / "reference_frames.csv", frame_rows)
    found = [p for p in plant_rows if p["status"].startswith("found")]
    write(out / "reference_found.csv", found)
    boundary, _ = pc.load_boundary(cfg, args.folder)
    if found:
        pm.write_plant_map(out / "reference_found.csv", out / "reference_map.html", boundary,
                           title=f"Reference plants found on {args.folder.name}")
    n_snap = sum(1 for p in found if p["status"] == "found (snapped)")
    print(f"reference plants: {len(ref)} | found (kept once): {len(found)} "
          f"({n_snap} snapped) | "
          f"too close: {len(too_close)} | no evidence on a row: {len(ref) - len(found) - len(too_close)}")
    print(f"results in {out}")


if __name__ == "__main__":
    main()
