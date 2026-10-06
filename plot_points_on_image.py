#!/usr/bin/env python3
"""
Draw your GPS coordinates directly ON the drone photo(s), joined in order.

Each coordinate is converted GPS -> pixel with the photo's own metadata (same
calculation as plant_count.py), so you can see where a point really lands in
the picture. The drone's own GPS position is also marked ("drone GPS"); for a
camera pointing straight down it should land near the image centre.

Usage
  python plot_points_on_image.py IMAGE.JPG --close --points pin1:20.3559923,73.9056710 pin2:20.3561594,73.9057928
  python plot_points_on_image.py "Laxman Mahale-1" --points-file points.txt --close

Images: original DJI_xxxx_D.JPG files or output/final_overlays/*_final.jpg,
a folder of them, or several files. The matching DJI_xxxx_MS_NIR.TIF and
_MS_R.TIF must be in the raw folder.
Points: label:lat,lon   label,lat,lon   "label lat, lon"   or just lat,lon
        --points-file: one point per line, e.g.  edge1 20.35599273, 73.90567189

Saves <image folder>/points_on_image/<image>_points.jpg ; a single image is
also shown in a window (any key closes it). --no-show only saves.
Requires: opencv-python, numpy (no other scripts needed).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# RGB camera georeference, included here so this script does not depend on the
# versions of gps_viewer.py / image-processing.py you have
# ---------------------------------------------------------------------------
import json
import math

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



class Georef:
    """lat/lon <-> pixel of the displayed image (original or *_final.jpg)."""

    def __init__(self, img_path: Path, shown_shape):
        name = img_path.name
        if name.lower().endswith("_final.jpg"):
            stem, src = name[: -len("_final.jpg")], img_path.parent.parent.parent
        else:
            stem, src = name.rsplit("_D.", 1)[0], img_path.parent
        rgb_path = next((src / f"{stem}{e}" for e in ("_D.JPG", "_D.jpg")
                         if (src / f"{stem}{e}").exists()), None)
        if rgb_path is None:
            raise RuntimeError(f"cannot find the original photo {stem}_D.JPG in {src}")
        g0 = rgb_georef(rgb_path)
        off = heading_offset_for(load_heading_calibration(src), g0["yaw_meta_deg"])
        self.g = rgb_georef(rgb_path, (g0["h"], g0["w"]), off)
        self.geo = {"lat": self.g["lat"], "lon": self.g["lon"], "alt_m": self.g["alt_m"],
                    "yaw_deg": self.g["yaw_deg"]}
        self.k = self.g["w"] / shown_shape[1]

    def to_px(self, lat, lon):
        xs, ys = latlon_to_rgb_px(self.g, [lat], [lon])
        return float(xs[0]) / self.k, float(ys[0]) / self.k


DEC = r"[-+]?\d+\.\d+"
PAIR = re.compile(r"(?<![\w.])(" + DEC + r")\s*[,; ]\s*(" + DEC + r")")
POINT = (0, 0, 255)       # red (BGR)
LINE = (0, 255, 255)      # yellow
DRONE = (255, 200, 0)     # light blue


def parse_line(line: str, n: int):
    m = PAIR.search(line)
    if not m:
        return None
    name = line[: m.start()].strip(" -:\t,") or f"P{n}"
    return name, float(m.group(1)), float(m.group(2))


def collect_images(paths):
    out = []
    for p in paths:
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.is_file() and not q.name.startswith("._")
                          and (q.name.lower().endswith("_d.jpg")
                               or q.name.lower().endswith("_final.jpg")))
        elif p.is_file():
            out.append(p)
        else:
            print(f"  not found: {p}")
    return out


def text(img, s, org, fs, color):
    t = max(1, int(round(2 * fs)))
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), t + 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, t, cv2.LINE_AA)


def draw(img_path: Path, pts, close: bool):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError("cannot open image")
    g = Georef(img_path, img.shape)
    geo = g.geo
    to_px = g.to_px
    k = g.k

    h, w = img.shape[:2]
    fs = max(0.8, w / 2600 * 1.4)
    lw = max(2, int(round(w / 1300)))
    P = [(name, *to_px(lat, lon)) for name, lat, lon in pts]

    # joined lines (cv2 clips anything outside the picture)
    seq = P + ([P[0]] if close and len(P) > 2 else [])
    for (_, x1, y1), (_, x2, y2) in zip(seq, seq[1:]):
        if all(np.isfinite(v) and abs(v) < 1e6 for v in (x1, y1, x2, y2)):
            cv2.line(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
                     LINE, lw, cv2.LINE_AA)

    rep = []
    for name, x, y in P:
        inside = 0 <= x < w and 0 <= y < h
        rep.append((name, x * k, y * k, inside))
        if inside:
            c = (int(round(x)), int(round(y)))
            r = max(8, int(10 * fs))
            cv2.circle(img, c, r, POINT, -1, cv2.LINE_AA)
            cv2.circle(img, c, r, (255, 255, 255), max(1, lw // 2), cv2.LINE_AA)
            text(img, name, (c[0] + r + 6, c[1] - r), fs, (255, 255, 255))

    # where the drone says it was (should be close to the image centre)
    dx, dy = to_px(geo["lat"], geo["lon"])
    if 0 <= dx < w and 0 <= dy < h:
        c = (int(round(dx)), int(round(dy)))
        cv2.drawMarker(img, c, DRONE, cv2.MARKER_TILTED_CROSS, int(40 * fs), lw + 1)
        text(img, "drone GPS", (c[0] + 14, c[1] + int(34 * fs)), fs * 0.8, DRONE)

    text(img, f"{img_path.name} | heading {geo['yaw_deg']:.1f} deg | "
              f"height {geo['alt_m']:.1f} m", (12, int(40 * fs)), fs * 0.9, (255, 255, 255))
    return img, rep, (dx * k, dy * k)


def main():
    ap = argparse.ArgumentParser(description="Draw GPS coordinates on drone images")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--points", nargs="*", default=[])
    ap.add_argument("--points-file", type=Path)
    ap.add_argument("--close", action="store_true", help="join last point back to first")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    pts = []
    for s in args.points:
        p = parse_line(s, len(pts) + 1)
        print(f"  could not read point '{s}'") if p is None else pts.append(p)
    if args.points_file:
        for line in args.points_file.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.strip().startswith("#"):
                p = parse_line(line, len(pts) + 1)
                print(f"  skipped line: {line.strip()}") if p is None else pts.append(p)
    if not pts:
        raise SystemExit("no points given (use --points or --points-file)")

    imgs = collect_images(args.images)
    if not imgs:
        raise SystemExit("no images found")
    shown = []
    for i, path in enumerate(imgs, 1):
        print(f"[{i}/{len(imgs)}] {path.name}")
        try:
            img, rep, (dx, dy) = draw(path, pts, args.close)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        for name, x, y, inside in rep:
            print(f"  {name:>12}: pixel ({x:7.0f}, {y:7.0f})  {'' if inside else '(outside this image)'}")
        print(f"  drone GPS   : pixel ({dx:7.0f}, {dy:7.0f})")
        out_dir = path.parent / "points_on_image"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"{path.stem}_points.jpg"
        save = img if img.shape[1] <= 2600 else cv2.resize(
            img, (2600, int(img.shape[0] * 2600 / img.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), save, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  saved {out}")
        shown.append(save)

    if shown and len(imgs) == 1 and not args.no_show:
        win = "points on image (any key to close)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        s = min(1500 / shown[0].shape[1], 900 / shown[0].shape[0], 1.0)
        cv2.resizeWindow(win, int(shown[0].shape[1] * s), int(shown[0].shape[0] * s))
        cv2.imshow(win, shown[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
