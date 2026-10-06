#!/usr/bin/env python3
"""
GPS image viewer: open a drone image, hover or click anywhere to see the GPS
coordinates of that spot on the ground.

Works with
  * original DJI RGB images         ...\\Laxman Mahale-1\\DJI_xxxx_D.JPG
  * final evidence images            ...\\output\\final_overlays\\DJI_xxxx_final.jpg
  * a folder of either (browse with N / P)

The matching multispectral TIFs (DJI_xxxx_MS_*.TIF) must be in the image folder
(for final images: in the folder above "output"). Their metadata gives the GPS
position, height and heading; the same conversion as plant_count.py is used.

Usage
  python gps_viewer.py "Laxman Mahale-1\\DJI_20260813155544_0001_D.JPG"
  python gps_viewer.py "Laxman Mahale-1\\output\\final_overlays"

Controls
  hover         live coordinates in the bottom bar
  left click    pin a point (lat/lon printed and saved to clicked_points.csv)
  + / -         zoom in / out        (mouse wheel also works on Windows)
  W A S D       pan                  0  reset view
  N / P         next / previous image in the folder
  C             clear pins           X  save a screenshot with the pins
  Q / Esc       quit

Requires: opencv-python, numpy, and image-processing.py next to this script.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ip", ROOT / "image-processing.py")
ip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ip)

VIEW_W, VIEW_H = 1400, 860          # size of the image area in the window
BAR_H = 34
WIN = "GPS viewer"


# ---------------------------------------------------------------------------
# georeferencing for one image
# ---------------------------------------------------------------------------

def find_sources(path: Path):
    """Return (stem, source_folder) for an original or a *_final.jpg image."""
    name = path.name
    if name.lower().endswith("_final.jpg"):
        stem = name[: -len("_final.jpg")]
        src = path.parent.parent.parent          # final_overlays -> output -> folder
    else:
        stem = name.rsplit("_D.", 1)[0]
        src = path.parent
    return stem, src


class Georef:
    """Pixel (of the displayed image) -> lat/lon."""

    def __init__(self, img_path: Path, shown_shape):
        stem, src = find_sources(img_path)
        bands = {b: src / f"{stem}_MS_{b}.TIF" for b in ("NIR", "R")}
        rgb_path = None
        for ext in ("_D.JPG", "_D.jpg"):
            if (src / f"{stem}{ext}").exists():
                rgb_path = src / f"{stem}{ext}"
        missing = [str(p) for p in bands.values() if not p.exists()]
        if missing or rgb_path is None:
            raise RuntimeError("cannot find the original files for this image:\n  "
                               + "\n  ".join(missing or [f"{src / stem}_D.JPG"]))

        nir = cv2.imread(str(bands["NIR"]), cv2.IMREAD_UNCHANGED)
        red = cv2.imread(str(bands["R"]), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(rgb_path))
        if nir is None or red is None or rgb is None:
            raise RuntimeError("could not read the original image files")
        self.geo = ip.georef_from_ms(bands["NIR"], nir.shape)

        print("  aligning RGB to multispectral (one-time, a few seconds) ...")
        _, aux = ip.register_rgb_to_ms(rgb, nir, refine_band=red)
        if not (isinstance(aux, np.ndarray) and aux.shape == (3, 3)):
            raise RuntimeError("RGB / multispectral alignment failed for this image")
        sx, sy = nir.shape[1] / rgb.shape[1], nir.shape[0] / rgb.shape[0]
        self.H = aux @ np.diag([sx, sy, 1.0])          # native RGB px -> MS px
        # displayed image may be a resized copy (final overlays are 2600 px wide)
        self.k = rgb.shape[1] / shown_shape[1] if shown_shape is not None else 1.0
        self.ms_shape = nir.shape[:2]
        self.alt = self.geo.get("alt_m", 0.0)

    def latlon(self, x, y):
        p = self.H @ np.array([x * self.k, y * self.k, 1.0])
        col, row = p[0] / p[2], p[1] / p[2]
        g = self.geo
        lat, lon = ip.pixel_to_latlon(col, row, g["lat"], g["lon"], g["yaw_deg"],
                                      g["gsd_m"], g["cx"], g["cy"])
        inside = 0 <= col < self.ms_shape[1] and 0 <= row < self.ms_shape[0]
        return lat, lon, inside


# ---------------------------------------------------------------------------
# viewer
# ---------------------------------------------------------------------------

class Viewer:
    def __init__(self, files, log_path: Path):
        self.files = files
        self.idx = 0
        self.log_path = log_path
        self.cache = {}
        self.load()
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, VIEW_W, VIEW_H + BAR_H)
        cv2.setMouseCallback(WIN, self.on_mouse)

    # ---- image loading
    def load(self):
        path = self.files[self.idx]
        print(f"\n[{self.idx + 1}/{len(self.files)}] {path.name}")
        self.img = cv2.imread(str(path))
        if self.img is None:
            raise RuntimeError(f"cannot open {path}")
        self.path = path
        self.pins = []
        self.hover = None
        self.zoom, self.cx, self.cy = 1.0, self.img.shape[1] / 2, self.img.shape[0] / 2
        if path not in self.cache:
            try:
                self.cache[path] = Georef(path, self.img.shape)
                g = self.cache[path].geo
                print(f"  image centre {g['lat']:.7f}, {g['lon']:.7f} | height "
                      f"{g['alt_m']:.1f} m | heading {g['yaw_deg']:.1f} deg | "
                      f"{g['gsd_m'] * 100:.2f} cm/px (multispectral)")
            except Exception as exc:
                print(f"  NO GPS for this image: {exc}")
                self.cache[path] = None
        self.gr = self.cache[path]

    # ---- view transform
    def fit_scale(self):
        h, w = self.img.shape[:2]
        return min(VIEW_W / w, VIEW_H / h)

    def scale(self):
        return self.fit_scale() * self.zoom

    def view_to_img(self, mx, my):
        s = self.scale()
        return self.cx + (mx - VIEW_W / 2) / s, self.cy + (my - VIEW_H / 2) / s

    def img_to_view(self, x, y):
        s = self.scale()
        return (x - self.cx) * s + VIEW_W / 2, (y - self.cy) * s + VIEW_H / 2

    def clamp(self):
        h, w = self.img.shape[:2]
        self.zoom = float(np.clip(self.zoom, 1.0, 40.0))
        s = self.scale()
        hw, hh = VIEW_W / 2 / s, VIEW_H / 2 / s
        self.cx = float(np.clip(self.cx, min(hw, w / 2), max(w - hw, w / 2)))
        self.cy = float(np.clip(self.cy, min(hh, h / 2), max(h - hh, h / 2)))

    # ---- mouse
    def on_mouse(self, event, mx, my, flags, _):
        if my >= VIEW_H:
            return
        x, y = self.view_to_img(mx, my)
        h, w = self.img.shape[:2]
        on_img = 0 <= x < w and 0 <= y < h
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = (x, y) if on_img else None
        elif event == cv2.EVENT_LBUTTONDOWN and on_img:
            self.add_pin(x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            f = 1.25 if flags > 0 else 1 / 1.25
            self.zoom_at(x, y, f)

    def zoom_at(self, x, y, f):
        old = self.scale()
        self.zoom *= f
        self.clamp()
        new = self.scale()
        # keep the point under the cursor in place
        self.cx = x - (x - self.cx) * old / new
        self.cy = y - (y - self.cy) * old / new
        self.clamp()

    def add_pin(self, x, y):
        if self.gr is None:
            print("  no GPS available for this image")
            return
        lat, lon, inside = self.gr.latlon(x, y)
        self.pins.append((x, y, lat, lon))
        n = len(self.pins)
        note = "" if inside else "  (outside multispectral view - less accurate)"
        print(f"  pin {n}: {lat:.8f}, {lon:.8f}   pixel ({x * self.gr.k:.0f}, "
              f"{y * self.gr.k:.0f}){note}")
        new = not self.log_path.exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            if new:
                wr.writerow(["time", "image", "pin", "rgb_x_px", "rgb_y_px", "lat", "lon",
                             "inside_ms_view"])
            wr.writerow([datetime.now().isoformat(timespec="seconds"), self.path.name, n,
                         round(x * self.gr.k, 1), round(y * self.gr.k, 1),
                         f"{lat:.8f}", f"{lon:.8f}", inside])

    # ---- drawing
    def render(self):
        s = self.scale()
        M = np.array([[s, 0, VIEW_W / 2 - self.cx * s],
                      [0, s, VIEW_H / 2 - self.cy * s]], np.float32)
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
        view = cv2.warpAffine(self.img, M, (VIEW_W, VIEW_H), flags=interp,
                              borderValue=(30, 30, 30))
        for n, (x, y, lat, lon) in enumerate(self.pins, 1):
            vx, vy = map(int, self.img_to_view(x, y))
            cv2.drawMarker(view, (vx, vy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            label = f"{n}: {lat:.7f}, {lon:.7f}"
            cv2.putText(view, label, (vx + 10, vy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, label, (vx + 10, vy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

        bar = np.full((BAR_H, VIEW_W, 3), 25, np.uint8)
        if self.gr is None:
            txt = "no GPS for this image (original DJI files not found)"
        elif self.hover is None:
            txt = "move the mouse over the image | click to pin | +/- zoom | WASD pan | N/P next/prev | Q quit"
        else:
            lat, lon, inside = self.gr.latlon(*self.hover)
            txt = f"lat {lat:.8f}   lon {lon:.8f}" + ("" if inside else "   (outside MS view)")
        txt = f"{self.path.name}  [{self.idx + 1}/{len(self.files)}]  zoom {self.zoom:.1f}x  |  {txt}"
        cv2.putText(bar, txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
                    cv2.LINE_AA)
        self.frame = np.vstack([view, bar])
        cv2.imshow(WIN, self.frame)

    # ---- main loop
    def run(self):
        step = 0.15
        while True:
            self.render()
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break                                   # window closed
            except cv2.error:
                break
            s = self.scale()
            if k in (ord("+"), ord("=")):
                self.zoom_at(self.cx, self.cy, 1.25)
            elif k in (ord("-"), ord("_")):
                self.zoom_at(self.cx, self.cy, 1 / 1.25)
            elif k == ord("0"):
                self.zoom, self.cx, self.cy = 1.0, self.img.shape[1] / 2, self.img.shape[0] / 2
            elif k in (ord("w"), ord("a"), ord("s"), ord("d")):
                dx = {ord("a"): -1, ord("d"): 1}.get(k, 0) * VIEW_W * step / s
                dy = {ord("w"): -1, ord("s"): 1}.get(k, 0) * VIEW_H * step / s
                self.cx += dx; self.cy += dy
                self.clamp()
            elif k == ord("c"):
                self.pins = []
            elif k == ord("x"):
                out = self.path.with_name(self.path.stem + "_pins.jpg")
                cv2.imwrite(str(out), self.frame)
                print(f"  saved {out}")
            elif k in (ord("n"), ord("p")) and len(self.files) > 1:
                self.idx = (self.idx + (1 if k == ord("n") else -1)) % len(self.files)
                self.load()
        cv2.destroyAllWindows()


def collect(path: Path):
    if path.is_file():
        return [path]
    files = sorted(p for p in path.iterdir()
                   if p.is_file() and not p.name.startswith("._")
                   and (p.name.lower().endswith("_final.jpg")
                        or p.name.lower().endswith("_d.jpg")))
    return files


def recompute(csv_path: Path):
    """Re-calculate lat/lon of points pinned with an older version of this viewer."""
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    cache, out = {}, []
    for r in rows:
        img = csv_path.parent / r["image"]
        if img not in cache:
            try:
                cache[img] = Georef(img, None)
            except Exception as exc:
                print(f"  {r['image']}: {exc}")
                cache[img] = None
        g = cache[img]
        if g is None:
            continue
        lat, lon, inside = g.latlon(float(r["rgb_x_px"]), float(r["rgb_y_px"]))
        old_lat, old_lon = float(r["lat"]), float(r["lon"])
        moved = ((lat - old_lat) * 111320) ** 2 + ((lon - old_lon) * 111320 *
                                                    np.cos(np.radians(lat))) ** 2
        out.append({**r, "lat_fixed": f"{lat:.8f}", "lon_fixed": f"{lon:.8f}",
                    "moved_m": round(float(np.sqrt(moved)), 2)})
        print(f"  {r['image']} pin {r['pin']}: {lat:.8f}, {lon:.8f}  "
              f"(moved {out[-1]['moved_m']} m)")
    if out:
        dst = csv_path.with_name(csv_path.stem + "_fixed.csv")
        with open(dst, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"wrote {dst}")


def main():
    ap = argparse.ArgumentParser(description="Click on a drone image to get GPS coordinates")
    ap.add_argument("path", type=Path, help="image file or folder of images "
                    "(or a clicked_points.csv with --recompute)")
    ap.add_argument("--recompute", action="store_true",
                    help="re-calculate the coordinates in an old clicked_points.csv")
    ap.add_argument("--log", type=Path, default=None,
                    help="CSV for pinned points (default: clicked_points.csv next to the images)")
    args = ap.parse_args()
    if not args.path.exists():
        sys.exit(f"not found: {args.path}")
    if args.recompute:
        recompute(args.path)
        return
    files = collect(args.path)
    if not files:
        sys.exit("no *_D.JPG or *_final.jpg images found")
    base = args.path if args.path.is_dir() else args.path.parent
    log = args.log or base / "clicked_points.csv"
    print(f"{len(files)} image(s) | pinned points are saved to {log}")
    Viewer(files, log).run()


if __name__ == "__main__":
    main()
