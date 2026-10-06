#!/usr/bin/env python3
"""
remove_overlap.py
-----------------
Same idea as detect_crop_segments_strip.py, for drone photos instead of video:

  Follow the ground from photo to photo along a flight line (image matching).
  Keep a photo, then skip photos until the ground has moved one full image
  height; keep the last photo before that point and cut off the few rows that
  still overlap the previous kept photo. The kept photos join up end to end:
  nothing repeated, nothing missing.

Flight lines are split where the heading turns. Tilted photos are skipped.

Input : folder with DJI_xxxx_D.JPG (+ the DJI_xxxx_MS_*.TIF bands)
Output: <folder>_no_overlap/   ready for plant_count.py as it is
          the kept photos and their MS_*.TIF bands, copied unchanged (all
          metadata kept). Consecutive kept photos only touch, apart from a small
          leftover overlap (overlap_px in segments.csv) that plant_count.py's
          normal duplicate removal takes care of.
          segments.csv   which photos were kept, travel and leftover overlap

Usage:
  python remove_overlap.py "Laxman Mahale-1"
  python remove_overlap.py FOLDER --out OUT_FOLDER

Requires: opencv-python, numpy
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

MATCH_WIDTH = 1600     # photos are matched at this width
TURN_DEG = 20.0        # heading change that starts a new flight line
MAX_TILT_DEG = 3.0     # skip photos whose camera is tilted more than this


def read_meta(path):
    raw = path.read_bytes()[:600_000]
    i = raw.find(b"<x:xmpmeta")
    j = raw.find(b"</x:xmpmeta>", i)
    d = {}
    if i >= 0 and j >= 0:
        x = raw[i:j].decode("utf-8", "replace")
        d = dict(re.findall(r'drone-dji:([A-Za-z0-9]+)="([^"]*)"', x))

    def f(*keys):
        for k in keys:
            try:
                return float(d[k])
            except (KeyError, ValueError):
                pass
        return None
    return {"lat": f("GpsLatitude"), "lon": f("GpsLongitude", "GpsLongtitude"),
            "yaw": f("FlightYawDegree", "GimbalYawDegree"), "pitch": f("GimbalPitchDegree"),
            "alt": f("RelativeAltitude"), "focal": f("CalibratedFocalLength")}


def angdiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


class Matcher:
    def __init__(self):
        self.sift = cv2.SIFT_create(6000)
        self.cache = {}

    def feats(self, path):
        if path not in self.cache:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            k = MATCH_WIDTH / img.shape[1]
            small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            kp, des = self.sift.detectAndCompute(small, None)
            self.cache[path] = (np.float32([p.pt for p in kp]) / k, des, img.shape[:2])
            if len(self.cache) > 3:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[path]

    def shift(self, pa, pb):
        """How far the ground moved from photo A to photo B, in px (dx, dy)."""
        A, B = self.feats(pa), self.feats(pb)
        if A[1] is None or B[1] is None:
            return None
        good = [m[0] for m in cv2.BFMatcher().knnMatch(A[1], B[1], k=2)
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 30:
            return None
        src = A[0][[m.queryIdx for m in good]]
        dst = B[0][[m.trainIdx for m in good]]
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0 * A[2][1] / MATCH_WIDTH)
        if H is None or mask.sum() < 30:
            return None
        h, w = A[2]
        c = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), H)[0, 0]
        return np.array([c[0] - w / 2, c[1] - h / 2], float)


def gps_shift(a, b):
    """Ground motion A -> B in B's image px from GPS (fallback)."""
    kx = 111320.0 * math.cos(math.radians(a["lat"]))
    de, dn = (a["lon"] - b["lon"]) * kx, (a["lat"] - b["lat"]) * 111320.0
    c, s = math.cos(math.radians(b["yaw"])), math.sin(math.radians(b["yaw"]))
    gsd = b["alt"] / (b["focal"] or 3700.0)
    return np.array([(de * c - dn * s) / gsd, (-de * s - dn * c) / gsd])


def main():
    ap = argparse.ArgumentParser(description="Keep non-overlapping drone photos")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paths = sorted(p for p in args.folder.iterdir() if p.is_file()
                   and p.suffix.lower() == ".jpg" and not p.name.startswith("._"))
    out = args.out or args.folder.with_name(args.folder.name + "_no_overlap")
    out.mkdir(parents=True, exist_ok=True)

    photos, rows = [], []
    for p in paths:
        m = read_meta(p)
        m["path"] = p
        if m["pitch"] is not None and abs(m["pitch"] + 90) > MAX_TILT_DEG:
            rows.append({"image": p.name, "kept": "no", "note": f"tilted (pitch {m['pitch']})"})
            continue
        photos.append(m)

    # flight lines
    lines, cur = [], []
    for m in photos:
        if cur and (m["yaw"] is None or cur[-1]["yaw"] is None
                    or abs(angdiff(m["yaw"], cur[-1]["yaw"])) > TURN_DEG):
            lines.append(cur)
            cur = []
        cur.append(m)
    if cur:
        lines.append(cur)
    print(f"{len(paths)} photos | {len(lines)} flight line(s) | "
          f"{len(paths) - len(photos)} tilted skipped")

    matcher = Matcher()
    kept_total = 0
    for li, line in enumerate(lines):
        h, w = cv2.imread(str(line[0]["path"]), cv2.IMREAD_REDUCED_GRAYSCALE_2).shape
        h, w = h * 2, w * 2
        # ground motion between consecutive photos
        steps = [np.zeros(2)]
        for a, b in zip(line, line[1:]):
            s = matcher.shift(a["path"], b["path"])
            if s is None and None not in (a["lat"], b["lat"], b["yaw"], b["alt"]):
                s = gps_shift(a, b)
            steps.append(s if s is not None else np.zeros(2))
        # motion axis: rows if the ground moves up/down, columns otherwise
        tot = np.sum(np.abs(np.array(steps)), axis=0)
        axis = 1 if tot[1] >= tot[0] else 0
        size = h if axis == 1 else w
        sign = 1.0 if sum(s[axis] for s in steps) >= 0 else -1.0
        travel = np.cumsum([sign * s[axis] for s in steps])   # px along the line

        # like the strip tracker: from each kept photo, move on while the ground
        # has travelled at most one image size; keep the last photo before that
        keep = [0]
        while keep[-1] < len(line) - 1:
            last = keep[-1]
            ok = [i for i in range(last + 1, len(line)) if travel[i] - travel[last] <= size]
            keep.append(ok[-1] if ok else last + 1)

        for n, i in enumerate(keep):
            m = line[i]
            hh, ww = h, w
            overlap = 0
            if n > 0:
                overlap = int(round(size - (travel[i] - travel[keep[n - 1]])))
                overlap = max(0, min(size, overlap))
            # the ground moves towards +sign, so the part already seen in the
            # previous kept photo is on that side of this photo
            x0, y0, x1, y1 = 0, 0, ww, hh
            if overlap:
                if axis == 1:
                    y0, y1 = (0, hh - overlap) if sign > 0 else (overlap, hh)
                else:
                    x0, x1 = (0, ww - overlap) if sign > 0 else (overlap, ww)
            if y1 - y0 < 10 or x1 - x0 < 10:
                continue
            # photo and its bands, copied byte for byte (metadata kept)
            stem = m["path"].name[:-len("_D.JPG")] if m["path"].name.upper().endswith("_D.JPG") \
                else m["path"].stem
            for f in m["path"].parent.glob(stem + "*"):
                if f.is_file() and not f.name.startswith("._"):
                    shutil.copy2(f, out / f.name)
            kept_total += 1
            rows.append({"image": m["path"].name, "kept": "yes", "line": li,
                         "travel_px": round(float(travel[i]), 1),
                         "overlap_px": overlap, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                         "note": "first of line" if n == 0 else ""})
            print(f"  line {li}: keep {m['path'].name}  (travel {travel[i]:.0f} px, "
                  f"overlap with previous kept photo {overlap} px)")
        for i, m in enumerate(line):
            if i not in keep:
                rows.append({"image": m["path"].name, "kept": "no", "line": li,
                             "travel_px": round(float(travel[i]), 1),
                             "note": "covered by kept photos"})

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "segments.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, restval="")
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: r["image"]))
    print(f"\nkept {kept_total} of {len(paths)} photos -> {out}")


if __name__ == "__main__":
    main()
