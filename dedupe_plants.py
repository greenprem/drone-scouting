#!/usr/bin/env python3
"""
Remove duplicate plant detections from plant_instances_batch.csv.

Two kinds of duplicates are removed:
  1. within-frame  - two circles on the same plant in one image
  2. cross-frame   - the same plant seen in several overlapping images

Drone GPS (without RTK) can be off by 1-5 m, which is far more than the gap
between two plants. So before matching, every image is shifted to line up with
the plants already placed from earlier images ("frame alignment"). Consecutive
images are taken seconds apart, so their GPS errors are almost the same and the
needed shift is small; the search is limited to --search-radius.

Outputs (next to the input CSV unless --out-dir is given):
  plants_unique.csv          one row per unique plant  (row count = plant count)
  plants_detection_map.csv   every input detection -> the unique plant it belongs to
  frame_alignment.csv        shift applied to each image + new / repeat counts
  final_overlays/<image>_final.jpg
                             evidence images: yellow = plant counted in this image,
                             magenta = repeat (already counted in an earlier image);
                             merged double circles are not drawn

Usage:
  python dedupe_plants.py "Laxman Mahale-1/output/plant_instances_batch.csv"
  python dedupe_plants.py input.csv --spacing 0.25 --no-align   # RTK drone
Requires: numpy (+ opencv-python for the images)
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

M_PER_DEG = 111320.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def to_float(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def point_in_polygon(lat, lon, poly) -> bool:
    """poly = [[lat, lon], ...] (any order around the perimeter)."""
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class Grid:
    """Tiny spatial hash for nearest-neighbour lookups (no scipy needed)."""

    def __init__(self, cell: float):
        self.cell = cell
        self.cells = defaultdict(list)

    def key(self, x, y):
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))

    def add(self, idx, x, y):
        self.cells[self.key(x, y)].append(idx)

    def move(self, idx, old_xy, new_xy):
        ko, kn = self.key(*old_xy), self.key(*new_xy)
        if ko != kn:
            self.cells[ko].remove(idx)
            self.cells[kn].append(idx)

    def near(self, x, y, r):
        cx, cy = self.key(x, y)
        n = int(math.ceil(r / self.cell))
        for i in range(cx - n, cx + n + 1):
            for j in range(cy - n, cy + n + 1):
                yield from self.cells.get((i, j), ())


def nn_distances(xy: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distance for each point (brute force in chunks)."""
    if len(xy) < 2:
        return np.array([])
    out = np.empty(len(xy))
    for s in range(0, len(xy), 500):
        d = np.hypot(*(xy[s:s + 500, None, :] - xy[None, :, :]).transpose(2, 0, 1))
        d[np.arange(d.shape[0]), np.arange(s, s + d.shape[0])] = np.inf
        out[s:s + 500] = d.min(axis=1)
    return out


# ---------------------------------------------------------------------------
# step 1: within-frame merge
# ---------------------------------------------------------------------------

def _centrality(r) -> float:
    """0 at the image centre, ~1 at the corners (from the RGB pixel position)."""
    w = to_float(r.get("rgb_width_px"), float("nan"))
    x, y = to_float(r.get("rgb_x_px")), to_float(r.get("rgb_y_px"))
    if not (np.isfinite(w) and np.isfinite(x) and np.isfinite(y)):
        return 1.0
    h = w * 0.75
    return math.hypot((x - w / 2) / (w / 2), (y - h / 2) / (h / 2)) / math.sqrt(2)


def merge_within_frame(xy: np.ndarray, radius: float):
    """Union points closer than `radius`. Returns cluster label per point."""
    n = len(xy)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    g = Grid(radius)
    for i, (x, y) in enumerate(xy):
        g.add(i, x, y)
    for i, (x, y) in enumerate(xy):
        for j in g.near(x, y, radius):
            if j > i and math.hypot(xy[j, 0] - x, xy[j, 1] - y) < radius:
                parent[find(j)] = find(i)
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


# ---------------------------------------------------------------------------
# step 2: frame alignment (offset voting)
# ---------------------------------------------------------------------------

def count_matches(q: np.ndarray, p: np.ndarray, off, tol: float) -> int:
    if len(p) == 0:
        return 0
    qs = q + off
    best = np.full(len(qs), np.inf)
    for s in range(0, len(p), 2000):
        d = np.hypot(*(qs[:, None, :] - p[None, s:s + 2000, :]).transpose(2, 0, 1))
        best = np.minimum(best, d.min(axis=1))
    return int((best < tol).sum())


def estimate_offset(q: np.ndarray, p: np.ndarray, search: float, tol: float,
                    spacing: float, min_overlap: int):
    """
    Find the shift that makes frame points q line up with placed points p.
    Returns (offset_xy, matches_at_offset, confidence, note).
    """
    zero = np.zeros(2)
    if len(p) < min_overlap or len(q) < min_overlap:
        return zero, 0, 0.0, "too few overlapping plants - GPS used as-is"

    # every (placed - new) difference within the search radius is a vote
    votes = []
    for s in range(0, len(q), 300):
        d = p[None, :, :] - q[s:s + 300, None, :]
        d = d.reshape(-1, 2)
        votes.append(d[(np.abs(d[:, 0]) < search) & (np.abs(d[:, 1]) < search)])
    votes = np.concatenate(votes) if votes else np.empty((0, 2))
    if len(votes) == 0:
        return zero, 0, 0.0, "no plants within search radius - GPS used as-is"

    b = tol / 2.0
    nb = int(math.ceil(2 * search / b))
    hist, xe, ye = np.histogram2d(votes[:, 0], votes[:, 1], bins=nb,
                                  range=[[-search, search], [-search, search]])
    # 3x3 smoothing so a peak split across bins still wins
    pad = np.pad(hist, 1)
    sm = sum(pad[i:i + nb, j:j + nb] for i in range(3) for j in range(3))

    # take the top peaks, score each one properly, prefer small shifts on ties
    cands = []
    flat = np.argsort(sm, axis=None)[::-1]
    for f in flat[:200]:
        i, j = np.unravel_index(f, sm.shape)
        c = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
        if any(np.hypot(*(c - o)) < spacing * 0.5 for o in cands):
            continue
        cands.append(c)
        if len(cands) == 8:
            break

    scored = []
    for c in cands:
        near = votes[np.hypot(*(votes - c).T) < tol]
        refined = near.mean(axis=0) if len(near) else c
        scored.append((count_matches(q, p, refined, tol), refined))
    scored.sort(key=lambda t: -t[0])
    best_n = scored[0][0]
    # among near-equal scores (periodic rows) pick the smallest shift
    ties = [t for t in scored if t[0] >= 0.95 * best_n]
    n, off = min(ties, key=lambda t: np.hypot(*t[1]))
    others = [t[0] for t in scored if np.hypot(*(t[1] - off)) > spacing * 0.5]
    second = max(others) if others else 0
    conf = n / max(second, 1)
    if n < min_overlap:
        return zero, n, conf, "weak match - GPS used as-is"
    note = ("ok" if conf >= 1.15 else
            "repeating pattern: count OK, position may be off by one plant spacing")
    return off, n, conf, note



# ---------------------------------------------------------------------------
# image-based frame alignment
# ---------------------------------------------------------------------------
# Plants repeat every ~30 cm and rows every ~90 cm, so lining frames up by their
# plants is ambiguous (a shift of one plant or one row fits almost as well).
# Instead, consecutive photos are matched by their picture content (soil
# texture, plastic wrinkles, pipes), which does not repeat. The pixel match is
# turned into a correction in metres for each frame.

MATCH_WIDTH = 1600          # images are matched at this width


def _fit_px_to_m(samples, lat0, lon0, kx):
    """Homography: native RGB pixel -> local metres (east, north) for one frame."""
    import cv2
    src = np.array([[s[0], s[1]] for s in samples], np.float64)
    dst = np.array([[(s[3] - lon0) * kx, (s[2] - lat0) * M_PER_DEG] for s in samples], np.float64)
    if len(src) < 4:
        return None
    Hm, _ = cv2.findHomography(src, dst, 0)
    return Hm


def _apply_h(Hm, pts):
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ Hm.T
    return p[:, :2] / p[:, 2:3]


class _Features:
    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.det = cv2.SIFT_create(nfeatures=6000) if hasattr(cv2, "SIFT_create") \
            else cv2.ORB_create(nfeatures=8000)
        self.norm = cv2.NORM_L2 if hasattr(cv2, "SIFT_create") else cv2.NORM_HAMMING
        self.cache = {}

    def get(self, path):
        if path in self.cache:
            return self.cache[path]
        cv2 = self.cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.cache[path] = None
            return None
        s = MATCH_WIDTH / img.shape[1]
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        small = cv2.createCLAHE(2.0, (8, 8)).apply(small)
        kp, des = self.det.detectAndCompute(small, None)
        self.cache[path] = (np.float32([k.pt for k in kp]) / s, des, img.shape)
        if len(self.cache) > 6:                                  # keep memory small
            self.cache.pop(next(iter(self.cache)))
        return self.cache[path]

    def match(self, pa, pb):
        """Homography native px of A -> native px of B, and inlier count."""
        A, B = self.get(pa), self.get(pb)
        if not A or not B or A[1] is None or B[1] is None or len(A[0]) < 20 or len(B[0]) < 20:
            return None, 0
        bf = self.cv2.BFMatcher(self.norm)
        good = []
        for m in bf.knnMatch(A[1], B[1], k=2):
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                good.append(m[0])
        if len(good) < 30:
            return None, len(good)
        src = A[0][[m.queryIdx for m in good]]
        dst = B[0][[m.trainIdx for m in good]]
        Hab, mask = self.cv2.findHomography(src, dst, self.cv2.RANSAC, 6.0)
        n = int(mask.sum()) if mask is not None else 0
        return (Hab if n >= 30 else None), n


def image_alignment(order, px2m, paths, centres, min_inliers=40):
    """
    Correction (2x3 similarity, metres -> metres) for every frame so that
    overlapping frames agree. Each frame is matched to the earlier frame that
    overlaps it most (by GPS distance); the first frame is the reference.
    """
    import cv2
    feats = _Features()
    corr, info = {}, {}
    done = []
    last_ok = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    for f in order:
        if not done:
            corr[f] = np.array([[1.0, 0, 0], [0, 1.0, 0]])
            info[f] = ("first image (reference)", "", 0)
            done.append(f)
            continue
        cands = sorted(done, key=lambda g: np.hypot(*(centres[g] - centres[f])))[:4]
        best = None
        if f in paths and f in px2m:
            for g in cands:
                if g not in paths or g not in px2m:
                    continue
                Hfg, n = feats.match(paths[f], paths[g])
                if Hfg is not None and n >= min_inliers and (best is None or n > best[2]):
                    best = (g, Hfg, n)
                if best and best[2] >= 200:
                    break
        if best is None:
            corr[f] = last_ok.copy()             # GPS bias drifts slowly: reuse last
            info[f] = ("no image match - used previous correction", "", 0)
            done.append(f)
            continue
        g, Hfg, n = best
        h, w = feats.get(paths[f])[2]
        gx, gy = np.meshgrid(np.linspace(0.05 * w, 0.95 * w, 9), np.linspace(0.05 * h, 0.95 * h, 7))
        pf = np.column_stack([gx.ravel(), gy.ravel()])
        pg = _apply_h(Hfg, pf)
        hg, wg = feats.get(paths[g])[2]
        ok = (pg[:, 0] >= 0) & (pg[:, 0] < wg) & (pg[:, 1] >= 0) & (pg[:, 1] < hg)
        if ok.sum() < 6:
            corr[f] = last_ok.copy()
            info[f] = ("too little overlap - used previous correction", g, n)
            done.append(f)
            continue
        m_f = _apply_h(px2m[f], pf[ok])                                  # where f puts them
        m_g = _apply_h(px2m[g], pg[ok])
        m_g = m_g @ corr[g][:, :2].T + corr[g][:, 2]                     # corrected truth
        S, _ = cv2.estimateAffinePartial2D(m_f, m_g, method=cv2.LMEDS)
        if S is not None:
            scale = math.hypot(S[0, 0], S[1, 0])
            rot = math.degrees(math.atan2(S[1, 0], S[0, 0]))
            if not (0.95 < scale < 1.05 and abs(rot) < 5):
                S = None
        if S is None:                                                    # translation only
            t = np.median(m_g - m_f, axis=0)
            S = np.array([[1.0, 0, t[0]], [0, 1.0, t[1]]])
        corr[f] = S
        last_ok = S
        info[f] = ("ok", g, n)
        done.append(f)
    return corr, info

# ---------------------------------------------------------------------------
# evidence images
# ---------------------------------------------------------------------------

NEW_COLOR = (0, 255, 255)      # yellow  (BGR)
REPEAT_COLOR = (255, 0, 255)   # magenta (BGR)
BOUNDARY_COLOR = (255, 255, 255)  # white outline
CORNER_COLOR = (0, 165, 255)      # orange boundary points
OUTSIDE_COLOR = (60, 60, 255)  # red     (BGR) - outside the farm boundary


def draw_final_overlays(base_dir: Path, out_dir: Path, order, frames, align_rows,
                        hide_repeats: bool = False, show_outside: bool = False,
                        boundary_marks=None):
    try:
        import cv2
    except ImportError:
        print("opencv not installed - skipping final_overlays")
        return
    first = frames[order[0]][0]
    if "rgb_x_px" not in first or not base_dir.is_dir():
        print("\nfinal_overlays skipped: needs rgb_x_px in the CSV and the _base images "
              "(run plant_count.py, which does everything in one go)")
        return
    img_dir = out_dir / "final_overlays"
    img_dir.mkdir(exist_ok=True)
    stats = {a["image_id"]: a for a in align_rows}
    written = 0
    for f in order:
        base_path = base_dir / f"{f}.jpg"
        vis = cv2.imread(str(base_path))
        if vis is None:
            print(f"  missing base image {base_path.name} - skipped")
            continue
        det = frames[f]
        scale = vis.shape[1] / max(to_float(det[0].get("rgb_width_px"), vis.shape[1]), 1)
        radius = max(5, int(round(14 * scale)))
        th = max(1, int(round(4 * scale)))

        # one circle per unique plant (merged double circles collapse to their mean)
        groups = defaultdict(list)
        for r in det:
            groups[r["_uid"]].append(r)
        for uid, rs in groups.items():
            x = np.mean([to_float(r["rgb_x_px"]) for r in rs]) * scale
            y = np.mean([to_float(r["rgb_y_px"]) for r in rs]) * scale
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if rs[0].get("_outside"):
                color = OUTSIDE_COLOR
            elif hide_repeats and rs[0]["_cross_dup"]:
                continue
            else:
                color = REPEAT_COLOR if rs[0]["_cross_dup"] else NEW_COLOR
            cv2.circle(vis, (int(round(x)), int(round(y))), radius, color, th, cv2.LINE_AA)

        # farm boundary outline + named corner points (native px -> image px)
        bm = (boundary_marks or {}).get(f) or []
        if len(bm) >= 3:
            hh, ww = vis.shape[:2]
            poly_px = np.array([[x * scale, y * scale] for _, x, y in bm])
            if np.all(np.isfinite(poly_px)) and np.abs(poly_px).max() < 1e6:
                cv2.polylines(vis, [np.round(poly_px * 16).astype(np.int64).reshape(-1, 1, 2)],
                              True, BOUNDARY_COLOR, max(2, th), cv2.LINE_AA, shift=4)
                for (name, _, _), (x, y) in zip(bm, poly_px):
                    if 0 <= x < ww and 0 <= y < hh:
                        c = (int(round(x)), int(round(y)))
                        r = max(10, 2 * radius)
                        cv2.circle(vis, c, r, CORNER_COLOR, max(2, th), cv2.LINE_AA)
                        cv2.drawMarker(vis, c, CORNER_COLOR, cv2.MARKER_CROSS, 2 * r, max(2, th))
                        fs_c = max(0.6, vis.shape[1] / 2600 * 1.1)
                        for col, t in (((0, 0, 0), th + 3), (CORNER_COLOR, th)):
                            cv2.putText(vis, name, (c[0] + r + 6, c[1] - r // 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, fs_c, col, t, cv2.LINE_AA)

        a = stats[f]
        lines = [(f"{f}", (255, 255, 255)),
                 (f"Counted in this image: {a['new_plants']}", NEW_COLOR),
                 (f"Repeats (counted earlier{', not drawn' if hide_repeats else ''}): "
                  f"{a['repeat_plants']}", REPEAT_COLOR),
                 (f"Outside farm boundary (not counted): {a['outside_farm']}", OUTSIDE_COLOR),
                 (f"Running farm total: {a['running_total']}", (255, 255, 255))]
        if not show_outside:
            lines.pop(3)
            lines[-1] = (f"Running field total: {a['running_total']}", (255, 255, 255))
        fs = max(0.6, vis.shape[1] / 2600 * 1.1)
        lh = int(38 * fs)
        pad = int(12 * fs)
        tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0] for t, _ in lines)
        box = vis[0:lh * len(lines) + 2 * pad, 0:tw + 2 * pad]
        box[:] = (box * 0.35).astype(np.uint8)
        for k, (t, c) in enumerate(lines):
            cv2.putText(vis, t, (pad, pad + lh * (k + 1) - int(10 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, c, 2, cv2.LINE_AA)
        cv2.imwrite(str(img_dir / f"{f}_final.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written += 1
    print(f"\nwrote {written} evidence images to {img_dir}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Remove duplicate plant detections")
    ap.add_argument("csv", type=Path, help="plant_instances_batch.csv")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--spacing", type=float, default=None,
                    help="plant-to-plant spacing in metres (default: auto)")
    ap.add_argument("--merge-radius", type=float, default=None,
                    help="within-frame merge distance, m (default 0.4 x spacing)")
    ap.add_argument("--match-radius", type=float, default=None,
                    help="cross-frame match distance, m (default 0.4 x spacing)")
    ap.add_argument("--search-radius", type=float, default=1.0,
                    help="max GPS shift between consecutive images, m (default 1.0)")
    ap.add_argument("--min-overlap", type=int, default=15,
                    help="min shared plants needed to trust an alignment")
    ap.add_argument("--no-align", action="store_true",
                    help="trust GPS as-is (use with RTK drones)")
    ap.add_argument("--no-images", action="store_true",
                    help="skip drawing final_overlays")
    ap.add_argument("--boundary", type=Path, default=None,
                    help="JSON file with the farm boundary [[lat, lon], ...]")
    ap.add_argument("--hide-repeats", action="store_true",
                    help="draw only plants counted in each image (no magenta repeats)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    raw_dir = args.csv.parent.parent            # <folder>/output/x.csv -> <folder>
    paths = {}
    for r in rows:
        st = r.get("image_id", "")
        if st and st not in paths:
            for ext in ("_D.JPG", "_D.jpg"):
                if (raw_dir / f"{st}{ext}").exists():
                    paths[st] = raw_dir / f"{st}{ext}"
    if not paths:
        print("raw images not found next to the output folder - plant-based alignment only")
    res = run_dedupe(rows, args.out_dir or args.csv.parent,
                     base_dir=args.csv.parent / "_base",
                     spacing=args.spacing, merge_radius=args.merge_radius,
                     match_radius=args.match_radius, search_radius=args.search_radius,
                     min_overlap=args.min_overlap, no_align=args.no_align,
                     draw_images=not args.no_images, hide_repeats=args.hide_repeats,
                     farm_boundary=(__import__("json").load(open(args.boundary))
                                    if args.boundary else None),
                     image_paths=paths or None)
    if res is None:
        raise SystemExit("no rows with lat/lon found")


def run_dedupe(rows, out_dir, base_dir=None, spacing=None, merge_radius=None,
               match_radius=None, search_radius=1.0, min_overlap=15, no_align=False,
               draw_images=True, hide_repeats=False, farm_boundary=None,
               boundary_marks=None, px2ll=None, image_paths=None):
    """
    De-duplicate plant detections (list of dicts, one per detection) and write
    plants_unique.csv, plants_detection_map.csv, frame_alignment.csv and
    final_overlays/ into out_dir. Returns a dict with per-image stats, or None.
    """
    from types import SimpleNamespace
    args = SimpleNamespace(spacing=spacing, merge_radius=merge_radius,
                           match_radius=match_radius, search_radius=search_radius,
                           min_overlap=min_overlap, no_align=no_align)
    out_dir = Path(out_dir)
    rows = [r for r in rows if not math.isnan(to_float(r.get("lat")))
            and not math.isnan(to_float(r.get("lon")))]
    if not rows:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    # lat/lon -> local metres (east, north)
    lat0 = float(np.mean([to_float(r["lat"]) for r in rows]))
    lon0 = float(np.mean([to_float(r["lon"]) for r in rows]))
    kx = M_PER_DEG * math.cos(math.radians(lat0))
    for r in rows:
        r["_x"] = (to_float(r["lon"]) - lon0) * kx
        r["_y"] = (to_float(r["lat"]) - lat0) * M_PER_DEG

    frames = defaultdict(list)
    for r in rows:
        frames[r["image_id"]].append(r)
    order = sorted(frames)              # DJI names sort in capture order

    # plant spacing (median nearest-neighbour distance inside frames)
    if args.spacing:
        spacing = args.spacing
    else:
        d = np.concatenate([nn_distances(np.array([[r["_x"], r["_y"]] for r in frames[f]]))
                            for f in order])
        d = d[np.isfinite(d) & (d > 0.01)]
        spacing = float(np.median(d)) if len(d) else 0.25
    merge_r = args.merge_radius or 0.4 * spacing
    match_r = args.match_radius or 0.4 * spacing
    print(f"{len(rows)} detections in {len(order)} images")
    print(f"plant spacing ~{spacing * 100:.1f} cm | merge {merge_r * 100:.1f} cm | "
          f"match {match_r * 100:.1f} cm | search +/-{args.search_radius:.2f} m"
          f"{' | alignment OFF' if args.no_align else ''}")

    # ---- frame corrections from image matching (falls back to plant voting)
    corr, img_info = {}, {}
    if not args.no_align and image_paths:
        px2m = {}
        for f in order:
            samples = (px2ll or {}).get(f)
            if not samples:                     # standalone: fit from detections
                samples = [(to_float(r.get("rgb_x_px")), to_float(r.get("rgb_y_px")),
                            to_float(r["lat"]), to_float(r["lon"])) for r in frames[f]]
                samples = [q for q in samples if all(np.isfinite(q))]
                if len(samples) < 8:
                    continue
            Hm = _fit_px_to_m(samples, lat0, lon0, kx)
            if Hm is not None:
                px2m[f] = Hm
        centres = {f: np.mean([[r["_x"], r["_y"]] for r in frames[f]], axis=0) for f in order}
        print("aligning frames by image content ...")
        try:
            corr, img_info = image_alignment(order, px2m,
                                             {k: Path(v) for k, v in image_paths.items()},
                                             centres)
        except Exception as exc:
            print(f"  image alignment failed ({exc}) - using plant-based alignment")
            corr = {}
        n_ok = sum(1 for v in img_info.values() if v[0] == "ok")
        if corr:
            print(f"  {n_ok}/{len(order)} frames matched by image content")
        if corr and n_ok == 0 and len(order) > 1:
            print("  no frame could be matched - falling back to plant-based alignment")
            corr = {}

    plants = []          # unique plants
    grid = Grid(max(match_r, 0.05))
    align_rows = []
    n_within = n_cross = 0

    for f in order:
        det = frames[f]
        xy = np.array([[r["_x"], r["_y"]] for r in det])

        # 1) within-frame merge
        lab = merge_within_frame(xy, merge_r)
        groups = defaultdict(list)
        for i, l in enumerate(lab):
            groups[l].append(i)
        g_xy = np.array([xy[idx].mean(axis=0) for idx in groups.values()])
        g_idx = list(groups.values())
        for idx in g_idx:
            for k, i in enumerate(idx):
                det[i]["_within_dup"] = k > 0
        n_within += len(xy) - len(g_idx)

        # 2) align frame: image-based correction if available, else plant voting
        S = None
        if f in corr:
            S = corr[f]
            g_xy = g_xy @ S[:, :2].T + S[:, 2]
            c0 = centres[f]
            off = (c0 @ S[:, :2].T + S[:, 2]) - c0
            note, g_ref, nin = img_info[f]
            rot = math.degrees(math.atan2(S[1, 0], S[0, 0]))
            align_rows.append({"image_id": f, "detections": len(xy),
                               "after_within_merge": len(g_xy),
                               "shift_east_m": round(float(off[0]), 3),
                               "shift_north_m": round(float(off[1]), 3),
                               "rotation_deg": round(rot, 2),
                               "matched_to": g_ref, "image_inliers": nin,
                               "matched_plants": 0, "confidence": "", "note": note})
        elif plants and not args.no_align:
            lo = g_xy.min(axis=0) - args.search_radius - match_r
            hi = g_xy.max(axis=0) + args.search_radius + match_r
            p_all = np.array([[p["x"], p["y"]] for p in plants])
            inside = np.all((p_all >= lo) & (p_all <= hi), axis=1)
            off, nm, conf, note = estimate_offset(
                g_xy, p_all[inside], args.search_radius, match_r, spacing,
                args.min_overlap)
        else:
            off, nm, conf, note = np.zeros(2), 0, 0.0, (
                "first image (reference)" if not plants else "alignment off")
        if S is None:
            g_xy = g_xy + off
            align_rows.append({"image_id": f, "detections": len(xy),
                               "after_within_merge": len(g_xy),
                               "shift_east_m": round(float(off[0]), 3),
                               "shift_north_m": round(float(off[1]), 3),
                               "rotation_deg": 0.0, "matched_to": "", "image_inliers": "",
                               "matched_plants": nm, "confidence": round(conf, 2),
                               "note": note})
            S = np.array([[1.0, 0, off[0]], [0, 1.0, off[1]]])

        # 3) match to existing plants (one-to-one, closest first) or add new
        pairs = []
        for gi, (x, y) in enumerate(g_xy):
            for pi in grid.near(x, y, match_r):
                d = math.hypot(plants[pi]["x"] - x, plants[pi]["y"] - y)
                if d < match_r:
                    pairs.append((d, gi, pi))
        pairs.sort()
        g_to_p, used = {}, set()
        for d, gi, pi in pairs:
            if gi not in g_to_p and pi not in used:
                g_to_p[gi] = pi
                used.add(pi)

        for gi, idx in enumerate(g_idx):
            x, y = g_xy[gi]
            cen = _centrality(det[idx[0]])
            n_matched = 0
            if gi in g_to_p:
                pi = g_to_p[gi]
                p = plants[pi]
                # no averaging: keep the position seen closest to an image centre,
                # where the position is most accurate
                if cen < p["cen"]:
                    old = (p["x"], p["y"])
                    p["x"], p["y"], p["cen"] = x, y, cen
                    grid.move(pi, old, (x, y))
                p["images"].append(f)
                n_cross += 1
                cross = True
            else:
                pi = len(plants)
                plants.append({"x": x, "y": y, "cen": cen,
                               "images": [f], "det": [], "pts": []})
                grid.add(pi, x, y)
                cross = False
            for i in idx:
                r = det[i]
                r["_uid"] = pi
                r["_cross_dup"] = cross
                cx_, cy_ = np.array([r["_x"], r["_y"]]) @ S[:, :2].T + S[:, 2]
                r["_cx"], r["_cy"] = float(cx_), float(cy_)
                plants[pi]["det"].append(r)
                plants[pi]["pts"].append((r["_cx"], r["_cy"]))
        if align_rows:
            align_rows[-1]["matched_plants"] = len(g_to_p)
            # full correction (raw local metres -> corrected), so other tools can
            # undo it: x' = a*x + b*y + c ; y' = d*x + e*y + f
            align_rows[-1].update({"corr_a": S[0, 0], "corr_b": S[0, 1], "corr_c": S[0, 2],
                                   "corr_d": S[1, 0], "corr_e": S[1, 1], "corr_f": S[1, 2],
                                   "origin_lat": lat0, "origin_lon": lon0})

    # farm boundary: plants outside it are kept in the files but not counted
    poly = ([[float(q[0]), float(q[1])] for q in farm_boundary]
            if farm_boundary and len(farm_boundary) >= 3 else None)
    for p in plants:
        lat = lat0 + p["y"] / M_PER_DEG
        lon = lon0 + p["x"] / kx
        p["inside"] = point_in_polygon(lat, lon, poly) if poly else True
    for f in order:
        for r in frames[f]:
            r["_outside"] = not plants[r["_uid"]]["inside"]
    n_outside = sum(1 for p in plants if not p["inside"])

    total = 0
    for a in align_rows:
        uids = {}
        for r in frames[a["image_id"]]:
            uids[r["_uid"]] = (r["_cross_dup"], r["_outside"])
        new_n = sum(1 for c, o in uids.values() if not c and not o)
        total += new_n
        a["new_plants"] = new_n
        a["repeat_plants"] = sum(1 for c, o in uids.values() if c and not o)
        a["outside_farm"] = sum(1 for c, o in uids.values() if o)
        a["running_total"] = total

    # -------------------------------------------------------------- outputs
    def xy_to_ll(x, y):
        return lat0 + y / M_PER_DEG, lon0 + x / kx

    uniq_path = out_dir / "plants_unique.csv"
    with open(uniq_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["plant_uid", "lat", "lon", "n_images", "n_detections", "images",
                    "first_image", "position_spread_cm", "ndvi_mean", "ndre_mean",
                    "area_px_mean", "detection_methods", "inside_farm"])
        for uid, p in enumerate(plants):
            lat, lon = xy_to_ll(p["x"], p["y"])
            spread = max(math.hypot(a - p["x"], b - p["y"]) for a, b in p["pts"])
            nd = [to_float(r.get("ndvi_mean")) for r in p["det"]]
            nr = [to_float(r.get("ndre_mean")) for r in p["det"]]
            ar = [to_float(r.get("area_px")) for r in p["det"]]
            mean = lambda v: round(float(np.nanmean(v)), 4) if np.isfinite(v).any() else ""
            w.writerow([uid, f"{lat:.8f}", f"{lon:.8f}", len(p["images"]), len(p["det"]),
                        ";".join(p["images"]), p["images"][0], round(spread * 100, 1),
                        mean(np.array(nd)), mean(np.array(nr)), mean(np.array(ar)),
                        ";".join(sorted({r.get("detection_method", "") for r in p["det"]})),
                        p["inside"] if poly else ""])

    map_path = out_dir / "plants_detection_map.csv"
    base = [k for k in rows[0] if not k.startswith("_")]
    with open(map_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(base + ["plant_uid", "duplicate_type", "aligned_lat", "aligned_lon"])
        for f in order:
            for r in frames[f]:
                dtype = ("within_frame" if r["_within_dup"]
                         else "cross_frame" if r["_cross_dup"] else "first_seen")
                lat, lon = xy_to_ll(r["_cx"], r["_cy"])
                w.writerow([r[k] for k in base] +
                           [r["_uid"], dtype, f"{lat:.8f}", f"{lon:.8f}"])

    al_path = out_dir / "frame_alignment.csv"
    keys = list(dict.fromkeys(k for a in align_rows for k in a))
    with open(al_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(align_rows)

    if draw_images and base_dir is not None:
        draw_final_overlays(Path(base_dir), out_dir, order, frames, align_rows,
                            hide_repeats, show_outside=poly is not None,
                            boundary_marks=boundary_marks)

    amb = sum(1 for a in align_rows if a["note"].startswith(("weak", "too few", "no plants")))
    print(f"\nwithin-frame duplicates removed : {n_within}")
    print(f"cross-frame duplicates removed  : {n_cross}")
    if poly:
        print(f"outside farm boundary (not counted): {n_outside}")
        print(f"UNIQUE PLANTS IN FARM           : {len(plants) - n_outside}")
    else:
        print(f"UNIQUE PLANTS                   : {len(plants)}")
    if amb:
        print(f"WARNING: {amb} image(s) could not be aligned (GPS used as-is), "
              "their overlap may be double counted - see frame_alignment.csv")
    print(f"\nwrote {uniq_path}\n      {map_path}\n      {al_path}")
    return {"unique": len(plants) - n_outside, "unique_all": len(plants),
            "outside_farm": n_outside, "uniq_path": uniq_path,
            "within_removed": n_within,
            "cross_removed": n_cross,
            "per_image": {a["image_id"]: a for a in align_rows}}


if __name__ == "__main__":
    main()
