#!/usr/bin/env python3
"""
Plot where the drone says each photo was taken, to check the GPS is believable.

Reads the DJI metadata (XMP) of every .JPG in a folder (TIFs are skipped, so
each capture appears once) and plots the positions on a flat east/north plane
in metres: flight order, heading arrows and image numbers. Suspicious jumps
are marked.

Usage
  python plot_drone_positions.py "Laxman Mahale 13 Aug"
  python plot_drone_positions.py FOLDER --no-show        (just save the files)

Writes FOLDER/drone_positions.png and FOLDER/drone_positions.csv
Requires: matplotlib  (pip install matplotlib)
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

M = 111320.0


def read_xmp(path: Path) -> dict:
    raw = path.read_bytes()[:400_000]            # XMP sits near the start of DJI JPGs
    i = raw.find(b"<x:xmpmeta")
    j = raw.find(b"</x:xmpmeta>", i)
    if i < 0 or j < 0:
        return {}
    xmp = raw[i:j].decode("utf-8", "replace")
    d = dict(re.findall(r'drone-dji:([A-Za-z0-9]+)="([^"]*)"', xmp))
    d.update(re.findall(r"<drone-dji:([A-Za-z0-9]+)>([^<]*)</drone-dji:", xmp))
    return d


def num(d, *keys):
    for k in keys:
        try:
            return float(d[k])
        except (KeyError, ValueError):
            pass
    return None


def main():
    ap = argparse.ArgumentParser(description="Plot drone GPS positions from JPG metadata")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--no-show", action="store_true", help="save files without opening a window")
    args = ap.parse_args()

    jpgs = sorted(p for p in args.folder.iterdir()
                  if p.is_file() and p.suffix.lower() == ".jpg" and not p.name.startswith("._"))
    if not jpgs:
        raise SystemExit(f"no .JPG files in {args.folder}")

    pts, missing = [], []
    for p in jpgs:
        d = read_xmp(p)
        lat = num(d, "GpsLatitude", "Latitude")
        lon = num(d, "GpsLongitude", "GpsLongtitude", "Longitude")   # DJI typo variant
        if lat is None or lon is None or (lat == 0 and lon == 0):
            missing.append(p.name)
            continue
        pts.append({"image": p.name, "lat": lat, "lon": lon,
                    "rel_alt_m": num(d, "RelativeAltitude"),
                    "abs_alt_m": num(d, "AbsoluteAltitude"),
                    "gimbal_yaw": num(d, "GimbalYawDegree"),
                    "flight_yaw": num(d, "FlightYawDegree"),
                    "gimbal_pitch": num(d, "GimbalPitchDegree"),
                    "rtk_flag": d.get("RtkFlag", "")})
    print(f"{len(jpgs)} JPGs | {len(pts)} with GPS | {len(missing)} without")
    if missing:
        print("  no GPS:", ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    if not pts:
        return

    lat0 = sum(p["lat"] for p in pts) / len(pts)
    lon0 = sum(p["lon"] for p in pts) / len(pts)
    kx = M * math.cos(math.radians(lat0))
    for p in pts:
        p["east_m"] = (p["lon"] - lon0) * kx
        p["north_m"] = (p["lat"] - lat0) * M

    # step between consecutive photos; flag jumps far above the usual step
    steps = [0.0] + [math.hypot(b["east_m"] - a["east_m"], b["north_m"] - a["north_m"])
                     for a, b in zip(pts, pts[1:])]
    nz = sorted(s for s in steps[1:] if s > 0)
    med = nz[len(nz) // 2] if nz else 0
    for p, s in zip(pts, steps):
        p["step_m"] = round(s, 2)
        p["suspicious"] = bool(med and s > 4 * med and s > 5)

    out_csv = args.folder / "drone_positions.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pts[0]))
        w.writeheader()
        w.writerows(pts)

    rel = [p["rel_alt_m"] for p in pts if p["rel_alt_m"] is not None]
    rtk = sorted({p["rtk_flag"] for p in pts if p["rtk_flag"] != ""})
    sus = [p for p in pts if p["suspicious"]]
    print(f"  area covered: {max(p['east_m'] for p in pts) - min(p['east_m'] for p in pts):.1f} m "
          f"x {max(p['north_m'] for p in pts) - min(p['north_m'] for p in pts):.1f} m")
    print(f"  typical step between photos: {med:.2f} m")
    if rel:
        print(f"  height above take-off: {min(rel):.1f} - {max(rel):.1f} m")
    print(f"  RTK flag values: {', '.join(rtk) if rtk else 'not recorded'}  (50 = RTK fixed)")
    for p in sus:
        print(f"  JUMP: {p['image']} is {p['step_m']} m from the previous photo")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 9))
    E = [p["east_m"] for p in pts]
    N = [p["north_m"] for p in pts]
    ax.plot(E, N, "-", color="#bbbbbb", lw=1, zorder=1, label="flight order")
    sc = ax.scatter(E, N, c=range(1, len(pts) + 1), cmap="viridis", s=36, zorder=3,
                    edgecolors="k", linewidths=0.4)
    fig.colorbar(sc, ax=ax, label="photo number (capture order)")

    span = max(max(E) - min(E), max(N) - min(N), 1.0)
    L = span * 0.03
    for p in pts:                                   # heading arrows (camera top direction)
        yaw = p["flight_yaw"] if p["flight_yaw"] is not None else p["gimbal_yaw"]
        if yaw is not None:
            r = math.radians(yaw)
            ax.annotate("", xy=(p["east_m"] + L * math.sin(r), p["north_m"] + L * math.cos(r)),
                        xytext=(p["east_m"], p["north_m"]),
                        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=0.8), zorder=2)
    step_lbl = max(1, len(pts) // 40)
    for i, p in enumerate(pts):
        if i % step_lbl == 0 or p["suspicious"]:
            ax.annotate(str(i + 1), (p["east_m"], p["north_m"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
    if sus:
        ax.scatter([p["east_m"] for p in sus], [p["north_m"] for p in sus], s=160,
                   facecolors="none", edgecolors="red", linewidths=1.8, zorder=4,
                   label="suspicious jump")

    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(f"Drone photo positions - {args.folder.name}  ({len(pts)} photos, "
                 f"centre {lat0:.6f}, {lon0:.6f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_png = args.folder / "drone_positions.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nwrote {out_png}\n      {out_csv}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
