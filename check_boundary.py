#!/usr/bin/env python3
"""
Check whether GPS coordinates are inside the Laxman farm boundary, and show
them on a map.

Usage
  python check_boundary.py 20.35625 73.90595
  python check_boundary.py 20.35625 73.90595 20.35600 73.90560     (several points)
  python check_boundary.py 20.35625 73.90595 --show                (open map viewer)
  python check_boundary.py --show                                  (boundary only)

Prints INSIDE / OUTSIDE and the distance to the nearest boundary edge in metres.
--show writes boundary_view.html and opens it in your browser
(green = inside, red = outside; click a point for details).
No extra packages needed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from pathlib import Path

# Laxman farm boundary, in order around the perimeter [lat, lon]
BOUNDARY = [
    [20.3559929, 73.9057859],
    [20.35624862, 73.90609114],
    [20.35635419, 73.90622041],
    [20.35652892, 73.90594860],
    [20.35630062, 73.90580264],
    [20.35623064, 73.90563831],
]


def is_inside(lat: float, lon: float, poly=BOUNDARY) -> bool:
    """Ray casting point-in-polygon (same rule as plant_count.py and the map)."""
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def distance_to_boundary_m(lat: float, lon: float, poly=BOUNDARY) -> float:
    """Shortest distance (metres) from the point to any boundary edge."""
    kx = 111320.0 * math.cos(math.radians(lat))
    ky = 111320.0
    px, py = 0.0, 0.0
    pts = [((p[1] - lon) * kx, (p[0] - lat) * ky) for p in poly]
    best = float("inf")
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        best = min(best, math.hypot(ax + t * dx - px, ay + t * dy - py))
    return best


VIEWER = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Boundary check</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>html,body,#map{height:100%;margin:0}
.box{position:absolute;z-index:1000;top:10px;right:10px;background:#fff;padding:10px 12px;
border-radius:8px;font:13px system-ui,sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.3)}
.d{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}</style>
</head><body><div id="map"></div>
<div class="box"><b>Laxman farm boundary</b><br>
<span class="d" style="background:#1a9850"></span>inside<br>
<span class="d" style="background:#e31a1c"></span>outside<br>
<small>Click the map to check any spot</small></div>
<script>
const B=__BOUNDARY__, P=__POINTS__;
const map=L.map('map',{maxZoom:23});
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
 {maxNativeZoom:19,maxZoom:23,attribution:'Imagery &copy; Esri'}).addTo(map);
const poly=L.polygon(B,{color:'#fff',weight:2,dashArray:'6 4',fillOpacity:.08}).addTo(map);
B.forEach((p,i)=>L.circleMarker(p,{radius:4,color:'#fff',weight:1,fillColor:'#ffd400',fillOpacity:1})
 .bindTooltip('corner '+(i+1)).addTo(map));
function inside(lat,lon){let c=false;for(let i=0,j=B.length-1;i<B.length;j=i++){
 const[yi,xi]=B[i],[yj,xj]=B[j];
 if((yi>lat)!==(yj>lat)&&lon<(xj-xi)*(lat-yi)/(yj-yi)+xi)c=!c}return c}
function add(lat,lon,label){const ins=inside(lat,lon);
 return L.circleMarker([lat,lon],{radius:7,color:'#fff',weight:2,fillColor:ins?'#1a9850':'#e31a1c',fillOpacity:1})
 .bindPopup(`<b>${label}</b><br>${lat.toFixed(8)}, ${lon.toFixed(8)}<br><b>${ins?'INSIDE':'OUTSIDE'}</b>`).addTo(map)}
const g=L.featureGroup([poly,...P.map((p,i)=>add(p[0],p[1],'point '+(i+1)))]);
map.fitBounds(g.getBounds(),{padding:[40,40]});
let clicked=null;
map.on('click',e=>{if(clicked)map.removeLayer(clicked);
 clicked=add(e.latlng.lat,e.latlng.lng,'clicked spot').openPopup()});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Is a GPS point inside the Laxman farm?")
    ap.add_argument("coords", nargs="*", type=float, help="lat lon [lat lon ...]")
    ap.add_argument("--show", action="store_true", help="open the map viewer")
    args = ap.parse_args()
    if len(args.coords) % 2:
        sys.exit("give coordinates in pairs: lat lon")
    pts = [(args.coords[i], args.coords[i + 1]) for i in range(0, len(args.coords), 2)]
    if not pts and not args.show:
        ap.print_help()
        return

    for n, (lat, lon) in enumerate(pts, 1):
        ins = is_inside(lat, lon)
        d = distance_to_boundary_m(lat, lon)
        side = "inside" if ins else "outside"
        print(f"point {n}: {lat:.8f}, {lon:.8f}  ->  {'INSIDE ' if ins else 'OUTSIDE'}"
              f"  ({d:.2f} m {side} the nearest edge)")

    if args.show:
        out = Path(__file__).resolve().parent / "boundary_view.html"
        out.write_text(VIEWER.replace("__BOUNDARY__", json.dumps(BOUNDARY))
                       .replace("__POINTS__", json.dumps(pts)), encoding="utf-8")
        print(f"map: {out}")
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
