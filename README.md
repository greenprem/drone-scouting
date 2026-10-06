# Drone Plant Scouting

Count and map every plant in a plastic-mulch field from drone photos, and follow
the same plants from flight to flight.

## Product idea

Vegetable farms on plastic mulch plant thousands of seedlings in regular rows.
Walking the field to count plants, find gaps or spot weak plants is slow and
inaccurate. A drone flight (DJI Mavic 3 Multispectral, RGB + NIR/Red/Red-Edge)
takes a few minutes, and these scripts turn the photos into:

- **Plant count** for the farm, with every plant counted once
- **A GPS position for every plant**, shown on an interactive satellite map
- **Plant health** (NDVI / NDRE) per plant
- **Evidence images**: every photo with the counted plants circled, so each
  number in the report can be checked by eye
- **Follow-up over the season**: the plant map of the first flight is placed onto
  later flights, so each plant can be tracked (still there? healthier?) even when
  the canopy has closed and plants can no longer be counted one by one

Typical users: farm managers, agronomists and crop-insurance or input-supply
teams who need stand counts, gap maps and health trends per field.

---

## Setup

```
pip install opencv-python numpy matplotlib
```

Put all scripts and `plant-count-config.json` in **one folder**. Each flight
(batch) is a sub-folder with the raw DJI files:

```
testing/
├── plant_count.py, dedupe_plants.py, image-processing.py, plant_map.py
├── project_reference.py, remove_overlap.py, ...      (all scripts)
├── plant-count-config.json
├── Laxman Mahale 05 Aug/          ← batch 1
│   ├── DJI_..._0001_D.JPG
│   ├── DJI_..._0001_MS_G.TIF / _MS_R.TIF / _MS_RE.TIF / _MS_NIR.TIF
│   └── ...
└── Laxman Mahale 13 Aug/          ← batch 2
```

Each capture needs its RGB photo (`_D.JPG`) **and** its multispectral bands
(`_MS_*.TIF`).

> **Mac users:** files copied from USB drives get hidden `._` copies. The
> scripts ignore them; to remove them: `dot_clean "FOLDER"`.

---

## Running a scouting

### 1. Normal: count every batch on its own

Run the plant count once per batch:

```
python plant_count.py --input "Laxman Mahale 05 Aug"
python plant_count.py --input "Laxman Mahale 13 Aug"
```

Use this while plants are still separate and visible (seedling to young plant).

### 2. Reference: count once, then follow the same plants

Count the first batch, then place its plants onto each later batch:

```
python plant_count.py --input "Laxman Mahale 05 Aug"

python project_reference.py "Laxman Mahale 05 Aug/output/plants_unique.csv" "Laxman Mahale 13 Aug"
python project_reference.py "Laxman Mahale 05 Aug/output/plants_unique.csv" "Laxman Mahale 22 Aug"
```

Use this once plants grow big and touch each other, when counting them again
from the photos becomes unreliable. The result tells, for every plant of the
first flight, whether it is still there, with its new NDVI/NDRE next to the old
one.

If the two flights do not line up (no RTK, or a visible offset), measure one
fixed object (a stake, pipe end, field corner) in both batches with
`gps_viewer.py` and add it:

```
python project_reference.py REF.csv NEW_BATCH --offset-from LAT1,LON1 --offset-to LAT2,LON2
```

> Tip: if `plant_count.py` has also been run on the new batch, the reference
> script uses that batch's `output/frame_alignment.csv` and
> `output/heading_calibration.json` and lines up better.

### 3. Optional: remove overlapping photos first

Any batch can be filtered before processing, so that only photos that do not
repeat the same ground are kept:

```
python remove_overlap.py "Laxman Mahale 05 Aug"
python plant_count.py --input "Laxman Mahale 05 Aug_no_overlap"
```

The new folder contains the kept photos and their bands, copied unchanged, so it
runs through `plant_count.py` (or the reference script) exactly like the
original folder.

---

## What each file does

### Main pipeline

| File | What it does |
|---|---|
| **`plant_count.py`** | The main script. For every photo in a batch: aligns RGB with the multispectral bands, finds the plastic mulch rows, detects plants (dark planting holes for seedlings, green blobs for leafy plants), applies the row rules (on a row, near its centre, one plant per planting position), converts each plant to GPS. Then removes duplicates, marks plants outside the farm boundary, and writes all CSVs, the evidence images and the map. |
| **`dedupe_plants.py`** | Called by `plant_count.py`. Removes plants counted twice: double circles in one photo, and the same plant seen in overlapping photos (matched by position after lining frames up). Also draws the final evidence images. Can be run on its own on a `plant_instances_batch.csv`. |
| **`image-processing.py`** | Shared functions: reading DJI metadata (GPS, height, heading, camera calibration), aligning RGB with the multispectral bands, NDVI/NDRE, pixel ↔ GPS conversion. Must sit next to the other scripts. |
| **`plant_map.py`** | Called by `plant_count.py`. Writes `plant_map.html`, an interactive satellite map with the batch's plants built in (just double-click it). |
| **`plant-count-config.json`** | All settings: detection thresholds, row rules, plant spacing, farm boundary (with corner names), heading correction, de-duplication options. |

### Follow-up and filtering

| File | What it does |
|---|---|
| **`project_reference.py`** | Places the plants of an earlier flight onto a new batch. Only the multispectral area of each photo is used; points must lie on a plastic row; slightly-off points are moved onto the nearest free plant (within 15 cm); each plant is kept only in the one photo where it shows the strongest plant evidence; two plants can never be closer than 60% of the plant spacing. Writes marked photos, per-plant status with new and old NDVI/NDRE, and a map. |
| **`remove_overlap.py`** | Like the strip tracker for video: follows the ground from photo to photo along each flight line and keeps a photo each time the ground has moved one full image. Kept photos are copied with their bands and metadata into `<folder>_no_overlap/`. Tilted photos are skipped. Overlap *between* neighbouring flight lines is not removed. |

### Checking tools

| File | What it does |
|---|---|
| **`gps_viewer.py`** | Opens a photo (or a folder): hover to see the GPS coordinate under the mouse, click to pin points (saved to `clicked_points.csv`). Used to measure farm corners and tie points between flights. |
| **`plot_points_on_image.py`** | Draws given GPS coordinates on a photo and joins them (e.g. farm corners, pins), plus a "drone GPS" marker that should sit near the image centre. Quick visual check that positions are right. |
| **`check_boundary.py`** | Says whether a coordinate is inside the farm boundary and how far from the edge; `--show` opens the boundary on a satellite map. |
| **`plot_drone_positions.py`** | Plots where every photo was taken (flight path, heading arrows, height, RTK flag) and flags suspicious GPS jumps. |
| **`plot_points.py`** | Plots coordinates and photo positions on a flat east/north plane in metres. |

---

## Outputs

### `plant_count.py` → `<batch>/output/`

| File | Contents |
|---|---|
| `plant_count_batch.csv` | One row per photo: raw detections, plants counted in this photo, repeats, outside the farm, running total, rows found, plant spacing, metadata |
| `plants_unique.csv` | **One row per plant** (= the field count): GPS, photos it appears in, NDVI/NDRE, inside farm or not. This is the file used as reference for later flights |
| `plant_instances_batch.csv` | Every raw detection before duplicates are removed |
| `plants_detection_map.csv` | Every detection linked to its unique plant (first seen / repeat) |
| `frame_alignment.csv` | How each photo was lined up with its neighbours (quality check) |
| `final_overlays/*_final.jpg` | Evidence images: **yellow** = counted here, **magenta** = already counted in an earlier photo, **red** = outside the farm boundary; farm corners in orange |
| `plant_map.html` | Interactive map of `plants_unique.csv` |

### `project_reference.py` → `<new batch>/output_reference/`

| File | Contents |
|---|---|
| `reference_plants.csv` | Every reference plant with its status (`found`, `found (snapped)`, `too close…`, `no plant evidence…`), photo, pixel, how far it was snapped, new and reference NDVI/NDRE |
| `reference_found.csv` | Only the found plants: **its row count is the number of plants found** |
| `reference_frames.csv` | Per photo: points in the TIF area, on rows, with evidence, kept in this photo |
| `overlays/*_ref.jpg` | Marked photos: each found plant is drawn once, on its best photo |
| `reference_map.html` | Map of the found plants |

### `remove_overlap.py` → `<batch>_no_overlap/`

The kept photos with their `_MS_*.TIF` bands (unchanged) and `segments.csv`
listing which photos were kept or skipped.

---

## Important settings (`plant-count-config.json`)

| Setting | Meaning |
|---|---|
| `farm_boundary` | Farm outline as `[lat, lon, "name"]` points in order around the field. Plants outside are drawn red and not counted. A `farm_boundary.json` inside a batch folder overrides it for that batch |
| `row_center_band_frac` | How far from the row centre a plant may be (0.25 = middle half of the row) |
| `crowd_min_gap_frac` | Minimum gap between two plants along a row, as a fraction of the plant spacing |
| `plant_spacing_cm` | `null` = measured automatically; set a number if the measured value is wrong |
| `plastic_v_min` | `"auto"` adapts to bright or cloudy days; a number forces a fixed brightness |
| `dark_spot_require_green` | `true` counts only planting holes with some green in them (ignores empty holes) |
| `heading_offset_deg` | `"auto"` measures the camera heading error from the photos; a number fixes it; `"off"` disables it |

---

## Good to know

- **RTK:** check `rtk_flag` in `plant_count_batch.csv` (50 = RTK fixed, cm
  accuracy). Without RTK, positions can be off by 1–5 m between flights; use
  the `--offset-from/--offset-to` option of the reference script.
- **Tilted photos** (camera not pointing straight down) give wrong positions
  and are skipped by `remove_overlap.py`; leave them out of reports.
- **Growth stage:** individual counting works best from seedling to young plant.
  For large or touching plants use the reference procedure.
- **Evidence first:** always look at a few `final_overlays` / `overlays`
  images before trusting a number.
