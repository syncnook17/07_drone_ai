"""
รวมจุด detection ที่มีพิกัด → dedup → เขียนไฟล์ภูมิสารสนเทศ + หน้าแผนที่ Leaflet

ไม่พึ่ง lib ภายนอก: GeoJSON = json, KML = XML เขียนเอง, CSV = csv, แผนที่ = HTML+Leaflet(CDN)
"""

from __future__ import annotations

import csv
import html
import json
import os
import statistics
from dataclasses import dataclass, field

from core.geo import haversine_m

GEOJSON_NAME = "detections.geojson"
KML_NAME = "detections.kml"
CSV_NAME = "detections.csv"
MAP_NAME = "map.html"
SCATTER_NAME = "map_scatter.png"

_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


@dataclass
class DetPoint:
    cls: str
    lat: float
    lon: float
    conf: float = 0.0
    count: int = 1                 # กี่เฟรม/กี่ภาพ ที่เห็นวัตถุนี้
    oblique: bool = False
    sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- dedup

def dedup_video(raw: list[dict]) -> list[DetPoint]:
    """raw item: {track_id, cls, lat, lon, conf, oblique, frame}
    รวมตาม track_id → 1 จุดต่อวัตถุ (พิกัด = median)"""
    by_id: dict = {}
    for r in raw:
        by_id.setdefault(r["track_id"], []).append(r)
    pts = []
    for tid, items in by_id.items():
        pts.append(
            DetPoint(
                cls=items[0]["cls"],
                lat=statistics.median(i["lat"] for i in items),
                lon=statistics.median(i["lon"] for i in items),
                conf=max(i["conf"] for i in items),
                count=len(items),
                oblique=any(i.get("oblique") for i in items),
                sources=[f"track {tid}"],
            )
        )
    return pts


def dedup_photos(raw: list[dict], merge_dist_m: float = 2.0) -> list[DetPoint]:
    """raw item: {cls, lat, lon, conf, oblique, source}
    รวมจุด class เดียวกันที่อยู่ใกล้กว่า merge_dist_m เป็นจุดเดียว"""
    clusters: list[DetPoint] = []
    for r in raw:
        hit = None
        for c in clusters:
            if c.cls == r["cls"] and haversine_m(c.lat, c.lon, r["lat"], r["lon"]) <= merge_dist_m:
                hit = c
                break
        if hit is None:
            clusters.append(
                DetPoint(
                    cls=r["cls"], lat=r["lat"], lon=r["lon"], conf=r["conf"],
                    count=1, oblique=bool(r.get("oblique")),
                    sources=[r.get("source", "")],
                )
            )
        else:
            n = hit.count
            hit.lat = (hit.lat * n + r["lat"]) / (n + 1)
            hit.lon = (hit.lon * n + r["lon"]) / (n + 1)
            hit.conf = max(hit.conf, r["conf"])
            hit.count = n + 1
            hit.oblique = hit.oblique or bool(r.get("oblique"))
            if r.get("source"):
                hit.sources.append(r["source"])
    return clusters


# ---------------------------------------------------------------- writers

def _class_colors(points: list[DetPoint]) -> dict[str, str]:
    names = sorted({p.cls for p in points})
    return {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(names)}


def write_geojson(points: list[DetPoint], out_dir: str) -> str:
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(p.lon, 8), round(p.lat, 8)]},
                "properties": {
                    "class": p.cls,
                    "confidence": round(p.conf, 3),
                    "count": p.count,
                    "oblique": p.oblique,
                    "sources": ", ".join(s for s in p.sources if s),
                },
            }
            for p in points
        ],
    }
    path = os.path.join(out_dir, GEOJSON_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    return path


def write_csv(points: list[DetPoint], out_dir: str) -> str:
    path = os.path.join(out_dir, CSV_NAME)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "latitude", "longitude", "confidence", "count", "oblique", "sources"])
        for p in points:
            w.writerow([p.cls, f"{p.lat:.8f}", f"{p.lon:.8f}", f"{p.conf:.3f}",
                        p.count, int(p.oblique), " | ".join(s for s in p.sources if s)])
    return path


def write_kml(points: list[DetPoint], out_dir: str) -> str:
    colors = _class_colors(points)

    def kml_color(hex_rgb: str) -> str:  # KML = aabbggrr
        h = hex_rgb.lstrip("#")
        return f"ff{h[4:6]}{h[2:4]}{h[0:2]}".lower()

    styles = "".join(
        f'<Style id="c{i}"><IconStyle><color>{kml_color(colors[n])}</color>'
        f'<scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>'
        f"</IconStyle></Style>"
        for i, n in enumerate(sorted(colors))
    )
    idx = {n: i for i, n in enumerate(sorted(colors))}
    marks = "".join(
        f"<Placemark><name>{html.escape(p.cls)}</name>"
        f"<description>conf {p.conf:.2f} · เห็น {p.count} ครั้ง"
        f"{' · เอียงมาก' if p.oblique else ''}</description>"
        f'<styleUrl>#c{idx[p.cls]}</styleUrl>'
        f"<Point><coordinates>{p.lon:.8f},{p.lat:.8f},0</coordinates></Point></Placemark>"
        for p in points
    )
    doc = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        "<name>DroneCVEdu detections</name>"
        f"{styles}{marks}</Document></kml>"
    )
    path = os.path.join(out_dir, KML_NAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


def scatter_png(points: list[DetPoint], out_dir: str) -> str | None:
    if not points:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = _class_colors(points)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, col in colors.items():
        xs = [p.lon for p in points if p.cls == name]
        ys = [p.lat for p in points if p.cls == name]
        ax.scatter(xs, ys, s=40, c=col, label=f"{name} ({len(xs)})", edgecolors="k", linewidths=0.4)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Detected object locations (no basemap)")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, SCATTER_NAME)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


_MAP_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>DroneCVEdu — แผนที่วัตถุที่ตรวจพบ</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body,#map{height:100%;margin:0}
  #warn{position:absolute;z-index:9999;top:8px;left:50px;right:8px;background:#fee;border:1px solid #c33;
        padding:6px 10px;border-radius:6px;font:13px system-ui;display:none}
  .legend{background:#fff;padding:6px 8px;border-radius:6px;font:12px system-ui;line-height:1.5;box-shadow:0 1px 4px rgba(0,0,0,.3)}
  .legend i{display:inline-block;width:11px;height:11px;margin-right:5px;border:1px solid #333}
</style></head><body>
<div id="warn"></div><div id="map"></div>
<script>
var DATA = __GEOJSON__;
var COLORS = __COLORS__;
var map = L.map('map');
var tiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19, attribution:'© OpenStreetMap'});
tiles.on('tileerror', function(){
  document.getElementById('warn').style.display='block';
  document.getElementById('warn').textContent='โหลดแผนที่ฐานไม่ได้ (ไม่มีเน็ต) — หมุดยังแสดงตามพิกัดจริง';
});
tiles.addTo(map);
var layer = L.geoJSON(DATA, {
  pointToLayer:function(f,latlng){
    var c = COLORS[f.properties.class] || '#e6194B';
    return L.circleMarker(latlng,{radius:7,color:'#000',weight:1,fillColor:c,fillOpacity:.9});
  },
  onEachFeature:function(f,l){
    var p=f.properties;
    l.bindPopup('<b>'+p.class+'</b><br>conf '+p.confidence+'<br>เห็น '+p.count+' ครั้ง'
      +(p.oblique?'<br><span style="color:#c33">ภาพเอียงมาก คลาดเคลื่อนสูง</span>':''));
  }
}).addTo(map);
if (DATA.features.length){ map.fitBounds(layer.getBounds().pad(0.2)); }
else { map.setView([13.736,100.523],13); }
var legend=L.control({position:'bottomright'});
legend.onAdd=function(){var d=L.DomUtil.create('div','legend');
  for(var k in COLORS){d.innerHTML+='<i style="background:'+COLORS[k]+'"></i>'+k+'<br>';}
  return d;};
legend.addTo(map);
</script></body></html>"""


def render_map_html(points: list[DetPoint], out_dir: str) -> str:
    colors = _class_colors(points)
    fc = json.loads(open(write_geojson(points, out_dir), encoding="utf-8").read())
    doc = (_MAP_TEMPLATE
           .replace("__GEOJSON__", json.dumps(fc, ensure_ascii=False))
           .replace("__COLORS__", json.dumps(colors, ensure_ascii=False)))
    path = os.path.join(out_dir, MAP_NAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


def export_all(points: list[DetPoint], out_dir: str) -> dict:
    """เขียนทุกไฟล์ + คืน dict สรุปสำหรับ session / UI"""
    os.makedirs(out_dir, exist_ok=True)
    geojson = write_geojson(points, out_dir)
    kml = write_kml(points, out_dir)
    csv_path = write_csv(points, out_dir)
    map_html = render_map_html(points, out_dir)
    scatter = scatter_png(points, out_dir)

    per_class: dict[str, int] = {}
    for p in points:
        per_class[p.cls] = per_class.get(p.cls, 0) + 1
    lats = [p.lat for p in points]
    lons = [p.lon for p in points]
    return {
        "n_points": len(points),
        "per_class": per_class,
        "bbox": [min(lons), min(lats), max(lons), max(lats)] if points else None,
        "has_oblique": any(p.oblique for p in points),
        "files": {"geojson": geojson, "kml": kml, "csv": csv_path,
                  "map_html": map_html, "scatter": scatter},
    }
