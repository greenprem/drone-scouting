"""
Write an interactive plant map (Leaflet) for one batch.

The batch's plants_unique.csv is embedded in the HTML, so it opens straight
from the output folder with a double-click - no file picking, no server.
Points outside the farm boundary are hidden (tick "Show points outside farm").
"""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"></script>
<style>
html,body{height:100%;margin:0;font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
#map{position:absolute;inset:0}
.panel{position:absolute;z-index:1000;top:12px;right:12px;background:#fff;color:#1d2a1f;padding:12px 14px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.25);width:230px}
.panel h1{font-size:16px;margin:0 0 8px}
.panel label{display:block;margin:6px 0}
button{width:100%;padding:8px;font:inherit;font-weight:600;background:#1a9850;color:#fff;border:0;border-radius:6px;cursor:pointer}
button:hover{background:#15803f}
#status{font-size:12px;color:#555;margin-top:6px}
.bar{height:10px;border-radius:3px;margin-top:8px}
.ends{display:flex;justify-content:space-between;font-size:12px;color:#555}
.leaflet-popup-content table td{padding:1px 6px 1px 0}
</style></head><body>
<div id="map"></div>
<div class="panel">
 <h1>__TITLE__</h1>
 <button id="open">Open another CSV</button>
 <input type="file" id="file" accept=".csv,text/csv" hidden>
 <div id="status">No file loaded</div>
 <div id="controls" hidden>
  <label>Color by <select id="metric"></select></label>
  <div class="bar" id="bar"></div><div class="ends"><span id="lo"></span><span id="hi"></span></div>
 </div>
 <label><input type="checkbox" id="outside"> Show points outside farm (grey)</label>
 <label><input type="checkbox" id="base" checked> Satellite basemap</label>
</div>
<script>
const map=L.map('map',{preferCanvas:true,maxZoom:23}).setView([20.356,73.906],17);
const sat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxNativeZoom:19,maxZoom:23,attribution:'Imagery &copy; Esri'}).addTo(map);
const osm=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxNativeZoom:19,maxZoom:23,attribution:'&copy; OpenStreetMap'});
document.getElementById('base').onchange=e=>{if(e.target.checked){map.removeLayer(osm);sat.addTo(map)}else{map.removeLayer(sat);osm.addTo(map)}};

// Farm boundary + batch data are written in by plant_count.py
const FARM=__BOUNDARY__;            // [[lat,lon],...] or null
const EMBED_NAME=__NAME__, EMBED_CSV=__DATA__;
const farmLayer=FARM?L.polygon(FARM,{color:'#fff',weight:2,dashArray:'6 4',fill:false,interactive:false}).addTo(map):null;
if(farmLayer)map.fitBounds(farmLayer.getBounds(),{padding:[20,20]});
function fitView(){if(farmLayer)map.fitBounds(farmLayer.getBounds(),{padding:[20,20]});else if(group)map.fitBounds(group.getBounds(),{padding:[20,20]})}
function inFarm(lat,lon){if(!FARM)return true;let c=false;for(let i=0,j=FARM.length-1;i<FARM.length;j=i++){
 const[yi,xi]=FARM[i],[yj,xj]=FARM[j];
 if((yi>lat)!==(yj>lat)&&lon<(xj-xi)*(lat-yi)/(yj-yi)+xi)c=!c}return c}

const stops=['#a50026','#f46d43','#fee08b','#a6d96a','#1a9850'];
document.getElementById('bar').style.background='linear-gradient(90deg,'+stops.join(',')+')';
function ramp(t){if(!isFinite(t))return '#888';t=Math.max(0,Math.min(1,t))*(stops.length-1);const i=Math.min(Math.floor(t),stops.length-2),f=t-i;
 const a=stops[i].match(/\w\w/g).map(h=>parseInt(h,16)),b=stops[i+1].match(/\w\w/g).map(h=>parseInt(h,16));
 return 'rgb('+a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(',')+')'}
function q(s,p){return s[Math.floor(p*(s.length-1))]}

const renderer=L.canvas({padding:.5});
let rows=[],markers=[],group=null,outRows=[],outGroup=null;
const status=document.getElementById('status'),metric=document.getElementById('metric');
const fileInput=document.getElementById('file');
document.getElementById('open').onclick=()=>fileInput.click();
fileInput.onchange=e=>{const f=e.target.files[0];if(!f)return;status.textContent='Loading '+f.name+'…';
 Papa.parse(f,{header:true,dynamicTyping:true,skipEmptyLines:true,complete:res=>load(f.name,res)});fileInput.value=''};

function load(name,res){
 const fields=res.meta.fields||[];
 const latK=fields.find(k=>/^(lat|latitude)$/i.test(k.trim())),lonK=fields.find(k=>/^(lon|lng|long|longitude)$/i.test(k.trim()));
 if(!latK||!lonK){status.textContent='This file has no lat/lon columns. Add columns named "lat" and "lon".';return}
 const valid=res.data.filter(r=>isFinite(r[latK])&&isFinite(r[lonK]));
 rows=valid.filter(r=>inFarm(r[latK],r[lonK]));
 outRows=valid.filter(r=>!inFarm(r[latK],r[lonK]));
 if(group)map.removeLayer(group);
 if(outGroup)map.removeLayer(outGroup);
 const popup=r=>'<table>'+fields.map(k=>`<tr><td><b>${k}</b></td><td>${r[k]??''}</td></tr>`).join('')+'</table>';
 outGroup=L.featureGroup(outRows.map(r=>L.circleMarker([r[latK],r[lonK]],{renderer,radius:2,weight:0,fillColor:'#999',fillOpacity:.6}).bindPopup(()=>popup(r))));
 if(document.getElementById('outside').checked)outGroup.addTo(map);
 if(!rows.length){markers=[];group=null;status.textContent=`None of the ${valid.length.toLocaleString()} points in ${name} are inside the farm boundary.`;return}
 markers=rows.map(r=>L.circleMarker([r[latK],r[lonK]],{renderer,radius:3,weight:0,fillOpacity:.9})
  .bindPopup(()=>popup(r),{maxWidth:360}));
 group=L.featureGroup(markers).addTo(map);
 fitView();
 const numeric=fields.filter(k=>k!==latK&&k!==lonK&&rows.some(r=>typeof r[k]==='number'));
 metric.innerHTML='<option value="">Single color</option>'+numeric.map(k=>`<option ${/ndvi/i.test(k)?'selected':''}>${k}</option>`).join('');
 document.getElementById('controls').hidden=false;
 status.textContent=`${rows.length.toLocaleString()} points inside the farm, ${outRows.length.toLocaleString()} outside skipped (${name})`;
 color();
}
function color(){const k=metric.value;
 if(!k){markers.forEach(m=>m.setStyle({fillColor:'#2b8cff'}));lo.textContent=hi.textContent='';return}
 const v=rows.map(r=>typeof r[k]==='number'?r[k]:NaN),s=v.filter(isFinite).sort((a,b)=>a-b);
 const a=q(s,.02),b=q(s,.98);
 lo.textContent=+a.toFixed(3);hi.textContent=+b.toFixed(3);
 markers.forEach((m,i)=>m.setStyle({fillColor:ramp((v[i]-a)/(b-a||1))}))}
const lo=document.getElementById('lo'),hi=document.getElementById('hi');
metric.onchange=color;
document.getElementById('outside').onchange=e=>{if(!outGroup)return;e.target.checked?outGroup.addTo(map):map.removeLayer(outGroup)};
if(EMBED_CSV){status.textContent='Loading '+EMBED_NAME+'…';
 Papa.parse(EMBED_CSV,{header:true,dynamicTyping:true,skipEmptyLines:true,complete:res=>load(EMBED_NAME,res)})}
map.on('zoomend',()=>{const z=map.getZoom();markers.forEach(m=>m.setRadius(z>=21?6:z>=20?4:3))});
</script></body></html>
"""


def _js(v) -> str:
    # safe inside <script>: escape "</" so the data can never close the tag
    return json.dumps(v).replace("</", "<\\/")


def write_plant_map(csv_path: Path, html_path: Path, boundary=None, title="Plant map"):
    csv_text = Path(csv_path).read_text(encoding="utf-8")
    html = (TEMPLATE
            .replace("__BOUNDARY__", _js(boundary if boundary and len(boundary) >= 3 else None))
            .replace("__NAME__", _js(Path(csv_path).name))
            .replace("__DATA__", _js(csv_text))
            .replace("__TITLE__", title.replace("<", "&lt;")))
    Path(html_path).write_text(html, encoding="utf-8")
    return html_path
