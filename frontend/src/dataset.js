import { apiFetch } from './api.js';
import { currentProfile } from './content/pipelines.js';
import { dom } from './dom.js';
import { drawFootprints } from './footprint.js';
import { fmtBytes, fmtLatLon, fmtTime, joinParts, num } from './format.js';
import { state } from './state.js';

let datasetRequest = 0;   // responses for a pair that is no longer selected are ignored

export async function loadDataset() {
  const pair = dom.pairSelect.value;
  const token = ++datasetRequest;
  dom.datasetPair.textContent = pair ? `${currentProfile().label} · ${pair}` : '';
  if (!state.connected || !pair) {
    showDatasetMessage(pair ? '' : 'No image pair available for this sensor pair.');
    return;
  }
  showDatasetMessage('Reading product labels…');
  try {
    const info = await apiFetch(`/api/dataset?${new URLSearchParams({ pipeline: state.pipeline, pair_id: pair })}`);
    if (token !== datasetRequest) return;
    renderDataset(info);
  } catch (err) {
    if (token !== datasetRequest) return;
    showDatasetMessage(`Could not read metadata for ${pair}: ${err.message}`);
  }
}

function showDatasetMessage(msg) {
  dom.datasetMessage.textContent = msg;
  dom.datasetMessage.hidden = !msg;
  dom.datasetContent.hidden = true;
}

// [label, value of one product record]. null renders as "—"; rows empty for both are dropped.
const META_ROWS = [
  ['Product ID',       r => r.product_id],
  ['Instrument',       r => joinParts(r.mission, r.instrument)],
  ['Acquired',         r => fmtTime(r.start_time, r.stop_time)],
  ['Orbit',            r => joinParts(r.orbit != null ? `#${r.orbit}` : null, r.orbit_direction?.toLowerCase())],
  ['Product',          r => joinParts(r.processing_level, r.format)],
  ['Image size',       r => (r.lines ? joinParts(`${num(r.lines, 0)} × ${num(r.samples, 0)} px`, r.data_type) : null)],
  ['Resolution (GSD)', r => (r.gsd_m != null ? `${num(r.gsd_m, r.gsd_m < 1 ? 3 : 2)} m/px` : null)],
  ['Footprint',        r => (r.footprint ? `${num(r.footprint.length_km, 1)} × ${num(r.footprint.width_km, 1)} km` : null)],
  ['Centre',           r => (r.footprint ? fmtLatLon(r.footprint.center_lat, r.footprint.center_lon) : null)],
  ['Altitude',         r => (r.altitude_km != null ? `${num(r.altitude_km, 2)} km` : null)],
  ['Sun elevation',    r => (r.sun_elev_deg != null ? `${num(r.sun_elev_deg, 2)}°` : null)],
  ['Sun azimuth',      r => (r.sun_azim_deg != null ? `${num(r.sun_azim_deg, 2)}°${r.sun_azim_convention === 'north_cw' ? ' from N' : ''}` : null)],
  ['Solar incidence',  r => (r.incidence_deg != null ? `${num(r.incidence_deg, 2)}°` : null)],
  ['Roll / pitch',     r => (r.roll_deg != null ? `${num(r.roll_deg, 2)}° / ${num(r.pitch_deg, 2)}°` : null)],
  ['Projection',       r => joinParts(r.projection, r.area)],
  ['Corners from',     r => r.corners_source],
  ['File size',        r => fmtBytes(r.file_size_bytes)],
];

function renderDataset(info) {
  const ref = info.reference;
  const o = info.ohrc, r = info.ref, pair = info.pair || {};

  // Pair-level comparison: [label, value, note]
  const facts = [
    ['Overlap',
      pair.overlap_km ? `${num(pair.overlap_km[0], 1)} × ${num(pair.overlap_km[1], 1)} km` : 'none',
      pair.overlap_pct_of_ohrc != null ? `bounding box, ${num(Math.min(pair.overlap_pct_of_ohrc, 100), 0)}% of the OHRC` : 'bounding box of the corners'],
    ['Acquired apart',
      pair.days_apart == null ? '—' : pair.days_apart >= 365 ? `${num(pair.days_apart / 365.25, 1)} years` : `${num(pair.days_apart, 0)} days`,
      'between the two exposures'],
    ['Scale ratio',
      pair.scale_ratio != null ? `${num(pair.scale_ratio, 1)}×` : '—',
      `${ref} pixel vs OHRC pixel`],
    ['Sun azimuth gap',
      pair.sun_azim_gap_deg != null ? `${num(pair.sun_azim_gap_deg, 1)}°` : '—',
      pair.sun_azim_gap_deg != null ? 'change in shadow direction' : `needs ${ref} azimuth from north`],
    ['Sun elevation gap',
      pair.sun_elev_gap_deg != null ? `${num(pair.sun_elev_gap_deg, 1)}°` : '—',
      pair.sun_elev_gap_deg != null ? 'change in shadow length' : `no ${ref} sun elevation`],
  ];
  dom.pairFacts.innerHTML = '';
  facts.forEach(([label, value, note]) => {
    const div = document.createElement('div');
    div.className = 'pair-fact';
    div.innerHTML = '<dt></dt><dd><span></span><small></small></dd>';
    div.querySelector('dt').textContent = label;
    div.querySelector('dd span').textContent = value;
    div.querySelector('small').textContent = note;
    dom.pairFacts.appendChild(div);
  });

  // Label metadata side by side, then any product-specific notes
  const rows = META_ROWS.map(([label, fn]) => [label, fn(o), fn(r)]);
  const noteKeys = [...new Set([...Object.keys(o.notes || {}), ...Object.keys(r.notes || {})])];
  noteKeys.forEach(k => rows.push([k, o.notes?.[k] ?? null, r.notes?.[k] ?? null]));

  const table = dom.metaTable;
  table.innerHTML = '<thead><tr><th scope="col"></th><th scope="col">OHRC (source)</th><th scope="col"></th></tr></thead><tbody></tbody>';
  table.querySelector('thead th:last-child').textContent = `${ref} (reference)`;
  const tbody = table.querySelector('tbody');
  rows.forEach(([label, a, b]) => {
    if (a == null && b == null) return;
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.scope = 'row';
    th.textContent = label;
    tr.appendChild(th);
    [a, b].forEach(v => {
      const td = document.createElement('td');
      td.textContent = v ?? '—';
      td.classList.toggle('is-none', v == null);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  drawFootprints(o, r, ref);
  dom.datasetMessage.hidden = true;
  dom.datasetContent.hidden = false;
}
