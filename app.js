// ============================================================
// SIH26166 — OHRC ↔ NAC / TMC-2 Registration Pipeline GUI
// Application Logic: Connection, SSE, Pipeline, Results
// ============================================================

// ---------- Pipeline Stage Definitions ----------
// algo / model / desc describe what backend/engine.py and backend/pipelines/* actually run.
// Keep them in sync with the code when a step changes.
const PIPELINE_GROUPS = [
  {
    label: 'Data loading',
    steps: [
      { id: 1, name: 'Loading OHRC Image',
        algo: 'PDS4 label parsing (corner lat/lon, GSD) + raster read',
        model: null,
        desc: 'Reads the Chandrayaan-2 OHRC PDS4 raster and its XML label, and takes the four corner coordinates and ground sample distance from the label.' },
      { id: 2, name: 'Loading {REF} Reference',
        algo: '{REF_LOAD_ALGO}',
        model: null,
        desc: '{REF_LOAD_DESC}' },
    ]
  },
  {
    label: 'Preprocessing',
    steps: [
      { id: 3, name: 'Overlap Detection & Crop',
        algo: 'Convex polygon intersection of the lat/lon footprints, 150 m padding',
        model: null,
        desc: 'Intersects the two ground footprints and crops both images to the shared region. 16-bit products are stretched to 8-bit on the crop only.' },
      { id: 4, name: 'Resolution Scheme',
        algo: '{FINE_ALGO}',
        model: null,
        desc: 'Picks two working resolutions: a coarse level for global alignment, grown by 1.25× until both images fit in 2 MP, and a fine level for tie points. {FINE_DESC}' },
      { id: 5, name: 'Appearance Preprocessing',
        algo: 'Percentile stretch (1–99%) → CLAHE (clip 2.0 OHRC, 4.0 {REF}) → histogram matching',
        model: null,
        desc: 'Makes two different cameras look alike: normalises brightness, boosts local contrast, then matches the reference histogram to the OHRC.' },
    ]
  },
  {
    label: 'Orientation',
    steps: [
      { id: 6, name: 'Orientation Search (ZNCC)',
        algo: 'ZNCC template matching on gradient magnitude, 4 flips × scales 0.80–1.24, phase-randomised null',
        model: null,
        desc: 'Checks whether both images show the same ground and which flip lines them up. A match must beat the phase-randomised noise floor by 2.5× to count.' },
      { id: 7, name: 'Residual Rotation Estimate',
        algo: 'Log-polar FFT magnitude + phase correlation (Fourier–Mellin)',
        model: null,
        desc: 'Rotates the OHRC by a known 7° and recovers it, which calibrates the sign convention of the rotation estimator for this pair.' },
      { id: 8, name: 'SuperPoint + SuperGlue',
        algo: 'Keypoint detection + attentional graph matching at 1024 px, score threshold 0.15',
        model: 'SuperPoint + SuperGlue (magic-leap-community/superglue_outdoor)',
        desc: 'Runs the learned matcher on the coarse, contrast-enhanced pair (GPU if available). This first pass is a diagnostic view; the fit starts from the geometric prior in step 9.' },
    ]
  },
  {
    label: 'Coarse fit',
    steps: [
      { id: 9, name: 'Geometric Prior',
        algo: 'Least-squares pixel → lat/lon affine per product, composed into OHRC → {REF}',
        model: null,
        desc: 'Builds a first OHRC-to-{REF} transform purely from the two corner models. Exact at the corners, linear in between. Reports scale and whether the pair is mirrored.' },
      { id: 10, name: 'Dense Shift Field',
        algo: 'Band-wise ZNCC on gradient magnitude (8 bands, null margin ≥ 2) + polynomial fit',
        model: null,
        desc: 'Measures how far the prior drifts along the strip, one horizontal band at a time, and fits a 1st/2nd-order polynomial to remove pushbroom geometry error.' },
      { id: 11, name: 'Coarse Refinement',
        algo: 'SuperGlue on the prior-warped pair → robust fit (similarity / affine / homography)',
        model: 'SuperPoint + SuperGlue (magic-leap-community/superglue_outdoor)',
        desc: 'Matches the pre-warped images again and applies the correction only if it has at least 25 inliers and moves the centre by no more than 250 px.' },
    ]
  },
  {
    label: 'Fine matching',
    steps: [
      { id: 12, name: 'Tiled Matching at Native Resolution',
        algo: '{TILE_ALGO}',
        model: null,
        desc: 'Cuts the OHRC into overlapping tiles at fine GSD, pre-warps each with the local prior and matches it by dense correlation with sub-pixel phase correlation. Tiles whose points disagree by more than 6 px are dropped. Optional (Advanced): failed tiles can be rescued with shadow suppression, a coarser scale or an intensity match, and crater-anchored matches can be added.' },
      { id: 13, name: 'Final Model (RANSAC)',
        algo: 'MAD-based threshold → RANSAC similarity → affine → MAGSAC homography',
        model: null,
        desc: 'Fits the final transform on all fine correspondences. It moves to a richer model only if inliers do not drop; a homography needs 40+ inliers and 10% coverage. Trusted if ≥ 15 inliers cover ≥ 5% of the frame.' },
    ]
  },
  {
    label: 'Output',
    steps: [
      { id: 14, name: 'ECC Intensity Refinement',
        algo: 'OpenCV findTransformECC on gradient magnitude · gated: tried coarse → fine, each result kept only if the fine-tile ECC score improves',
        model: null,
        desc: 'Polishes the fitted transform by maximising intensity correlation instead of point distances. In gated mode (default) a result that does not improve the fine-level alignment is rejected, so ECC can no longer make the fit worse. Legacy and Off are selectable in the run configuration.' },
      { id: 15, name: 'Sub-pixel Refinement',
        algo: 'Not run in this build: the model passes through unchanged',
        model: null,
        desc: 'Sub-pixel offsets already come from the phase correlation inside step 12’s dense matcher, so this stage currently only reports.' },
      { id: 16, name: 'Consensus Tie Points',
        algo: 'Step 13 inliers + per-point reprojection residual',
        model: null,
        desc: 'Keeps the inlier correspondences as tie points and measures each one’s residual under the final model (green = low error, red = high).' },
      { id: 17, name: 'Final Warp & Export',
        algo: 'Perspective warp (bilinear) + overlay / checkerboard / correspondence figures',
        model: null,
        desc: 'Warps the OHRC into the {REF} frame for the result views and saves the homography at native and fine resolution (.npy).' },
      { id: 18, name: 'Evaluation & Report',
        algo: 'In-sample RMSE · spatial held-out RMSE (interleaved bands along the strip) · NMI vs shifted baseline · trust checks',
        model: null,
        desc: 'Scores the registration: point error on the fit data and on held-out stretches of ground, plus normalised mutual information against a deliberately misaligned copy. Trusted only if inliers, coverage, held-out error (≤ 5 px) and NMI (≥ 1.5× baseline) all pass. Writes metrics.json.' },
    ]
  }
];

const ALL_STEPS = PIPELINE_GROUPS.flatMap(g => g.steps.map(s => ({ ...s, group: g.label })));
const TOTAL_STEPS = ALL_STEPS.length;
const STEP_BY_ID = Object.fromEntries(ALL_STEPS.map(s => [s.id, s]));

// ---------- Sensor-pair pipelines (must match backend/pipelines/*) ----------
const PIPELINE_PROFILES = {
  ohrc_nac: {
    label: 'OHRC ↔ LRO NAC',
    ref: 'NAC',
    version: 'v4',
    title: 'OHRC ↔ NAC Registration Pipeline',
    text: {
      REF_LOAD_ALGO: 'PDS3 image read; corners and GSD from the known-product table',
      REF_LOAD_DESC: 'Reads the LRO NAC PDS3 image and looks up its footprint corners and native GSD.',
      FINE_ALGO: 'Fine GSD = coarser native GSD · coarse GSD ×1.25 until ≤ 2 MP',
      FINE_DESC: 'Fine matches the coarser of the two native GSDs.',
      TILE_ALGO: '640 px tiles, 25% overlap, ≤ 400 tiles · 3×3 ZNCC patches + phase correlation · per-tile median consensus',
    },
  },
  ohrc_tmc: {
    label: 'OHRC ↔ TMC-2',
    ref: 'TMC',
    version: 'v5',
    title: 'OHRC ↔ TMC-2 Registration Pipeline',
    text: {
      REF_LOAD_ALGO: 'PDS4 16-bit read · corner model refit on the OHRC latitude band',
      REF_LOAD_DESC: 'Reads the 16-bit TMC-2 strip; corners and GSD come from its own label. The corner model is refit on just the OHRC’s latitude band, because a full strip is a trapezoid no affine can represent.',
      FINE_ALGO: 'Fine GSD = 0.5 × coarser native GSD · anti-aliased resampling',
      FINE_DESC: 'Fine is half the coarser GSD to double the tile grid across the narrow overlap; anti-aliased resampling handles the ~19× scale gap.',
      TILE_ALGO: 'Adaptive tiles (256–640 px), 25% overlap, ≤ 400 tiles · 3×3 ZNCC patches + phase correlation · per-tile median consensus',
    },
  },
};

function currentProfile() {
  return PIPELINE_PROFILES[state.pipeline] || PIPELINE_PROFILES.ohrc_nac;
}

function fillTemplate(str) {
  const p = currentProfile();
  return str.replace(/\{(\w+)\}/g, (m, key) => key === 'REF' ? p.ref : (p.text[key] ?? m));
}

// ---------- Application State ----------
const state = {
  colabUrl: '',
  connected: false,
  pipeline: 'ohrc_nac',
  pairsByPipeline: {},  // pipeline id -> [pair ids] from /api/pipelines
  lastRun: null,        // { pipeline, pair_id } of the run whose results are shown
  running: false,
  currentStep: 0,
  progress: 0,
  stepStates: {},       // stepId -> 'pending' | 'running' | 'done' | 'error'
  stepDetails: {},      // stepId -> { detail, image, metrics }
  results: null,        // final results object
  resultImages: {},     // tab name -> base64 image
  flowCols: 0,          // boxes per row in the current flow layout
  openStep: null,       // step whose popup is open
};

const STATE_LABELS = { pending: 'Pending', running: 'Running', done: 'Done', error: 'Failed' };

// ---------- DOM References ----------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
  urlInput:       $('#colab-url-input'),
  connectBtn:     $('#connect-btn'),
  connStatus:     $('#connection-status'),
  configSection:  $('#config-section'),
  pipelineSelect: $('#pipeline-select'),
  pairSelect:     $('#pair-select'),
  eccMode:        $('#ecc-mode'),
  matcherSelect:  $('#matcher-select'),
  optShadow:      $('#opt-shadow'),
  optRescale:     $('#opt-rescale'),
  optPolarity:    $('#opt-polarity'),
  optCraters:     $('#opt-craters'),
  optRetry:       $('#opt-retry'),
  sunElev:        $('#sun-elev'),
  sunAzim:        $('#sun-azim'),
  sunConv:        $('#sun-conv'),
  sunHint:        $('#sun-hint'),
  datasetSection: $('#dataset-section'),
  datasetPair:    $('#dataset-pair'),
  datasetMessage: $('#dataset-message'),
  datasetContent: $('#dataset-content'),
  pairFacts:      $('#pair-facts'),
  footprintMap:   $('#footprint-map'),
  footprintLegend: $('#footprint-legend'),
  metaTable:      $('#meta-table'),
  resultNotes:    $('#result-notes'),
  headerTitle:    $('#header-title'),
  runBtn:         $('#run-btn'),
  runProgress:    $('#run-progress'),
  runStatus:      $('#run-status'),
  progressFill:   $('.progress-fill'),
  progressLabel:  $('.progress-label'),
  pipelineFlow:   $('#pipeline-flow'),
  flowGrid:       $('#flow-grid'),
  flowLines:      $('#flow-lines'),
  popup:          $('#step-popup'),
  popupBackdrop:  $('#popup-backdrop'),
  resultsSection: $('#results-section'),
  metricsGrid:    $('#metrics-grid'),
  viewerImage:    $('#viewer-image'),
  viewerCanvas:   $('.viewer-canvas'),
};

// ---------- API Helpers ----------
async function apiFetch(endpoint, options = {}) {
  const url = state.colabUrl.replace(/\/+$/, '') + endpoint;
  const headers = { 'Content-Type': 'application/json', ...options.headers };
  // ngrok free tier requires this header to bypass the warning page
  headers['ngrok-skip-browser-warning'] = 'true';
  const resp = await fetch(url, { ...options, headers });
  if (!resp.ok) throw new Error(`API error: ${resp.status} ${resp.statusText}`);
  return resp.json();
}

// ---------- Connection ----------
function setConnectionStatus(status, text) {
  const el = dom.connStatus;
  el.className = `status status-${status}`;
  el.querySelector('.status-text').textContent = text;
}

async function handleConnect() {
  const url = dom.urlInput.value.trim();
  if (!url) { dom.urlInput.focus(); return; }

  state.colabUrl = url;
  setConnectionStatus('connecting', 'Connecting…');
  dom.connectBtn.textContent = 'Connecting…';
  dom.connectBtn.disabled = true;

  try {
    const data = await apiFetch('/api/connect');
    state.connected = true;
    setConnectionStatus('connected', `Connected · ${data.gpu ? 'GPU' : 'CPU only'}`);
    dom.connectBtn.textContent = 'Connected';
    dom.configSection.hidden = false;
    dom.datasetSection.hidden = false;
    dom.runBtn.disabled = false;

    // Load the sensor-pair pipelines and the image pairs each one has data for
    try {
      const data = await apiFetch('/api/pipelines');
      state.pairsByPipeline = {};
      dom.pipelineSelect.innerHTML = '';
      data.pipelines.forEach(p => {
        state.pairsByPipeline[p.id] = p.pairs || [];
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = p.pairs && p.pairs.length ? p.label : `${p.label} (no data)`;
        dom.pipelineSelect.appendChild(opt);
      });
      dom.pipelineSelect.value = state.pairsByPipeline[state.pipeline] ? state.pipeline : data.default;
    } catch { /* older server: keep the static options */ }
    handlePipelineChange();

  } catch (err) {
    state.connected = false;
    setConnectionStatus('disconnected', 'Connection failed');
    dom.connectBtn.textContent = 'Retry';
    dom.connectBtn.disabled = false;
    console.error('Connection error:', err);
  }
}

function handlePipelineChange() {
  state.pipeline = dom.pipelineSelect.value;
  const profile = currentProfile();

  // Image pairs for this sensor pair
  const pairs = state.pairsByPipeline[state.pipeline] || [];
  dom.pairSelect.innerHTML = '';
  pairs.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p; opt.textContent = p;
    dom.pairSelect.appendChild(opt);
  });
  if (!pairs.length) {
    const opt = document.createElement('option');
    opt.value = ''; opt.textContent = `No ${profile.ref} pairs found in data/raw/${profile.ref.toLowerCase()}`;
    dom.pairSelect.appendChild(opt);
  }
  dom.runBtn.disabled = !state.connected || state.running || !pairs.length;

  dom.headerTitle.textContent = profile.title;
  $$('.ref-name').forEach(el => { el.textContent = profile.ref; });
  if (!state.running) renderPipeline();
  loadPairInfo();
  loadDataset();
}

// ---------- Sun geometry (pre-filled from sun.json / labels, editable) ----------
function fmtSun(sun) {
  if (!sun) return 'not available';
  const parts = [];
  if (sun.elev_deg != null) parts.push(`elevation ${sun.elev_deg.toFixed(2)}°`);
  if (sun.azim_deg != null) parts.push(`azimuth ${sun.azim_deg.toFixed(2)}° (${sun.azim_convention || 'unknown'})`);
  return `${parts.join(', ')} from ${sun.source}`;
}

function setSunField(input, value) {
  input.value = value ?? '';
  input.dataset.prefill = input.value;
}

async function loadPairInfo() {
  const pair = dom.pairSelect.value;
  const ref = currentProfile().ref;
  setSunField(dom.sunElev, ''); setSunField(dom.sunAzim, ''); setSunField(dom.sunConv, '');
  dom.sunHint.textContent = '';
  if (!state.connected || !pair) return;
  try {
    const info = await apiFetch(`/api/pair_info?${new URLSearchParams({ pipeline: state.pipeline, pair_id: pair })}`);
    const r = info.ref_sun;
    setSunField(dom.sunElev, r?.elev_deg != null ? r.elev_deg.toFixed(2) : '');
    setSunField(dom.sunAzim, r?.azim_deg != null ? r.azim_deg.toFixed(2) : '');
    setSunField(dom.sunConv, r?.azim_convention || '');
    dom.sunHint.textContent =
      `OHRC: ${fmtSun(info.ohrc_sun)}. ${ref}: ${r ? fmtSun(r) : `no sun.json in data/raw/${ref.toLowerCase()}/${pair}/`}. ` +
      (ref === 'NAC' ? 'LROC product pages list the incidence angle: elevation = 90° − incidence. ' : '') +
      'Azimuths are compared only when both are measured clockwise from north.';
  } catch (err) {
    dom.sunHint.textContent = `Could not read sun geometry for ${pair}: ${err.message}`;
  }
}

// Only fields the user changed are sent; untouched ones keep the file/label value on the server.
function sunOverrides() {
  const out = {};
  if (dom.sunElev.value !== dom.sunElev.dataset.prefill && dom.sunElev.value !== '') out.ref_sun_elev = dom.sunElev.value;
  if (dom.sunAzim.value !== dom.sunAzim.dataset.prefill && dom.sunAzim.value !== '') out.ref_sun_azim = dom.sunAzim.value;
  if (dom.sunConv.value !== dom.sunConv.dataset.prefill && dom.sunConv.value !== '') out.ref_sun_conv = dom.sunConv.value;
  return out;
}

// ---------- Dataset (label metadata of the chosen pair) ----------
const MOON_KM_PER_DEG = 2 * Math.PI * 1737.4 / 360;
let datasetRequest = 0;   // ignores responses for a pair that is no longer selected

async function loadDataset() {
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

const num = (v, digits) => (v == null ? null : Number(v).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }));

const asUtc = (t) => Date.parse(/Z$/.test(t) ? t : `${t}Z`);

function fmtTime(start, stop) {
  if (!start) return null;
  const t = start.replace('T', ' ').replace(/Z$/, '').slice(0, 19);
  const secs = stop ? (asUtc(stop) - asUtc(start)) / 1000 : NaN;
  return Number.isFinite(secs) ? `${t} UTC (${num(secs, 1)} s)` : `${t} UTC`;
}

function fmtBytes(b) {
  if (b == null) return null;
  return b >= 1e9 ? `${num(b / 1e9, 2)} GB` : `${num(b / 1e6, 1)} MB`;
}

function fmtLatLon(lat, lon) {
  if (lat == null || lon == null) return null;
  return `${num(Math.abs(lat), 3)}° ${lat >= 0 ? 'N' : 'S'}, ${num(lon, 3)}° E`;
}

function joinParts(...parts) {
  const p = parts.filter(v => v != null && v !== '');
  return p.length ? p.join(' · ') : null;
}

// Each row: [label, value from one product record]. null renders as "—"; rows empty on both sides are dropped.
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

  // Side-by-side label metadata, plus any product-specific notes
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

// Both footprints in a local km frame around the OHRC centre (north up). A long reference strip
// (TMC-2 spans ~1000 km) is cropped to a window around the OHRC so the overlap stays readable.
function drawFootprints(o, r, ref) {
  const svg = dom.footprintMap;
  const W = 260, H = 300, PAD = 14, FOOT = 18;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  dom.footprintLegend.innerHTML = '';
  if (!o.corners) {
    svg.innerHTML = '';
    dom.footprintLegend.textContent = 'No footprint in the OHRC label.';
    return;
  }

  const lat0 = o.footprint.center_lat, lon0 = o.footprint.center_lon;
  const kx = MOON_KM_PER_DEG * Math.cos(lat0 * Math.PI / 180);
  const toKm = (c) => [(((c.lon - lon0 + 540) % 360) - 180) * kx, -(c.lat - lat0) * MOON_KM_PER_DEG];
  const order = ['UL', 'UR', 'LR', 'LL'];
  const oPts = order.map(k => toKm(o.corners[k]));
  const rPts = r.corners ? order.map(k => toKm(r.corners[k])) : null;

  const bounds = (list) => list.reduce((b, [x, y]) => [Math.min(b[0], x), Math.min(b[1], y), Math.max(b[2], x), Math.max(b[3], y)],
    [Infinity, Infinity, -Infinity, -Infinity]);
  let [x0, y0, x1, y1] = bounds(rPts ? [...oPts, ...rPts] : oPts);
  const [ox0, oy0, ox1, oy1] = bounds(oPts);
  const reach = 1.25 * Math.max(ox1 - ox0, oy1 - oy0);
  const cropped = x0 < -reach || y0 < -reach || x1 > reach || y1 > reach;
  x0 = Math.max(x0, -reach); y0 = Math.max(y0, -reach);
  x1 = Math.min(x1, reach); y1 = Math.min(y1, reach);

  const scale = Math.min((W - 2 * PAD) / (x1 - x0), (H - 2 * PAD - FOOT) / (y1 - y0));
  const offX = (W - (x1 - x0) * scale) / 2 - x0 * scale;
  const offY = (H - FOOT - (y1 - y0) * scale) / 2 - y0 * scale;
  const pts = (list) => list.map(([x, y]) => `${(x * scale + offX).toFixed(1)},${(y * scale + offY).toFixed(1)}`).join(' ');

  // Scale bar: the largest 1/2/5 × 10^n km that fits in a quarter of the width
  const target = (W - 2 * PAD) / 4 / scale;
  const pow = 10 ** Math.floor(Math.log10(target));
  const barKm = [5, 2, 1].map(m => m * pow).find(v => v <= target) || pow;
  const barPx = barKm * scale;

  svg.innerHTML = `
    <defs><clipPath id="fp-clip"><rect x="0" y="0" width="${W}" height="${H - FOOT}"/></clipPath></defs>
    <g clip-path="url(#fp-clip)">
      ${rPts ? `<polygon class="fp-ref" points="${pts(rPts)}"/>` : ''}
      <polygon class="fp-ohrc" points="${pts(oPts)}"/>
    </g>
    <text class="fp-text" x="${W - PAD}" y="${PAD + 4}" text-anchor="end">N ↑</text>
    <line class="fp-scale" x1="${PAD}" y1="${H - 10}" x2="${(PAD + barPx).toFixed(1)}" y2="${H - 10}"/>
    <text class="fp-text" x="${(PAD + barPx + 6).toFixed(1)}" y="${H - 6}">${barKm} km</text>`;

  const legend = (cls, text) => {
    const span = document.createElement('span');
    span.innerHTML = `<i class="legend-swatch ${cls}"></i>`;
    span.append(text);
    dom.footprintLegend.appendChild(span);
  };
  legend('ohrc', 'OHRC');
  if (rPts) legend('ref', ref);
  const note = (text) => {
    const span = document.createElement('span');
    span.className = 'footprint-note';
    span.textContent = text;
    dom.footprintLegend.appendChild(span);
  };
  if (cropped) note(`${ref} footprint continues beyond this view (${num(r.footprint.length_km, 0)} km long).`);
  if (!rPts) note(`No ${ref} corners available.`);
}

// ---------- Pipeline Flow ----------
// Steps are laid out as a snake: row 0 runs left→right, row 1 right→left, and so on,
// with a U-turn arrow at the end of each row. On narrow screens it collapses to a column.
// Measured on the outer flow box (minus the side room U-turns need), so toggling the
// single-column padding can't feed back into the column count.
function flowColumns() {
  const width = dom.pipelineFlow.clientWidth - 88;
  if (width >= 1080) return 6;
  if (width >= 820) return 4;
  if (width >= 600) return 3;
  if (width >= 440) return 2;
  return 1;
}

function stepState(id) {
  return state.stepStates[id] || 'pending';
}

function stepBox(id) {
  return dom.flowGrid.querySelector(`[data-step-id="${id}"]`);
}

function renderPipeline() {
  const cols = flowColumns();
  state.flowCols = cols;
  dom.pipelineFlow.classList.toggle('flow--single', cols === 1);
  dom.flowGrid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
  dom.flowGrid.innerHTML = '';

  ALL_STEPS.forEach((step, i) => {
    const row = Math.floor(i / cols);
    const pos = i % cols;
    const box = document.createElement('button');
    box.type = 'button';
    box.className = 'step';
    box.dataset.stepId = step.id;
    box.style.gridRow = row + 1;
    box.style.gridColumn = (row % 2 === 0 ? pos : cols - 1 - pos) + 1;
    box.setAttribute('aria-haspopup', 'dialog');
    box.setAttribute('aria-expanded', String(state.openStep === step.id));
    box.innerHTML = `
      <span class="step-meta"><span>${String(step.id).padStart(2, '0')}</span><span class="step-group"></span></span>
      <span class="step-name"></span>
      <span class="step-state"></span>`;
    box.querySelector('.step-group').textContent = step.group;
    box.querySelector('.step-name').textContent = fillTemplate(step.name);
    box.addEventListener('click', (e) => { e.stopPropagation(); togglePopup(step.id); });
    dom.flowGrid.appendChild(box);
    updateStepBox(step.id);
  });

  drawConnectors();
  if (state.openStep) fillPopup(state.openStep);
}

function updateStepBox(id) {
  const box = stepBox(id);
  if (!box) return;
  const s = stepState(id);
  box.classList.remove('step--pending', 'step--running', 'step--done', 'step--error');
  box.classList.add(`step--${s}`);
  box.querySelector('.step-state').textContent = STATE_LABELS[s];
}

function drawConnectors() {
  const boxes = [...dom.flowGrid.children];
  const cols = state.flowCols;
  const base = dom.pipelineFlow.getBoundingClientRect();
  const rel = (el) => {
    const r = el.getBoundingClientRect();
    const left = r.left - base.left, top = r.top - base.top;
    return { left, top, right: left + r.width, bottom: top + r.height, cx: left + r.width / 2, cy: top + r.height / 2 };
  };
  const GAP = 6;     // space between arrow and box edge
  const BEND = 34;   // how far a U-turn swings outside the row

  let paths = '';
  for (let i = 0; i < boxes.length - 1; i++) {
    const a = rel(boxes[i]), b = rel(boxes[i + 1]);
    const row = Math.floor(i / cols);
    let d;
    if (cols === 1) {
      d = `M${a.cx} ${a.bottom + GAP} L${b.cx} ${b.top - GAP}`;
    } else if (row === Math.floor((i + 1) / cols)) {
      d = b.left > a.right
        ? `M${a.right + GAP} ${a.cy} L${b.left - GAP} ${b.cy}`
        : `M${a.left - GAP} ${a.cy} L${b.right + GAP} ${b.cy}`;
    } else if (row % 2 === 0) {
      const x0 = a.right + GAP, x1 = b.right + GAP;
      d = `M${x0} ${a.cy} C${x0 + BEND} ${a.cy} ${x1 + BEND} ${b.cy} ${x1} ${b.cy}`;
    } else {
      const x0 = a.left - GAP, x1 = b.left - GAP;
      d = `M${x0} ${a.cy} C${x0 - BEND} ${a.cy} ${x1 - BEND} ${b.cy} ${x1} ${b.cy}`;
    }
    const done = stepState(ALL_STEPS[i].id) === 'done' && stepState(ALL_STEPS[i + 1].id) !== 'pending';
    paths += `<path d="${d}" class="link${done ? ' link--done' : ''}" marker-end="url(#${done ? 'arrow-done' : 'arrow'})"/>`;
  }

  dom.flowLines.innerHTML = `
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M0 0L10 5L0 10z" class="arrowhead"/>
      </marker>
      <marker id="arrow-done" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M0 0L10 5L0 10z" class="arrowhead arrowhead--done"/>
      </marker>
    </defs>${paths}`;
}

// ---------- Step Popup ----------
function togglePopup(id) {
  if (state.openStep === id) closePopup();
  else openPopup(id);
}

function openPopup(id) {
  if (state.openStep) stepBox(state.openStep)?.setAttribute('aria-expanded', 'false');
  state.openStep = id;
  stepBox(id)?.setAttribute('aria-expanded', 'true');
  fillPopup(id);
  positionPopup(true);
  if (dom.popup.classList.contains('popup--modal')) dom.popup.querySelector('.popup-close').focus();
}

function closePopup() {
  if (!state.openStep) return;
  const box = stepBox(state.openStep);
  box?.setAttribute('aria-expanded', 'false');
  state.openStep = null;
  dom.popup.hidden = true;
  setModal(false);
  return box;
}

// Steps that have produced a preview image open as a large centred window over a dimmed
// page; everything else is a small popup floating above its box.
function setModal(on) {
  dom.popup.classList.toggle('popup--modal', on);
  dom.popup.setAttribute('aria-modal', String(on));
  dom.popupBackdrop.hidden = !on;
  document.body.classList.toggle('modal-open', on);
}

function fillPopup(id) {
  const step = STEP_BY_ID[id];
  const detail = state.stepDetails[id];
  const p = dom.popup;

  p.querySelector('.popup-num').textContent = `Step ${String(id).padStart(2, '0')}`;
  p.querySelector('.popup-title').textContent = fillTemplate(step.name);
  p.querySelector('.popup-algo').textContent = fillTemplate(step.algo);
  const modelEl = p.querySelector('.popup-model');
  modelEl.textContent = step.model ? fillTemplate(step.model) : 'None (classical computer vision)';
  modelEl.classList.toggle('is-none', !step.model);
  p.querySelector('.popup-summary').textContent = fillTemplate(step.desc);

  // Output from the current / last run, if this step has reported anything
  p.querySelector('.popup-run').hidden = !detail?.detail;
  p.querySelector('.popup-detail').textContent = detail?.detail || '';

  const figure = p.querySelector('.popup-figure');
  const img = p.querySelector('.popup-image');
  figure.hidden = !detail?.image;
  if (detail?.image) {
    // A strip much taller or wider than it is across is shown unscaled and scrolls.
    const markStrip = () => figure.classList.toggle('popup-figure--strip',
      Math.max(img.naturalWidth, img.naturalHeight) > 2.2 * Math.min(img.naturalWidth, img.naturalHeight));
    img.onload = markStrip;
    if (img.getAttribute('src') !== detail.image) img.src = detail.image;
    else if (img.complete) markStrip();
    img.alt = `${fillTemplate(step.name)} output`;
  } else {
    img.removeAttribute('src');
  }
  setModal(!!detail?.image);

  p.hidden = false;
  positionPopup();
}

// Anchored mode: float the popup above its box. It drops below only when the page itself has
// no room above (first row near the top of the document). `reveal` scrolls it into view when a
// user opens it. Modal mode is centred by CSS and needs no positioning.
function positionPopup(reveal = false) {
  if (!state.openStep) return;
  const p = dom.popup;
  if (p.classList.contains('popup--modal')) {
    p.style.left = p.style.top = '';
    return;
  }
  const box = stepBox(state.openStep);
  if (!box) return;
  const base = dom.pipelineFlow.getBoundingClientRect();
  const r = box.getBoundingClientRect();
  const pw = p.offsetWidth, ph = p.offsetHeight;
  const OFFSET = 12, MARGIN = 8;

  const boxCx = r.left - base.left + r.width / 2;
  const left = Math.max(0, Math.min(boxCx - pw / 2, base.width - pw));
  const above = r.top + window.scrollY - ph - OFFSET >= MARGIN;

  p.style.left = `${left}px`;
  p.style.top = above
    ? `${r.top - base.top - ph - OFFSET}px`
    : `${r.bottom - base.top + OFFSET}px`;
  p.classList.toggle('popup--below', !above);
  p.querySelector('.popup-arrow').style.left = `${Math.max(14, Math.min(boxCx - left, pw - 14))}px`;

  if (reveal) {
    const pr = p.getBoundingClientRect();
    let dy = 0;
    if (pr.top < MARGIN) dy = pr.top - MARGIN;
    else if (pr.bottom > window.innerHeight - MARGIN) dy = pr.bottom - window.innerHeight + MARGIN;
    if (dy) window.scrollBy({ top: dy, behavior: 'smooth' });
  }
}

// ---------- Progress ----------
function updateProgress(progress) {
  state.progress = progress;
  dom.progressFill.style.width = `${progress}%`;
  dom.progressLabel.textContent = `${Math.round(progress)}%`;
}

// ---------- SSE Pipeline Run ----------
function handleRunPipeline() {
  if (!state.connected || state.running) return;

  state.running = true;
  state.stepStates = {};
  state.stepDetails = {};
  state.results = null;
  state.resultImages = {};

  // Reset all steps to pending
  ALL_STEPS.forEach(s => { state.stepStates[s.id] = 'pending'; });

  dom.runProgress.hidden = false;
  dom.runStatus.textContent = 'Starting…';
  dom.resultsSection.hidden = true;
  dom.runBtn.disabled = true;
  dom.runBtn.textContent = 'Running…';
  dom.pipelineSelect.disabled = true;
  dom.pairSelect.disabled = true;
  setRunControlsDisabled(true);
  state.lastRun = { pipeline: state.pipeline, pair_id: dom.pairSelect.value };
  state.runError = null;
  document.querySelector('.error-banner')?.remove();
  updateProgress(0);
  renderPipeline();
  dom.pipelineFlow.scrollIntoView({ behavior: 'smooth', block: 'start' });
  startConnectionHealthCheck();

  const config = {
    pipeline: state.pipeline,
    pair_id: dom.pairSelect.value,
    ecc_mode: dom.eccMode.value,
    matcher: dom.matcherSelect.value,
    shadow_mask: dom.optShadow.checked,
    tile_rescale: dom.optRescale.checked,
    tile_polarity: dom.optPolarity.checked,
    crater_matching: dom.optCraters.checked,
    auto_retry: dom.optRetry.checked,
    ...sunOverrides(),
  };

  const params = new URLSearchParams(config);
  const url = state.colabUrl.replace(/\/+$/, '') + '/api/run?' + params.toString();

  // Use fetch instead of EventSource to include custom headers (for ngrok)
  startSSE(url);
}

async function startSSE(url) {
  try {
    const resp = await fetch(url, {
      headers: { 'ngrok-skip-browser-warning': 'true' }
    });

    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const event = JSON.parse(line.slice(6));
            handlePipelineEvent(event);
          } catch (e) {
            console.warn('SSE parse error:', e, line);
          }
        }
      }
    }

    // Pipeline finished
    if (state.runError) onPipelineError(state.runError);
    else onPipelineComplete();
  } catch (err) {
    console.error('SSE error:', err);
    const msg = err.message.includes('Failed to fetch') || err.message.includes('NetworkError')
      ? 'Backend server disconnected. Check if the server is still running.'
      : err.message;
    onPipelineError(msg);
  }
}

function handlePipelineEvent(event) {
  const { step, status, progress, detail, image, metrics, images } = event;

  // Step 0 is the server's pipeline-level failure (exception in a stage)
  if (step === 0 && status === 'error') {
    state.runError = detail || 'Pipeline failed on the server';
    return;
  }

  if (step && status) {
    state.stepStates[step] = status;
    state.currentStep = step;

    if (!state.stepDetails[step]) state.stepDetails[step] = {};
    if (detail) state.stepDetails[step].detail = detail;
    if (image) state.stepDetails[step].image = image;
    if (metrics) state.stepDetails[step].metrics = metrics;

    const s = STEP_BY_ID[step];
    dom.runStatus.textContent = `Step ${step}/${TOTAL_STEPS} · ${s ? fillTemplate(s.name) : event.name}` +
      (detail ? ` — ${detail}` : status === 'running' ? ' — running' : '');

    updateStepBox(step);
    drawConnectors();
    if (state.openStep === step) fillPopup(step);
    if (status === 'running') stepBox(step)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  if (progress !== undefined) {
    updateProgress(progress);
  }

  // Store result images
  if (images) {
    Object.assign(state.resultImages, images);
  }

  // Store final results
  if (metrics && status === 'done' && step === TOTAL_STEPS) {
    state.results = metrics;
  }
}

function setRunControlsDisabled(on) {
  [dom.eccMode, dom.matcherSelect, dom.optShadow, dom.optRescale, dom.optPolarity, dom.optCraters,
   dom.optRetry, dom.sunElev, dom.sunAzim, dom.sunConv].forEach(el => { el.disabled = on; });
}

function unlockControls() {
  dom.runBtn.disabled = !(state.pairsByPipeline[state.pipeline] || []).length;
  dom.pipelineSelect.disabled = false;
  dom.pairSelect.disabled = false;
  setRunControlsDisabled(false);
}

function onPipelineComplete() {
  state.running = false;
  unlockControls();
  dom.runBtn.textContent = 'Run pipeline';
  updateProgress(100);
  dom.runStatus.textContent = 'Finished';

  // Show results if available
  if (state.results) {
    showResults(state.results);
  }
}

function onPipelineError(errorMsg) {
  state.running = false;
  unlockControls();
  dom.runBtn.textContent = 'Retry pipeline';

  // Mark the current step as error
  if (state.currentStep) {
    state.stepStates[state.currentStep] = 'error';
    if (!state.stepDetails[state.currentStep]) state.stepDetails[state.currentStep] = {};
    state.stepDetails[state.currentStep].detail = errorMsg || 'Connection lost';
    updateStepBox(state.currentStep);
    drawConnectors();
    if (state.openStep === state.currentStep) fillPopup(state.currentStep);
  }
  dom.runStatus.textContent = `Failed at step ${state.currentStep || '?'}`;

  showErrorBanner(errorMsg || 'Connection to backend server lost. Please check if the server is running and try again.');
}

function showErrorBanner(message) {
  document.querySelector('.error-banner')?.remove();

  const banner = document.createElement('div');
  banner.className = 'error-banner';
  banner.setAttribute('role', 'alert');
  banner.innerHTML = `
    <div class="error-banner-content">
      <span class="error-banner-text"></span>
      <button type="button" class="btn btn-secondary">Dismiss</button>
    </div>
  `;
  banner.querySelector('.error-banner-text').textContent = message;
  banner.querySelector('button').addEventListener('click', () => banner.remove());
  document.body.prepend(banner);

  // Auto-dismiss after 15 seconds
  setTimeout(() => banner.remove(), 15000);
}

function startConnectionHealthCheck() {
  // Check connection every 10 seconds while pipeline is running
  if (state._healthCheckInterval) clearInterval(state._healthCheckInterval);
  state._healthCheckInterval = setInterval(async () => {
    if (!state.running) {
      clearInterval(state._healthCheckInterval);
      return;
    }
    try {
      const resp = await fetch(state.colabUrl.replace(/\/+$/, '') + '/api/connect', {
        headers: { 'ngrok-skip-browser-warning': 'true' },
        signal: AbortSignal.timeout(5000)
      });
      if (!resp.ok) throw new Error('Server returned error');
    } catch {
      setConnectionStatus('disconnected', 'Server disconnected');
    }
  }, 10000);
}

// ---------- Results ----------
function showResults(metrics) {
  dom.resultsSection.hidden = false;

  const cards = [
    {
      label: 'In-sample RMSE',
      value: metrics.rmse_px_in_sample != null ? `${metrics.rmse_px_in_sample.toFixed(3)} px` : '—',
      unit: metrics.rmse_m_in_sample != null ? `${metrics.rmse_m_in_sample.toFixed(2)} m on the ground` : '',
      quality: metrics.rmse_px_in_sample < 5 ? 'good' : metrics.rmse_px_in_sample < 15 ? 'warn' : 'bad'
    },
    heldoutCard(metrics),
    {
      label: 'NMI, aligned',
      value: metrics.nmi_aligned != null ? metrics.nmi_aligned.toFixed(4) : '—',
      unit: metrics.nmi_misaligned_baseline != null ? `misaligned baseline ${metrics.nmi_misaligned_baseline.toFixed(4)}` : '',
      quality: metrics.nmi_aligned > metrics.nmi_misaligned_baseline * 1.15 ? 'good' : 'warn'
    },
    {
      label: 'Spatial coverage',
      value: metrics.spatial_coverage_pct != null ? `${metrics.spatial_coverage_pct.toFixed(1)}%` : '—',
      unit: `of the ${(PIPELINE_PROFILES[state.lastRun?.pipeline] || currentProfile()).ref} frame`,
      quality: metrics.spatial_coverage_pct > 10 ? 'good' : 'warn'
    },
    {
      label: 'Inliers',
      value: metrics.n_inliers != null ? metrics.n_inliers.toString() : '—',
      unit: `of ${metrics.n_correspondences || '?'} correspondences`,
      quality: metrics.n_inliers >= 15 ? 'good' : 'bad'
    },
    {
      label: 'Model',
      value: metrics.model || '—',
      unit: metrics.trusted ? 'Trusted: all checks pass' : 'Low confidence: see trust checks',
      quality: metrics.trusted ? 'good' : 'warn'
    },
  ];
  if (metrics.difficulty) {
    const lvl = metrics.difficulty.level;
    cards.push({
      label: 'Pair difficulty',
      value: lvl.toUpperCase(),
      unit: `score ${metrics.difficulty.score} · lighting, scale, texture`,
      quality: lvl === 'easy' ? 'good' : lvl === 'medium' ? 'warn' : 'bad'
    });
  }

  dom.metricsGrid.innerHTML = cards.map(c => `
    <div class="metric-card ${c.quality}">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value">${c.value}</div>
      <div class="metric-unit">${c.unit}</div>
    </div>
  `).join('');

  renderResultNotes(metrics);
  setupImageViewer();
  $$('.btn-download').forEach(btn => { btn.disabled = false; });
  dom.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Spatial held-out is the honest figure; older metrics files only have the random split.
function heldoutCard(m) {
  const spatial = m.rmse_px_heldout_spatial;
  const px = spatial ?? m.rmse_px_heldout;
  const metres = spatial != null ? m.rmse_m_heldout_spatial : m.rmse_m_heldout;
  return {
    label: spatial != null ? 'Held-out RMSE (spatial)' : 'Held-out RMSE',
    value: px != null ? `${px.toFixed(3)} px` : '—',
    unit: px != null ? `${metres.toFixed(2)} m on the ground` : 'no held-out fit succeeded',
    quality: px == null ? 'bad' : px <= 5 ? 'good' : 'warn'
  };
}

function noteBlock(title, rows) {
  const block = document.createElement('div');
  block.className = 'note-block';
  const h = document.createElement('div');
  h.className = 'note-title';
  h.textContent = title;
  const ul = document.createElement('ul');
  // Pass/fail marks get a column only when the list has any; plain info lists drop it.
  ul.className = rows.some(r => r.mark !== 'info') ? 'note-list' : 'note-list note-list--plain';
  rows.forEach(({ mark, text, value }) => {
    const li = document.createElement('li');
    const m = document.createElement('span');
    m.className = `note-mark ${mark}`;
    m.textContent = mark === 'pass' ? '✓' : mark === 'fail' ? '✗' : '';
    const t = document.createElement('span');
    t.textContent = text;
    li.append(m, t);
    if (value != null) {
      const v = document.createElement('span');
      v.className = 'note-value';
      v.textContent = value;
      li.append(v);
    }
    ul.append(li);
  });
  block.append(h, ul);
  return block;
}

function renderResultNotes(m) {
  dom.resultNotes.innerHTML = '';
  if (m.trust_checks) {
    dom.resultNotes.append(noteBlock(m.trusted ? 'Trust checks: all pass' : 'Trust checks',
      m.trust_checks.map(c => ({ mark: c.pass ? 'pass' : 'fail', text: `${c.name} (${c.limit})`, value: c.value ?? 'n/a' }))));
  }
  if (m.difficulty) {
    dom.resultNotes.append(noteBlock(`Pair difficulty: ${m.difficulty.level} (score ${m.difficulty.score})`,
      m.difficulty.factors.map(f => ({ mark: 'info', text: f }))));
  }
  const run = [];
  if (m.ecc_mode) {
    const e = m.ecc_info || {};
    let v = m.ecc_applied ? 'applied' : 'not applied';
    if (e.mode === 'gated' && e.score_before != null) v += ` · score ${e.score_before.toFixed(3)} → ${(e.score_after ?? e.score_before).toFixed(3)}`;
    run.push({ mark: 'info', text: `ECC (${m.ecc_mode})`, value: v });
  }
  if (m.rmse_px_in_sample_pre_ecc != null) run.push({ mark: 'info', text: 'In-sample RMSE before ECC', value: `${m.rmse_px_in_sample_pre_ecc.toFixed(3)} px` });
  if (m.matcher) run.push({ mark: 'info', text: 'Coarse matcher', value: m.matcher });
  if (m.tiles_rescued) run.push({ mark: 'info', text: 'Tiles rescued', value: Object.entries(m.rescued_by || {}).map(([k, v]) => `${v} ${k}`).join(', ') });
  if (m.crater_stats) run.push({ mark: 'info', text: 'Crater matches', value: `${m.crater_stats.matched} of ${m.crater_stats.paired} paired` });
  if (m.inliers_by_source && Object.keys(m.inliers_by_source).length > 1) {
    run.push({ mark: 'info', text: 'Inliers by source', value: Object.entries(m.inliers_by_source).map(([k, v]) => `${v} ${k}`).join(', ') });
  }
  if ((m.attempts || []).length > 1) {
    m.attempts.forEach(a => {
      run.push({
        mark: a.error ? 'fail' : a.trusted ? 'pass' : 'info',
        text: `Attempt ${a.attempt}: ${a.label}${a.attempt === m.selected_attempt ? ' (used)' : ''}`,
        value: a.error ? 'failed' : a.rmse_px_heldout_spatial != null ? `held-out ${a.rmse_px_heldout_spatial.toFixed(2)} px` : `${a.n_inliers} inliers`
      });
    });
  }
  if (run.length) dom.resultNotes.append(noteBlock('This run', run));
}

function showTab(tabName) {
  $$('.tab-btn').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tabName);
    t.setAttribute('aria-selected', String(t.dataset.tab === tabName));
  });
  const imgData = state.resultImages[tabName];
  if (imgData) {
    dom.viewerImage.src = imgData;
    dom.viewerImage.hidden = false;
    dom.viewerCanvas.querySelector('.viewer-placeholder')?.remove();
  } else {
    dom.viewerImage.hidden = true;
    if (!dom.viewerCanvas.querySelector('.viewer-placeholder')) {
      const ph = document.createElement('div');
      ph.className = 'viewer-placeholder';
      ph.textContent = 'No image available for this view';
      dom.viewerCanvas.appendChild(ph);
    }
  }
}

function setupImageViewer() {
  const tabs = [...$$('.tab-btn')].map(t => t.dataset.tab);
  showTab(tabs.find(t => state.resultImages[t]) || 'overlay');
}

// The backend currently writes only metrics.json (plus the .npy homographies) to outputs/.
async function handleDownload(fileType) {
  const run = state.lastRun;
  if (!run) return;
  const profile = PIPELINE_PROFILES[run.pipeline] || currentProfile();
  const names = { metrics: 'metrics.json' };
  try {
    const params = new URLSearchParams({ pipeline: run.pipeline, pair_id: run.pair_id });
    const resp = await fetch(
      state.colabUrl.replace(/\/+$/, '') + `/api/results/download/${fileType}?${params}`,
      { headers: { 'ngrok-skip-browser-warning': 'true' } }
    );
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.error || `Download failed (${resp.status})`);
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${run.pair_id}_${profile.version}_${names[fileType] || fileType}`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('Download error:', err);
    showErrorBanner(`${fileType}: ${err.message}`);
  }
}

// ---------- Event Listeners ----------
dom.connectBtn.addEventListener('click', handleConnect);
dom.urlInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') handleConnect(); });
dom.runBtn.addEventListener('click', handleRunPipeline);
dom.pipelineSelect.addEventListener('change', handlePipelineChange);
dom.pairSelect.addEventListener('change', () => { loadPairInfo(); loadDataset(); });
$$('.tab-btn').forEach(btn => btn.addEventListener('click', () => showTab(btn.dataset.tab)));
$$('.btn-download').forEach(btn => btn.addEventListener('click', () => handleDownload(btn.dataset.file)));

dom.popup.addEventListener('click', (e) => e.stopPropagation());
dom.popup.querySelector('.popup-close').addEventListener('click', () => closePopup()?.focus());
document.addEventListener('click', () => closePopup());
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && state.openStep) closePopup()?.focus();
});

// Re-flow the snake when the width changes the column count; otherwise just redraw arrows.
new ResizeObserver(() => {
  if (flowColumns() !== state.flowCols) renderPipeline();
  else { drawConnectors(); positionPopup(); }
}).observe(dom.pipelineFlow);

// ---------- Initialize ----------
renderPipeline();
