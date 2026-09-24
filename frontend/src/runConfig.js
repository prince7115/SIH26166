import { apiFetch } from './api.js';
import { currentProfile } from './content/pipelines.js';
import { loadDataset } from './dataset.js';
import { $$, dom } from './dom.js';
import { renderPipeline } from './flow.js';
import { fmtSun } from './format.js';
import { state } from './state.js';

export function handlePipelineChange() {
  state.pipeline = dom.pipelineSelect.value;
  const profile = currentProfile();

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

// Sun fields are pre-filled from sun.json or the labels and can be edited.
function setSunField(input, value) {
  input.value = value ?? '';
  input.dataset.prefill = input.value;
}

export async function loadPairInfo() {
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

// Only fields the user changed are sent; the others keep the file/label value on the server.
function sunOverrides() {
  const out = {};
  if (dom.sunElev.value !== dom.sunElev.dataset.prefill && dom.sunElev.value !== '') out.ref_sun_elev = dom.sunElev.value;
  if (dom.sunAzim.value !== dom.sunAzim.dataset.prefill && dom.sunAzim.value !== '') out.ref_sun_azim = dom.sunAzim.value;
  if (dom.sunConv.value !== dom.sunConv.dataset.prefill && dom.sunConv.value !== '') out.ref_sun_conv = dom.sunConv.value;
  return out;
}

export function runParameters() {
  return {
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
}

function setRunControlsDisabled(on) {
  [dom.eccMode, dom.matcherSelect, dom.optShadow, dom.optRescale, dom.optPolarity, dom.optCraters,
   dom.optRetry, dom.sunElev, dom.sunAzim, dom.sunConv].forEach(el => { el.disabled = on; });
}

export function lockControls() {
  dom.runBtn.disabled = true;
  dom.pipelineSelect.disabled = true;
  dom.pairSelect.disabled = true;
  setRunControlsDisabled(true);
}

export function unlockControls() {
  dom.runBtn.disabled = !(state.pairsByPipeline[state.pipeline] || []).length;
  dom.pipelineSelect.disabled = false;
  dom.pairSelect.disabled = false;
  setRunControlsDisabled(false);
}
