import { apiFetch, ping } from './api.js';
import { dom } from './dom.js';
import { handlePipelineChange } from './runConfig.js';
import { state } from './state.js';

function setConnectionStatus(status, text) {
  const el = dom.connStatus;
  el.className = `status status-${status}`;
  el.querySelector('.status-text').textContent = text;
}

export async function handleConnect() {
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

    try {
      const { pipelines, default: fallback } = await apiFetch('/api/pipelines');
      state.pairsByPipeline = {};
      dom.pipelineSelect.innerHTML = '';
      pipelines.forEach(p => {
        state.pairsByPipeline[p.id] = p.pairs || [];
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = p.pairs && p.pairs.length ? p.label : `${p.label} (no data)`;
        dom.pipelineSelect.appendChild(opt);
      });
      dom.pipelineSelect.value = state.pairsByPipeline[state.pipeline] ? state.pipeline : fallback;
    } catch { /* keep the page's built-in pipeline options */ }
    handlePipelineChange();

  } catch (err) {
    state.connected = false;
    setConnectionStatus('disconnected', 'Connection failed');
    dom.connectBtn.textContent = 'Retry';
    dom.connectBtn.disabled = false;
    console.error('Connection error:', err);
  }
}

// While a run is in progress, check every 10 s that the server still answers.
export function startConnectionHealthCheck() {
  if (state.healthCheck) clearInterval(state.healthCheck);
  state.healthCheck = setInterval(async () => {
    if (!state.running) {
      clearInterval(state.healthCheck);
      return;
    }
    try {
      await ping(5000);
    } catch {
      setConnectionStatus('disconnected', 'Server disconnected');
    }
  }, 10000);
}
