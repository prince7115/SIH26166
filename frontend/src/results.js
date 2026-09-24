import { fetchBlob } from './api.js';
import { showErrorBanner } from './banner.js';
import { PIPELINE_PROFILES, currentProfile } from './content/pipelines.js';
import { $$, dom } from './dom.js';
import { state } from './state.js';

export function showResults(metrics) {
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

// Prefer the spatial held-out RMSE; older metrics files only have the random split.
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
  // The pass/fail column is shown only when some row has a mark.
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

export function showTab(tabName) {
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

export async function handleDownload(fileType) {
  const run = state.lastRun;
  if (!run) return;
  const profile = PIPELINE_PROFILES[run.pipeline] || currentProfile();
  try {
    const params = new URLSearchParams({ pipeline: run.pipeline, pair_id: run.pair_id });
    const blob = await fetchBlob(`/api/results/download/${fileType}?${params}`);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${run.pair_id}_${profile.version}_${fileType === 'metrics' ? 'metrics.json' : fileType}`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('Download error:', err);
    showErrorBanner(`${fileType}: ${err.message}`);
  }
}
