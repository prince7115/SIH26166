import { handleConnect } from './connection.js';
import { $$, dom } from './dom.js';
import { drawConnectors, flowColumns, renderPipeline } from './flow.js';
import { closePopup, positionPopup } from './popup.js';
import { handleDownload, showTab } from './results.js';
import { handleRunPipeline } from './run.js';
import { handlePipelineChange, loadPairInfo } from './runConfig.js';
import { loadDataset } from './dataset.js';
import { state } from './state.js';

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

// Re-flow the steps when the width changes the column count; otherwise only redraw the arrows.
new ResizeObserver(() => {
  if (flowColumns() !== state.flowCols) renderPipeline();
  else { drawConnectors(); positionPopup(); }
}).observe(dom.pipelineFlow);

renderPipeline();
