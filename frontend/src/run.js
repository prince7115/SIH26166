import { streamEvents } from './api.js';
import { clearErrorBanner, showErrorBanner } from './banner.js';
import { startConnectionHealthCheck } from './connection.js';
import { ALL_STEPS, STEP_BY_ID, TOTAL_STEPS } from './content/steps.js';
import { fillTemplate } from './content/pipelines.js';
import { dom, stepBox } from './dom.js';
import { drawConnectors, renderPipeline, updateStepBox } from './flow.js';
import { fillPopup } from './popup.js';
import { showResults } from './results.js';
import { lockControls, runParameters, unlockControls } from './runConfig.js';
import { state } from './state.js';

function updateProgress(progress) {
  dom.progressFill.style.width = `${progress}%`;
  dom.progressLabel.textContent = `${Math.round(progress)}%`;
}

export async function handleRunPipeline() {
  if (!state.connected || state.running) return;

  state.running = true;
  state.stepStates = Object.fromEntries(ALL_STEPS.map(s => [s.id, 'pending']));
  state.stepDetails = {};
  state.results = null;
  state.resultImages = {};

  dom.runProgress.hidden = false;
  dom.runStatus.textContent = 'Starting…';
  dom.resultsSection.hidden = true;
  lockControls();
  dom.runBtn.textContent = 'Running…';
  state.lastRun = { pipeline: state.pipeline, pair_id: dom.pairSelect.value };
  state.runError = null;
  clearErrorBanner();
  updateProgress(0);
  renderPipeline();
  dom.pipelineFlow.scrollIntoView({ behavior: 'smooth', block: 'start' });
  startConnectionHealthCheck();

  try {
    await streamEvents(`/api/run?${new URLSearchParams(runParameters())}`, handlePipelineEvent);
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

  // Step 0 is a server-side failure of the whole run.
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

    const s = STEP_BY_ID[step];
    dom.runStatus.textContent = `Step ${step}/${TOTAL_STEPS} · ${s ? fillTemplate(s.name) : event.name}` +
      (detail ? ` — ${detail}` : status === 'running' ? ' — running' : '');

    updateStepBox(step);
    drawConnectors();
    if (state.openStep === step) fillPopup(step);
    if (status === 'running') stepBox(step)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  if (progress !== undefined) updateProgress(progress);
  if (images) Object.assign(state.resultImages, images);
  if (metrics && status === 'done' && step === TOTAL_STEPS) state.results = metrics;
}

function onPipelineComplete() {
  state.running = false;
  unlockControls();
  dom.runBtn.textContent = 'Run pipeline';
  updateProgress(100);
  dom.runStatus.textContent = 'Finished';
  if (state.results) showResults(state.results);
}

function onPipelineError(errorMsg) {
  state.running = false;
  unlockControls();
  dom.runBtn.textContent = 'Retry pipeline';

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
