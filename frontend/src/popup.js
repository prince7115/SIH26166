import { STEP_BY_ID } from './content/steps.js';
import { fillTemplate } from './content/pipelines.js';
import { dom, stepBox } from './dom.js';
import { state } from './state.js';

export function togglePopup(id) {
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

// Returns the step box that had the popup, so the caller can move focus back to it.
export function closePopup() {
  if (!state.openStep) return;
  const box = stepBox(state.openStep);
  box?.setAttribute('aria-expanded', 'false');
  state.openStep = null;
  dom.popup.hidden = true;
  setModal(false);
  return box;
}

// A step with a preview image opens as a large centred window over a dimmed page;
// any other step opens as a small popup above its box.
function setModal(on) {
  dom.popup.classList.toggle('popup--modal', on);
  dom.popup.setAttribute('aria-modal', String(on));
  dom.popupBackdrop.hidden = !on;
  document.body.classList.toggle('modal-open', on);
}

export function fillPopup(id) {
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

  p.querySelector('.popup-run').hidden = !detail?.detail;
  p.querySelector('.popup-detail').textContent = detail?.detail || '';

  const figure = p.querySelector('.popup-figure');
  const img = p.querySelector('.popup-image');
  figure.hidden = !detail?.image;
  if (detail?.image) {
    // A strip much longer than it is wide is shown at full size and scrolls.
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

// The small popup floats above its box, or below it when the page has no room above.
// reveal scrolls it into view when the user opens it. The modal window is centred by CSS.
export function positionPopup(reveal = false) {
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
