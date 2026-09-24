import { ALL_STEPS, STATE_LABELS } from './content/steps.js';
import { fillTemplate } from './content/pipelines.js';
import { dom, stepBox } from './dom.js';
import { fillPopup, togglePopup } from './popup.js';
import { state } from './state.js';

// Steps snake across the page: row 0 left to right, row 1 right to left, and so on, with a
// U-turn arrow at the end of each row; narrow screens get a single column. The width is taken
// from the outer flow box minus the room U-turns need, so the single-column padding cannot
// feed back into the column count.
export function flowColumns() {
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

export function renderPipeline() {
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

export function updateStepBox(id) {
  const box = stepBox(id);
  if (!box) return;
  const s = stepState(id);
  box.classList.remove('step--pending', 'step--running', 'step--done', 'step--error');
  box.classList.add(`step--${s}`);
  box.querySelector('.step-state').textContent = STATE_LABELS[s];
}

export function drawConnectors() {
  const boxes = [...dom.flowGrid.children];
  const cols = state.flowCols;
  const base = dom.pipelineFlow.getBoundingClientRect();
  const rel = (el) => {
    const r = el.getBoundingClientRect();
    const left = r.left - base.left, top = r.top - base.top;
    return { left, top, right: left + r.width, bottom: top + r.height, cx: left + r.width / 2, cy: top + r.height / 2 };
  };
  const GAP = 6;     // between an arrow and the box edge
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
