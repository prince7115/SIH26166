import { dom } from './dom.js';
import { num } from './format.js';

const MOON_KM_PER_DEG = 2 * Math.PI * 1737.4 / 360;

// Both footprints in a local km frame around the OHRC centre, north up. A long reference strip
// (TMC-2 spans ~1000 km) is cropped to a window around the OHRC so the overlap stays readable.
export function drawFootprints(o, r, ref) {
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

  // Scale bar: the largest 1/2/5 x 10^n km that fits in a quarter of the width.
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
