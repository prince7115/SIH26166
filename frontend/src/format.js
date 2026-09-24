export const num = (v, digits) => (v == null ? null : Number(v).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }));

const asUtc = (t) => Date.parse(/Z$/.test(t) ? t : `${t}Z`);

export function fmtTime(start, stop) {
  if (!start) return null;
  const t = start.replace('T', ' ').replace(/Z$/, '').slice(0, 19);
  const secs = stop ? (asUtc(stop) - asUtc(start)) / 1000 : NaN;
  return Number.isFinite(secs) ? `${t} UTC (${num(secs, 1)} s)` : `${t} UTC`;
}

export function fmtBytes(b) {
  if (b == null) return null;
  return b >= 1e9 ? `${num(b / 1e9, 2)} GB` : `${num(b / 1e6, 1)} MB`;
}

export function fmtLatLon(lat, lon) {
  if (lat == null || lon == null) return null;
  return `${num(Math.abs(lat), 3)}° ${lat >= 0 ? 'N' : 'S'}, ${num(lon, 3)}° E`;
}

export function fmtSun(sun) {
  if (!sun) return 'not available';
  const parts = [];
  if (sun.elev_deg != null) parts.push(`elevation ${sun.elev_deg.toFixed(2)}°`);
  if (sun.azim_deg != null) parts.push(`azimuth ${sun.azim_deg.toFixed(2)}° (${sun.azim_convention || 'unknown'})`);
  return `${parts.join(', ')} from ${sun.source}`;
}

export function joinParts(...parts) {
  const p = parts.filter(v => v != null && v !== '');
  return p.length ? p.join(' · ') : null;
}
