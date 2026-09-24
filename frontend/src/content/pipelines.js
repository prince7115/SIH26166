import { state } from '../state.js';

// Website copy per sensor pair. Keys, labels and versions match backend/trinetra/sensors.
export const PIPELINE_PROFILES = {
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

export function currentProfile() {
  return PIPELINE_PROFILES[state.pipeline] || PIPELINE_PROFILES.ohrc_nac;
}

export function fillTemplate(str) {
  const p = currentProfile();
  return str.replace(/\{(\w+)\}/g, (m, key) => key === 'REF' ? p.ref : (p.text[key] ?? m));
}
