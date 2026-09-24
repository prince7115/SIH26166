// What each backend step runs, shown in the step popups. {PLACEHOLDERS} are filled per
// sensor pair from pipelines.js. Keep these in sync with backend/trinetra when a step changes.
const PIPELINE_GROUPS = [
  {
    label: 'Data loading',
    steps: [
      { id: 1, name: 'Loading OHRC Image',
        algo: 'PDS4 label parsing (corner lat/lon, GSD) + raster read',
        model: null,
        desc: 'Reads the Chandrayaan-2 OHRC PDS4 raster and its XML label, and takes the four corner coordinates and ground sample distance from the label.' },
      { id: 2, name: 'Loading {REF} Reference',
        algo: '{REF_LOAD_ALGO}',
        model: null,
        desc: '{REF_LOAD_DESC}' },
    ]
  },
  {
    label: 'Preprocessing',
    steps: [
      { id: 3, name: 'Overlap Detection & Crop',
        algo: 'Convex polygon intersection of the lat/lon footprints, 150 m padding',
        model: null,
        desc: 'Intersects the two ground footprints and crops both images to the shared region. 16-bit products are stretched to 8-bit on the crop only.' },
      { id: 4, name: 'Resolution Scheme',
        algo: '{FINE_ALGO}',
        model: null,
        desc: 'Picks two working resolutions: a coarse level for global alignment, grown by 1.25× until both images fit in 2 MP, and a fine level for tie points. {FINE_DESC}' },
      { id: 5, name: 'Appearance Preprocessing',
        algo: 'Percentile stretch (1–99%) → CLAHE (clip 2.0 OHRC, 4.0 {REF}) → histogram matching',
        model: null,
        desc: 'Makes two different cameras look alike: normalises brightness, boosts local contrast, then matches the reference histogram to the OHRC.' },
    ]
  },
  {
    label: 'Orientation',
    steps: [
      { id: 6, name: 'Orientation Search (ZNCC)',
        algo: 'ZNCC template matching on gradient magnitude, 4 flips × scales 0.80–1.24, phase-randomised null',
        model: null,
        desc: 'Checks whether both images show the same ground and which flip lines them up. A match must beat the phase-randomised noise floor by 2.5× to count.' },
      { id: 7, name: 'Residual Rotation Estimate',
        algo: 'Log-polar FFT magnitude + phase correlation (Fourier–Mellin)',
        model: null,
        desc: 'Rotates the OHRC by a known 7° and recovers it, which calibrates the sign convention of the rotation estimator for this pair.' },
      { id: 8, name: 'SuperPoint + SuperGlue',
        algo: 'Keypoint detection + attentional graph matching at 1024 px, score threshold 0.15',
        model: 'SuperPoint + SuperGlue (magic-leap-community/superglue_outdoor)',
        desc: 'Runs the learned matcher on the coarse, contrast-enhanced pair (GPU if available). This first pass is a diagnostic view; the fit starts from the geometric prior in step 9.' },
    ]
  },
  {
    label: 'Coarse fit',
    steps: [
      { id: 9, name: 'Geometric Prior',
        algo: 'Least-squares pixel → lat/lon affine per product, composed into OHRC → {REF}',
        model: null,
        desc: 'Builds a first OHRC-to-{REF} transform purely from the two corner models. Exact at the corners, linear in between. Reports scale and whether the pair is mirrored.' },
      { id: 10, name: 'Dense Shift Field',
        algo: 'Band-wise ZNCC on gradient magnitude (8 bands, null margin ≥ 2) + polynomial fit',
        model: null,
        desc: 'Measures how far the prior drifts along the strip, one horizontal band at a time, and fits a 1st/2nd-order polynomial to remove pushbroom geometry error.' },
      { id: 11, name: 'Coarse Refinement',
        algo: 'SuperGlue on the prior-warped pair → robust fit (similarity / affine / homography)',
        model: 'SuperPoint + SuperGlue (magic-leap-community/superglue_outdoor)',
        desc: 'Matches the pre-warped images again and applies the correction only if it has at least 25 inliers and moves the centre by no more than 250 px.' },
    ]
  },
  {
    label: 'Fine matching',
    steps: [
      { id: 12, name: 'Tiled Matching at Native Resolution',
        algo: '{TILE_ALGO}',
        model: null,
        desc: 'Cuts the OHRC into overlapping tiles at fine GSD, pre-warps each with the local prior and matches it by dense correlation with sub-pixel phase correlation. Tiles whose points disagree by more than 6 px are dropped. Optional (Advanced): failed tiles can be rescued with shadow suppression, a coarser scale or an intensity match, and crater-anchored matches can be added.' },
      { id: 13, name: 'Final Model (RANSAC)',
        algo: 'MAD-based threshold → RANSAC similarity → affine → MAGSAC homography',
        model: null,
        desc: 'Fits the final transform on all fine correspondences. It moves to a richer model only if inliers do not drop; a homography needs 40+ inliers and 10% coverage. Trusted if ≥ 15 inliers cover ≥ 5% of the frame.' },
    ]
  },
  {
    label: 'Output',
    steps: [
      { id: 14, name: 'ECC Intensity Refinement',
        algo: 'OpenCV findTransformECC on gradient magnitude · gated: tried coarse → fine, each result kept only if the fine-tile ECC score improves',
        model: null,
        desc: 'Polishes the fitted transform by maximising intensity correlation instead of point distances. In gated mode (default) a result that does not improve the fine-level alignment is rejected, so ECC can no longer make the fit worse. Legacy and Off are selectable in the run configuration.' },
      { id: 15, name: 'Sub-pixel Refinement',
        algo: 'Not run in this build: the model passes through unchanged',
        model: null,
        desc: 'Sub-pixel offsets already come from the phase correlation inside step 12’s dense matcher, so this stage currently only reports.' },
      { id: 16, name: 'Consensus Tie Points',
        algo: 'Step 13 inliers + per-point reprojection residual',
        model: null,
        desc: 'Keeps the inlier correspondences as tie points and measures each one’s residual under the final model (green = low error, red = high).' },
      { id: 17, name: 'Final Warp & Export',
        algo: 'Perspective warp (bilinear) + overlay / checkerboard / correspondence figures',
        model: null,
        desc: 'Warps the OHRC into the {REF} frame for the result views and saves the homography at native and fine resolution (.npy).' },
      { id: 18, name: 'Evaluation & Report',
        algo: 'In-sample RMSE · spatial held-out RMSE (interleaved bands along the strip) · NMI vs shifted baseline · trust checks',
        model: null,
        desc: 'Scores the registration: point error on the fit data and on held-out stretches of ground, plus normalised mutual information against a deliberately misaligned copy. Trusted only if inliers, coverage, held-out error (≤ 5 px) and NMI (≥ 1.5× baseline) all pass. Writes metrics.json.' },
    ]
  }
];

export const ALL_STEPS = PIPELINE_GROUPS.flatMap(g => g.steps.map(s => ({ ...s, group: g.label })));
export const TOTAL_STEPS = ALL_STEPS.length;
export const STEP_BY_ID = Object.fromEntries(ALL_STEPS.map(s => [s.id, s]));
export const STATE_LABELS = { pending: 'Pending', running: 'Running', done: 'Done', error: 'Failed' };
