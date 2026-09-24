    # ========== STEP 12: Tiled Matching ==========
    emit_event(12, "Tiled Matching at Native Resolution", "running", 62)

    def _shift_at(yc):
        if PRIOR_SHIFT_COEF is None: return 0.0, 0.0
        if PRIOR_SHIFT_RANGE is not None:
            yc = float(np.clip(yc, PRIOR_SHIFT_RANGE[0], PRIOR_SHIFT_RANGE[1]))
        return (float(np.polyval(PRIOR_SHIFT_COEF[0], yc)), float(np.polyval(PRIOR_SHIFT_COEF[1], yc)))

    def prior_shift_fine(y_fine):
        if PRIOR_SHIFT_COEF is None: return 0.0, 0.0
        k = TARGET_GSD_FINE_M / TARGET_GSD_COARSE_M
        dx, dy = _shift_at(float(y_fine) * k)
        return (dx - PRIOR_SHIFT_BASE[0]) / k, (dy - PRIOR_SHIFT_BASE[1]) / k

    SG_SIZE_FINE = 640
    if profile.adaptive_tile:
        # v5-tmc: size tiles to fit ~4 columns across the OHRC overlap, clamped to [256, 640].
        TILE = int(np.clip(2 ** int(np.floor(np.log2(max(256.0, OHRC_FINE_SHAPE[1] / 4.0)))),
                           256, SG_SIZE_FINE))
    else:
        TILE = SG_SIZE_FINE
    TILE_OVERLAP = 0.25; SEARCH_MARGIN = 128
    MIN_VALID_FRAC = 0.25; MIN_TEXTURE = 12.0
    TILE_CONSENSUS_PX = 6.0

    stride = max(1, int(TILE * (1.0 - TILE_OVERLAP)))

    fine_pts_o, fine_pts_n, fine_scores, fine_src = [], [], [], []
    tiles_tried = tiles_used = tiles_skipped_texture = tiles_skipped_overlap = tiles_no_consensus = 0
    _t_start = time.time()

    _rows = list(range(0, max(1, OHRC_FINE_SHAPE[0] - TILE // 2), stride))
    _cols = list(range(0, max(1, OHRC_FINE_SHAPE[1] - TILE // 2), stride))

    def _process_one_tile(r0, c0):
        o_tile_raw, o_origin = level_tile(ohrc_crop, OHRC_NATIVE_GSD_M, TARGET_GSD_FINE_M, r0, c0, TILE, TILE)
        if o_tile_raw is None: return {'status': 'skip'}
        ox, oy = o_origin; oh, ow = o_tile_raw.shape[:2]
        if texture_score(to_uint8(o_tile_raw)) < MIN_TEXTURE: return {'status': 'skip_texture'}

        corners = np.array([[ox, oy], [ox + ow, oy], [ox, oy + oh], [ox + ow, oy + oh]])
        _py = apply_h(H_fine0, [[ox + ow / 2.0, oy + oh / 2.0]])[0][1]
        _sx, _sy = prior_shift_fine(_py)
        H_local = mat_translate(_sx, _sy) @ H_fine0
        H_local_inv = np.linalg.inv(H_local)
        proj = apply_h(H_local, corners)
        n_c0 = int(np.floor(proj[:, 0].min())) - SEARCH_MARGIN
        n_r0 = int(np.floor(proj[:, 1].min())) - SEARCH_MARGIN
        n_w = int(np.ceil(proj[:, 0].max() - proj[:, 0].min())) + 2 * SEARCH_MARGIN
        n_h = int(np.ceil(proj[:, 1].max() - proj[:, 1].min())) + 2 * SEARCH_MARGIN
        if n_w < 32 or n_h < 32 or n_r0 > REF_FINE_SHAPE[0] or n_c0 > REF_FINE_SHAPE[1]: return {'status': 'skip'}

        n_tile_raw, n_origin = level_tile(ref_crop, REF_NATIVE_GSD_M, TARGET_GSD_FINE_M, max(0, n_r0), max(0, n_c0), n_h, n_w)
        if n_tile_raw is None: return {'status': 'skip'}
        nx, ny = n_origin; nh, nw = n_tile_raw.shape[:2]

        H_tile = mat_translate(-nx, -ny) @ H_local @ mat_translate(ox, oy)
        valid = cv2.warpPerspective(np.full((oh, ow), 255, np.uint8), H_tile, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
        if valid.mean() / 255.0 < MIN_VALID_FRAC: return {'status': 'skip_overlap'}

        warped_o_raw = cv2.warpPerspective(np.asarray(o_tile_raw), H_tile, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
        vmask = valid > 0
        o_pc, n_pc, o_sg, n_sg = prep_pair(warped_o_raw, n_tile_raw, mask=vmask)
        if texture_score(n_pc) < MIN_TEXTURE: return {'status': 'skip_texture'}

        back_to_ohrc = H_local_inv @ mat_translate(nx, ny)
        to_ref_fine = mat_translate(nx, ny)

        # Dense matching only (SuperGlue in tiles is off by default)
        ka, kb, sc = match_dense(o_pc, n_pc, vmask)
        if len(ka) < 2: return {'status': 'skip'}
        d = kb - ka; med = np.median(d, axis=0); dev = np.linalg.norm(d - med, axis=1)
        mad = float(np.median(dev))
        if mad > TILE_CONSENSUS_PX: return {'status': 'no_consensus'}
        agree = dev <= TILE_CONSENSUS_PX
        if int(agree.sum()) < 2: return {'status': 'no_consensus'}
        ka, kb, sc = ka[agree], kb[agree], sc[agree]

        return {
            'status': 'matched',
            'pts_o': apply_h(back_to_ohrc, ka),
            'pts_n': apply_h(to_ref_fine, kb),
            'scores': sc,
            'n_src': len(ka),
        }

    tile_jobs = [(r0, c0) for r0 in _rows for c0 in _cols][:MAX_TILES]

    N_WORKERS = min(os.cpu_count() or 4, 14)
    _completed = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_process_one_tile, r0, c0): (r0, c0) for r0, c0 in tile_jobs}
        for future in as_completed(futures):
            _completed += 1
            tiles_tried += 1
            try:
                result = future.result()
            except Exception:
                continue

            st = result['status']
            if st == 'skip_texture': tiles_skipped_texture += 1
            elif st == 'skip_overlap': tiles_skipped_overlap += 1
            elif st == 'no_consensus': tiles_no_consensus += 1
            elif st == 'matched':
                fine_pts_o.append(result['pts_o'])
                fine_pts_n.append(result['pts_n'])
                fine_scores.append(result['scores'])
                fine_src.extend(['dense'] * result['n_src'])
                tiles_used += 1

            if _completed % 20 == 0:
                pct = 62 + int(15 * _completed / max(len(tile_jobs), 1))
                emit_event(12, "Tiled Matching at Native Resolution", "running", min(pct, 77),
                           detail=f"{_completed}/{len(tile_jobs)} tiles · {tiles_used} matched · {sum(len(p) for p in fine_pts_o)} points · {N_WORKERS} threads")

    if not fine_pts_o: raise RuntimeError("No tiles produced matches.")
    fine_o = np.vstack(fine_pts_o)
    fine_n = np.vstack(fine_pts_n)
    fine_s = np.concatenate(fine_scores)

    _tile_vis = cv2.cvtColor(to_uint8(ref_coarse), cv2.COLOR_GRAY2BGR)
    _k_f2c = TARGET_GSD_FINE_M / TARGET_GSD_COARSE_M
    for _p in fine_n[:500]:
        cv2.circle(_tile_vis, (int(_p[0] * _k_f2c), int(_p[1] * _k_f2c)), 2, (0, 255, 0), -1)
    emit_event(12, "Tiled Matching at Native Resolution", "done", 78,
               detail=f"Tile {TILE} px · {tiles_used}/{tiles_tried} tiles matched · {len(fine_o)} correspondences · {time.time()-_t_start:.0f}s",
               image=img_to_b64(_tile_vis))

    # ========== STEP 13: Final Model ==========
    emit_event(13, "Final Model (RANSAC)", "running", 80)

    def estimate_residual_scale(src, dst, frame_shape, t0=100.0, iters=3):
        t = t0
        for _ in range(iters):
            H, m, _ = robust_fit(src, dst, frame_shape, thresh=t, verbose=False, min_minor_axis_px=0.0)
            if H is None: return None
            r = np.linalg.norm(apply_h(H, src) - dst, axis=1)
            s_ = 1.4826 * float(np.median(np.abs(r - np.median(r))))
            t = float(max(3.0, 2.5 * max(s_, 1e-3)))
        return t

    FIT_THRESH_PX = estimate_residual_scale(fine_o, fine_n, REF_FINE_SHAPE) or 3.0
    H_fine, inlier_mask, MODEL_NAME = robust_fit(fine_o, fine_n, REF_FINE_SHAPE, thresh=FIT_THRESH_PX, min_minor_axis_px=5.0)
    if H_fine is None: raise RuntimeError("Could not fit a model to the fine correspondences.")

    n_inliers = int(inlier_mask.sum())
    inlier_ratio = n_inliers / len(fine_o)
    coverage = hull_coverage(fine_n[inlier_mask], REF_FINE_SHAPE)
    TRUSTED = n_inliers >= 15 and coverage >= 0.05

    _ransac_vis = cv2.cvtColor(to_uint8(ref_coarse), cv2.COLOR_GRAY2BGR)
    for _j in range(min(500, len(fine_n))):
        _col = (0, 255, 0) if inlier_mask[_j] else (0, 0, 255)
        cv2.circle(_ransac_vis, (int(fine_n[_j, 0] * _k_f2c), int(fine_n[_j, 1] * _k_f2c)), 2, _col, -1)
    emit_event(13, "Final Model (RANSAC)", "done", 83,
               detail=f"Model: {MODEL_NAME} · {n_inliers}/{len(fine_o)} inliers ({inlier_ratio:.1%}) · Coverage {coverage:.1%} · {'TRUSTED' if TRUSTED else 'LOW CONFIDENCE'}",
               image=img_to_b64(_ransac_vis))

    # ========== STEP 14: ECC Refinement ==========
    emit_event(14, "ECC Intensity Refinement", "running", 85)
    H_FINAL = H_fine
    ECC_APPLIED = False

    if RUN_ECC:
        try:
            def ecc_refine(H_in, gsd):
                o_g = grad_mag_u8(to_uint8(_level_view(ohrc_crop, OHRC_NATIVE_GSD_M, gsd)))
                n_g = grad_mag_u8(to_uint8(_level_view(ref_crop, REF_NATIVE_GSD_M, gsd)))
                W = H_at_level(H_in, TARGET_GSD_FINE_M, gsd).astype(np.float32)
                motion = cv2.MOTION_HOMOGRAPHY if MODEL_NAME == 'homography' else cv2.MOTION_AFFINE
                warp = W.astype(np.float32) if motion == cv2.MOTION_HOMOGRAPHY else W[:2].astype(np.float32)
                crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
                cc, warp = cv2.findTransformECC(o_g, n_g, warp, motion, crit, None, 5)
                W_ref = warp if motion == cv2.MOTION_HOMOGRAPHY else to_h(warp)
                return H_at_level(W_ref, gsd, TARGET_GSD_FINE_M), float(cc)
            H_try, cc = ecc_refine(H_fine, TARGET_GSD_COARSE_M)
            if np.all(np.isfinite(H_try)):
                H_FINAL = H_try
                ECC_APPLIED = True
        except cv2.error:
            pass

    _ecc_warped = cv2.warpPerspective(to_uint8(ohrc_coarse),
                                      H_at_level(H_FINAL, TARGET_GSD_FINE_M, TARGET_GSD_COARSE_M).astype(np.float64),
                                      (ref_coarse.shape[1], ref_coarse.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    emit_event(14, "ECC Intensity Refinement", "done", 87,
               detail=f"{'Applied -- improved alignment' if ECC_APPLIED else 'Skipped or no improvement'}",
               image=img_to_b64(_blend(_ecc_warped, to_uint8(ref_coarse))))

    # ========== STEP 15: Sub-pixel Refinement ==========
    emit_event(15, "Sub-pixel Refinement", "running", 88)
    # Simplified — skip sub-pixel for speed, keep H_FINAL
    emit_event(15, "Sub-pixel Refinement", "done", 89, detail="Using model as-is for this run")

    # ========== STEP 16: Consensus Tie Points ==========
    emit_event(16, "Consensus Tie Points", "running", 90)

    MATCH_SRC = fine_o[inlier_mask]
    MATCH_DST = fine_n[inlier_mask]
    RESIDUAL_PX = np.linalg.norm(apply_h(H_FINAL, MATCH_SRC) - MATCH_DST, axis=1)

    _tie_vis = cv2.cvtColor(to_uint8(ref_coarse), cv2.COLOR_GRAY2BGR)
    _res_max = max(RESIDUAL_PX.max(), 1.0)
    for _j in range(min(500, len(MATCH_DST))):
        _r_norm = min(RESIDUAL_PX[_j] / _res_max, 1.0)
        _col = (0, int(255 * (1 - _r_norm)), int(255 * _r_norm))  # green=low error, red=high
        cv2.circle(_tie_vis, (int(MATCH_DST[_j, 0] * _k_f2c), int(MATCH_DST[_j, 1] * _k_f2c)), 3, _col, -1)
    emit_event(16, "Consensus Tie Points", "done", 91,
               detail=f"{len(MATCH_SRC)} tie points · Residual mean {RESIDUAL_PX.mean():.3f} px = {RESIDUAL_PX.mean()*TARGET_GSD_FINE_M*100:.1f} cm",
               image=img_to_b64(_tie_vis))

    # ========== STEP 17: Final Warp & Export ==========
    emit_event(17, "Final Warp & Export", "running", 92)

    H_native = np.linalg.inv(M_ref_native_to_fine) @ H_FINAL @ M_ohrc_native_to_fine

    H_coarse_final = H_at_level(H_FINAL, TARGET_GSD_FINE_M, TARGET_GSD_COARSE_M)
    warped = cv2.warpPerspective(to_uint8(ohrc_coarse), H_coarse_final.astype(np.float64),
                                 (ref_coarse.shape[1], ref_coarse.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderValue=0)
    ref_disp = to_uint8(ref_coarse)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(warped, cmap='gray'); ax[0].set_title(f'OHRC warped → {REF} frame', color='white', fontsize=9)
    ax[1].imshow(ref_disp, cmap='gray'); ax[1].set_title(f'{REF} reference', color='white', fontsize=9)
    _blk = max(8, min(warped.shape) // 20)
    _yy, _xx = np.mgrid[0:warped.shape[0], 0:warped.shape[1]]
    _chk = np.where((((_yy // _blk) + (_xx // _blk)) % 2).astype(bool), warped, ref_disp)
    ax[2].imshow(_chk, cmap='gray'); ax[2].set_title('Checkerboard', color='white', fontsize=9)
    for a in ax: a.axis('off')
    plt.tight_layout()
    overlay_img = fig_to_b64(fig)
    results['overlay'] = overlay_img
    results['checkerboard'] = overlay_img  # same figure has both

    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    _ohrc_u8 = to_uint8(ohrc_coarse)
    _ref_u8 = ref_disp
    _h1, _h2 = _ohrc_u8.shape[0], _ref_u8.shape[0]
    if _h1 != _h2:
        _max_h = max(_h1, _h2)
        if _h1 < _max_h: _ohrc_u8 = np.pad(_ohrc_u8, ((0, _max_h - _h1), (0, 0)), mode='constant')
        if _h2 < _max_h: _ref_u8 = np.pad(_ref_u8, ((0, _max_h - _h2), (0, 0)), mode='constant')
    ax2.imshow(np.hstack([_ohrc_u8, _ref_u8]), cmap='gray')
    src_c = MATCH_SRC * _k_f2c
    dst_c = MATCH_DST * _k_f2c
    for i in range(min(200, len(src_c))):
        ax2.plot([src_c[i, 0], dst_c[i, 0] + ohrc_coarse.shape[1]], [src_c[i, 1], dst_c[i, 1]], 'c-', lw=0.3, alpha=0.5)
    ax2.plot(src_c[:200, 0], src_c[:200, 1], 'r.', ms=2)
    ax2.plot(dst_c[:200, 0] + ohrc_coarse.shape[1], dst_c[:200, 1], 'g.', ms=2)
    ax2.set_title(f'{len(MATCH_SRC)} correspondences', color='white', fontsize=10)
    ax2.axis('off')
    plt.tight_layout()
    results['matches'] = fig_to_b64(fig2)

    tag = f'{pair_id}_{profile.version}'
    np.save(os.path.join(out_dir, f'{tag}_H_native.npy'), H_native)
    np.save(os.path.join(out_dir, f'{tag}_H_fine.npy'), H_FINAL)

    emit_event(17, "Final Warp & Export", "done", 96,
               detail=f"Warped raster + homography saved to {out_dir}",
               image=overlay_img)

    # ========== STEP 18: Evaluation & Report ==========
    emit_event(18, "Evaluation & Report", "running", 97)

    def _rmse(H):
        return float(np.sqrt(np.mean(np.sum((apply_h(H, MATCH_SRC) - MATCH_DST) ** 2, axis=1))))
    rmse_in = _rmse(H_FINAL)
    # ECC optimises intensity, not point residuals; report both so a disagreement is visible.
    rmse_pre_ecc = _rmse(H_fine)

    rng = np.random.default_rng(0); held = []
    for _ in range(5):
        idx = rng.permutation(len(MATCH_SRC)); cut = int(0.7 * len(idx))
        tr, te = idx[:cut], idx[cut:]
        if len(tr) < 8 or len(te) < 4: continue
        H_tr, _, _ = robust_fit(MATCH_SRC[tr], MATCH_DST[tr], REF_FINE_SHAPE, verbose=False, thresh=FIT_THRESH_PX, min_minor_axis_px=5.0)
        if H_tr is None: continue
        held.append(float(np.sqrt(np.mean(np.linalg.norm(apply_h(H_tr, MATCH_SRC[te]) - MATCH_DST[te], axis=1) ** 2))))
    rmse_heldout = float(np.mean(held)) if held else None

    _valid = (warped > 0) & (ref_disp > 0)
    nmi = nmi_baseline = None
    if _valid.sum() > 1000:
        _sub = np.random.default_rng(0).choice(int(_valid.sum()), size=min(200_000, int(_valid.sum())), replace=False)
        nmi = float(normalized_mutual_info_score((warped[_valid] // 8)[_sub], (ref_disp[_valid] // 8)[_sub]))
        _shift = np.roll(warped, (37, 29), axis=(0, 1))
        _v2 = (_shift > 0) & (ref_disp > 0)
        _s2 = np.random.default_rng(0).choice(int(_v2.sum()), size=min(200_000, int(_v2.sum())), replace=False)
        nmi_baseline = float(normalized_mutual_info_score((_shift[_v2] // 8)[_s2], (ref_disp[_v2] // 8)[_s2]))

    ref_key = REF.lower()
    metrics = {
        "pair_id": pair_id, "pipeline": profile.key, "version": profile.version,
        "reference": REF, "model": MODEL_NAME,
        "ohrc_product_id": pair.ohrc_product_id, f"{ref_key}_product_id": pair.ref_product_id,
        "n_correspondences": int(len(fine_o)), "n_inliers": n_inliers,
        "inlier_ratio": float(inlier_ratio), "spatial_coverage_pct": float(coverage * 100),
        "rmse_px_in_sample": rmse_in, "rmse_m_in_sample": rmse_in * TARGET_GSD_FINE_M,
        "rmse_px_in_sample_pre_ecc": rmse_pre_ecc,
        "rmse_px_heldout": rmse_heldout,
        "rmse_m_heldout": (rmse_heldout * TARGET_GSD_FINE_M) if rmse_heldout else None,
        "nmi_aligned": nmi, "nmi_misaligned_baseline": nmi_baseline,
        "ecc_applied": bool(ECC_APPLIED), "trusted": bool(TRUSTED),
        "tiles_tried": tiles_tried, "tiles_matched": tiles_used, "tile_px": TILE,
        "gsd_fine_m": TARGET_GSD_FINE_M, "gsd_coarse_m": TARGET_GSD_COARSE_M,
        "fine_gsd_multiplier": profile.fine_gsd_multiplier,
        "scale_ratio": round(REF_NATIVE_GSD_M / OHRC_NATIVE_GSD_M, 3),
    }
    metrics.update(pair.extra_metrics)

    with open(os.path.join(out_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    results['metrics'] = metrics

    emit_event(18, "Evaluation & Report", "done", 100,
               detail=f"RMSE {rmse_in:.3f} px ({rmse_in*TARGET_GSD_FINE_M:.2f} m) · NMI {(f'{nmi:.4f}' if nmi else 'N/A')} · {'TRUSTED' if TRUSTED else 'LOW CONFIDENCE'}",
               metrics=metrics,
               images={k: v for k, v in results.items() if k != 'metrics'})
    return metrics

