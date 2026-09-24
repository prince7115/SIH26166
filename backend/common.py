"""Pipeline-agnostic helpers shared by every registration pipeline.

Geodesy, transform algebra, appearance preprocessing, orientation search, guarded model
fitting, the dense matcher and image encoding. Everything here was lifted unchanged from the
v4 server; the PDS4 / stretch / anti-aliased resampling helpers at the bottom come from the
v5-tmc notebook.
"""
import os, io, json, glob, zipfile, base64
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from skimage.exposure import match_histograms
from PIL import Image as PILImage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MOON_R_KM      = 1737.4
KM_PER_DEG_LAT = 2.0 * np.pi * MOON_R_KM / 360.0

def km_per_deg_lon(lat_deg):
    return KM_PER_DEG_LAT * float(np.cos(np.radians(lat_deg)))

def unwrap_lon(lon, ref):
    lon = np.asarray(lon, dtype=np.float64)
    return lon - 360.0 * np.round((lon - float(ref)) / 360.0)

def ground_extent_km(lat_min, lat_max, lon_min, lon_max):
    mid_lat = 0.5 * (lat_min + lat_max)
    return ((lat_max - lat_min) * KM_PER_DEG_LAT,
            (lon_max - lon_min) * km_per_deg_lon(mid_lat))

def fit_affine_pixel_to_latlon(corners_deg, shape):
    n_lines, n_samples = shape
    pix = np.array([[0, 0], [0, n_samples - 1],
                    [n_lines - 1, 0], [n_lines - 1, n_samples - 1]], dtype=np.float64)
    keys = ("UL", "UR", "LL", "LR")
    lat = np.array([corners_deg[k]["lat"] for k in keys], dtype=np.float64)
    lon_ref = float(corners_deg["UL"]["lon"])
    lon = np.array([float(unwrap_lon(corners_deg[k]["lon"], lon_ref)) for k in keys])
    A = np.column_stack([pix[:, 0], pix[:, 1], np.ones(4)])
    coef_lat, *_ = np.linalg.lstsq(A, lat, rcond=None)
    coef_lon, *_ = np.linalg.lstsq(A, lon, rcond=None)
    M = np.array([coef_lat[:2], coef_lon[:2]])
    t = np.array([coef_lat[2], coef_lon[2]])
    M_inv = np.linalg.inv(M)
    def forward(row, col):
        return tuple(M @ np.array([row, col], dtype=np.float64) + t)
    def inverse(lat_, lon_):
        lon_u = float(unwrap_lon(lon_, lon_ref))
        row, col = M_inv @ (np.array([lat_, lon_u], dtype=np.float64) - t)
        return row, col
    return forward, inverse

# Transform helpers
def mat_translate(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])

def mat_scale(sx, sy=None):
    sy = sx if sy is None else sy
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])

def to_h(M23):
    return np.vstack([np.asarray(M23, dtype=np.float64), [0.0, 0.0, 1.0]])

def apply_h(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0: return pts
    p = np.hstack([pts, np.ones((len(pts), 1))])
    q = (np.asarray(H, dtype=np.float64) @ p.T).T
    return q[:, :2] / q[:, 2:3]

def H_at_level(H, gsd_from, gsd_to):
    k = gsd_from / gsd_to
    return mat_scale(k) @ np.asarray(H, dtype=np.float64) @ mat_scale(1.0 / k)

def mat_flip(flip_code, h, w):
    if flip_code is None: return np.eye(3)
    fx = -1.0 if flip_code in (1, -1) else 1.0
    fy = -1.0 if flip_code in (0, -1) else 1.0
    tx = (w - 1) if fx < 0 else 0.0
    ty = (h - 1) if fy < 0 else 0.0
    return np.array([[fx, 0.0, tx], [0.0, fy, ty], [0.0, 0.0, 1.0]])


# Image preprocessing
def to_uint8(img, lo_p=1.0, hi_p=99.0, mask=None):
    a = np.asarray(img, dtype=np.float32)
    sel = np.isfinite(a)
    if mask is not None: sel &= np.asarray(mask, dtype=bool)
    finite = a[sel]
    if finite.size < 8: finite = a[np.isfinite(a)]
    if finite.size == 0: return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(finite, lo_p), np.percentile(finite, hi_p)
    if hi <= lo: hi = lo + 1.0
    a = np.nan_to_num(a, nan=lo)
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

def apply_clahe(img_u8, clip_limit, tile=(8, 8)):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(img_u8)

def grad_mag_u8(img_u8, sigma=1.2):
    f = np.asarray(img_u8, dtype=np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), sigma)
    return to_uint8(g, 1, 99)

def texture_score(img_u8):
    a = np.asarray(img_u8, np.float32)
    if a.size < 64: return 0.0
    return float(cv2.Laplacian(a, cv2.CV_32F, ksize=3).var())

def to_rgb_pil(gray_u8):
    return PILImage.fromarray(np.stack([gray_u8] * 3, axis=-1))

def prep_pair(ohrc_raw, nac_raw, mask=None):
    o_pc, n_pc = to_uint8(ohrc_raw, mask=mask), to_uint8(nac_raw)
    o_sg = apply_clahe(o_pc, 2.0)
    n_sg = apply_clahe(n_pc, 4.0)
    n_sg = np.clip(match_histograms(n_sg.astype(np.float32), o_sg.astype(np.float32)), 0, 255).astype(np.uint8)
    return o_pc, n_pc, o_sg, n_sg

# Resolution
def level_view(raster, native_gsd, target_gsd):
    f = native_gsd / target_gsd
    step = max(1, int(1.0 / f / 2)) if f < 1 else 1
    small = np.asarray(raster[::step, ::step])
    f2 = (native_gsd * step) / target_gsd
    if abs(f2 - 1.0) < 1e-6: return small
    return cv2.resize(small, None, fx=f2, fy=f2,
                      interpolation=cv2.INTER_AREA if f2 < 1 else cv2.INTER_LINEAR)

def level_tile(raster, native_gsd, target_gsd, r0, c0, h, w):
    s = target_gsd / native_gsd
    R0, C0 = max(0, int(np.floor(r0 * s))), max(0, int(np.floor(c0 * s)))
    R1 = min(raster.shape[0], int(np.ceil((r0 + h) * s)))
    C1 = min(raster.shape[1], int(np.ceil((c0 + w) * s)))
    if R1 - R0 < 8 * s or C1 - C0 < 8 * s: return None, None
    patch = np.asarray(raster[R0:R1, C0:C1])
    oh, ow = max(1, int(round((R1 - R0) / s))), max(1, int(round((C1 - C0) / s)))
    return cv2.resize(patch, (ow, oh), interpolation=cv2.INTER_AREA), (C0 / s, R0 / s)

def level_shape(crop_shape, native_gsd, target_gsd):
    k = native_gsd / target_gsd
    return (int(round(crop_shape[0] * k)), int(round(crop_shape[1] * k)))

# Orientation search
def _phase_randomize(img, seed=0):
    f = np.asarray(img, np.float32)
    F = np.fft.fft2(f)
    r = np.random.default_rng(seed)
    ph = r.uniform(-np.pi, np.pi, F.shape)
    return to_uint8(np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * ph))))

def _zgrad(x):
    g = grad_mag_u8(x).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-9)

def _best_ncc(A_z, A_img, B, scales):
    best = (-1.0, None, None)
    for s in scales:
        h, w = int(round(B.shape[0] * s)), int(round(B.shape[1] * s))
        if h < 24 or w < 16 or h > 4000 or w > 4000: continue
        Bs = cv2.resize(B, (w, h), interpolation=cv2.INTER_CUBIC)
        if h <= A_img.shape[0] and w <= A_img.shape[1]:
            search_z, tpl_z = A_z, _zgrad(Bs)
            search_shape, tpl_shape = A_img.shape, (h, w)
        elif A_img.shape[0] <= h and A_img.shape[1] <= w:
            search_z, tpl_z = _zgrad(Bs), A_z
            search_shape, tpl_shape = (h, w), A_img.shape
        else: continue
        if (tpl_shape[0] * tpl_shape[1]) / float(search_shape[0] * search_shape[1]) < 0.35: continue
        try: res = cv2.matchTemplate(search_z, tpl_z, cv2.TM_CCOEFF_NORMED)
        except cv2.error: continue
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]: best = (float(mx), round(float(s), 3), loc)
    return best

def ncc_orientation_search(a_u8, b_u8, scales=None, max_side=400):
    scales = np.arange(0.80, 1.26, 0.02) if scales is None else scales
    _k = min(1.0, max_side / max(max(a_u8.shape[:2]), max(b_u8.shape[:2])))
    def shrink(x):
        return cv2.resize(x, None, fx=_k, fy=_k, interpolation=cv2.INTER_AREA) if _k < 1.0 else x
    A, B = shrink(a_u8), shrink(b_u8)
    A_z = _zgrad(A)
    rows = []
    for name, code, Bv in (("no flip", None, B), ("horizontal", 1, B[:, ::-1].copy()),
                           ("vertical", 0, B[::-1, :].copy()), ("both", -1, B[::-1, ::-1].copy())):
        mx, sc, loc = _best_ncc(A_z, A, Bv, scales)
        if sc is not None: rows.append((mx, name, code, sc, loc))
    rows.sort(key=lambda t: -t[0])
    floor = max([_best_ncc(A_z, A, _phase_randomize(B, seed=k), scales)[0] for k in range(3)] or [1e-6])
    return rows, max(float(floor), 1e-6)

# Model fitting
_USAC = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)
_EMPTY = (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32))

def hull_coverage(pts, frame_shape):
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(pts) < 3: return 0.0
    return float(cv2.contourArea(cv2.convexHull(pts)) / (frame_shape[0] * frame_shape[1]))

def is_degenerate(pts, min_minor_axis_px=5.0):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) < 3: return True
    ev = np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T) + 1e-12 * np.eye(2))
    return bool(np.sqrt(max(ev.min(), 0.0)) < min_minor_axis_px)

def minor_axis_px(pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) < 3: return 0.0
    ev = np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T) + 1e-12 * np.eye(2))
    return float(np.sqrt(max(ev.min(), 0.0)))

def robust_fit(src, dst, frame_shape, thresh=3.0, verbose=False, min_minor_axis_px=5.0):
    src = np.asarray(src, np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst, np.float32).reshape(-1, 1, 2)
    if len(src) < 4: return None, None, None
    best = (None, None, None)
    M, m = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thresh, maxIters=20000, confidence=0.999)
    if M is not None and m is not None: best = (to_h(M), m.ravel().astype(bool), 'similarity')
    if best[1] is not None and best[1].sum() >= 20:
        M, m = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thresh, maxIters=20000, confidence=0.999)
        if M is not None and m is not None and m.ravel().sum() >= best[1].sum():
            best = (to_h(M), m.ravel().astype(bool), 'affine')
    if best[1] is not None and best[1].sum() >= 40:
        cov = hull_coverage(dst.reshape(-1, 2)[best[1]], frame_shape)
        if cov >= 0.10:
            H, m = cv2.findHomography(src, dst, _USAC, thresh, maxIters=50000, confidence=0.9995)
            if H is not None and m is not None and m.ravel().sum() >= best[1].sum():
                best = (H, m.ravel().astype(bool), 'homography')
    H, mask, name = best
    if H is None: return None, None, None
    if is_degenerate(dst.reshape(-1, 2)[mask], min_minor_axis_px=min_minor_axis_px):
        return None, None, None
    return H, mask, name

# SuperGlue
def pad_to_square(img_u8):
    h, w = img_u8.shape[:2]
    n = max(h, w)
    if n == h == w: return img_u8, 0, 0
    out = np.zeros((n, n), np.uint8)
    ox, oy = (n - w) // 2, (n - h) // 2
    out[oy:oy + h, ox:ox + w] = img_u8
    return out, ox, oy

# Dense matcher
def match_dense(warped_u8, ref_u8, valid_mask, subdiv=3, min_margin=2.0):
    h, w = ref_u8.shape[:2]
    ys, xs = np.where(valid_mask)
    if len(ys) < 400: return _EMPTY
    Y0, Y1, X0, X1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    b_z = _zgrad(ref_u8)
    a_z = _zgrad(warped_u8)
    ka, kb, sc = [], [], []
    for iy in range(subdiv):
        for ix in range(subdiv):
            ph = (Y1 - Y0) // subdiv; pw = (X1 - X0) // subdiv
            if ph < 64 or pw < 64: continue
            ty = Y0 + iy * ph; tx = X0 + ix * pw
            th = int(ph * 0.85); tw_ = int(pw * 0.85)
            ty += (ph - th) // 2; tx += (pw - tw_) // 2
            if valid_mask[ty:ty + th, tx:tx + tw_].mean() < 0.9: continue
            tpl = a_z[ty:ty + th, tx:tx + tw_]
            if tpl.shape[0] > b_z.shape[0] or tpl.shape[1] > b_z.shape[1]: continue
            try:
                _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(b_z, tpl, cv2.TM_CCOEFF_NORMED))
                null = max(cv2.minMaxLoc(cv2.matchTemplate(
                    _zgrad(_phase_randomize(ref_u8, seed=q)), tpl, cv2.TM_CCOEFF_NORMED))[1] for q in range(2))
            except cv2.error: continue
            if mx / max(null, 1e-6) < min_margin: continue
            dx, dy = loc[0] - tx, loc[1] - ty
            sy0, sx0 = int(ty + dy), int(tx + dx)
            if sy0 < 0 or sx0 < 0 or sy0 + th > h or sx0 + tw_ > w: continue
            Ap = a_z[ty:ty + th, tx:tx + tw_]
            Bp = b_z[sy0:sy0 + th, sx0:sx0 + tw_]
            try:
                win = cv2.createHanningWindow((tw_, th), cv2.CV_32F)
                (sdx, sdy), _r = cv2.phaseCorrelate(np.ascontiguousarray(Ap) * win, np.ascontiguousarray(Bp) * win)
                if abs(sdx) > 2 or abs(sdy) > 2: sdx = sdy = 0.0
            except cv2.error: sdx = sdy = 0.0
            cx, cy = tx + tw_ / 2.0, ty + th / 2.0
            ka.append([cx, cy]); kb.append([cx + dx + sdx, cy + dy + sdy])
            sc.append(float(min(1.0, mx)))
    if not ka: return _EMPTY
    return (np.asarray(ka, np.float32), np.asarray(kb, np.float32), np.asarray(sc, np.float32))


# ============================================================
# IMAGE ENCODING
# ============================================================
def img_to_b64(img, max_side=800):
    """Convert a numpy grayscale image to a base64 PNG data URL."""
    if img is None: return None
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()

def fig_to_b64(fig, dpi=120):
    """Convert a matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='#0a0e1a', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def side_by_side(a, b, height=256):
    """Two rasters stretched to uint8, resized to a common height and stacked."""
    def _fit(x):
        x = to_uint8(x)
        if x.shape[0] <= 4: return x
        return cv2.resize(x, (max(1, int(height * x.shape[1] / max(x.shape[0], 1))), height),
                          interpolation=cv2.INTER_AREA)
    a, b = _fit(a), _fit(b)
    if a.shape[0] != b.shape[0]:
        h = max(a.shape[0], b.shape[0])
        a = np.pad(a, ((0, h - a.shape[0]), (0, 0)))
        b = np.pad(b, ((0, h - b.shape[0]), (0, 0)))
    return np.hstack([a, b])


# ============================================================
# FILE DISCOVERY
# ============================================================
_SKIP_DIRS = {"calibrated", "browse", ".ipynb_checkpoints", "__pycache__"}

def _walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in _SKIP_DIRS]
        for fn in filenames: yield os.path.join(dirpath, fn)

def find_first(root, exts, recursive=True):
    if not os.path.isdir(root): return None
    exts = {e.lower() for e in exts}
    src = _walk(root) if recursive else (
        os.path.join(root, f) for f in sorted(os.listdir(root))
        if os.path.isfile(os.path.join(root, f)))
    for p in sorted(src):
        if os.path.splitext(p)[1].lower() in exts: return p
    return None

def _localise(path, data_raw):
    """LINK.json files were written on Colab (/content/drive/.../data/raw/...). If that path
    does not exist here, re-root everything after 'data/raw/' onto this machine's DATA_RAW."""
    if os.path.exists(path): return path
    norm = path.replace("\\", "/")
    marker = "/data/raw/"
    if marker in norm:
        cand = os.path.join(data_raw, *norm.split(marker, 1)[1].split("/"))
        if os.path.exists(cand): return cand
    return path

def resolve_pair_dir(data_raw, kind, pair_id):
    """(img, xml) for data/raw/<kind>/<pair_id>, following a LINK.json if one is there.

    LINK.json is how a product shared between pairs is stored once (the OHRC in pair02_tmc
    is the same file pair01_equatorial uses)."""
    d = os.path.join(data_raw, kind, pair_id)
    link = os.path.join(d, "LINK.json")
    if os.path.exists(link):
        with open(link) as f:
            ln = json.load(f)
        img, xml = _localise(ln["img"], data_raw), _localise(ln["xml"], data_raw)
        missing = [p for p in (img, xml) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{kind}/{pair_id}/LINK.json points at files that are not there: "
                + ", ".join(missing))
        return img, xml

    img = find_first(d, [".img", ".qub"])
    if img is None:                                   # auto-unzip a downloaded bundle
        z = find_first(d, [".zip"])
        if z:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(os.path.dirname(z))
            img = find_first(d, [".img", ".qub"])
    xml = find_first(d, [".xml"])
    if img is None or xml is None:
        raise FileNotFoundError(f"Need a raster and a .xml label under {d} "
                                f"(found img={img}, xml={xml}).")
    return img, xml

def has_product(data_raw, kind, pair_id):
    """True if data/raw/<kind>/<pair_id> holds a raster or a LINK.json."""
    d = os.path.join(data_raw, kind, pair_id)
    return (os.path.exists(os.path.join(d, "LINK.json"))
            or find_first(d, [".img", ".qub", ".zip"]) is not None)


# ============================================================
# PDS4 LOADER (v5-tmc) — dtype and GSD read from the label
# ============================================================
NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1",
      "isda": "https://isda.issdc.gov.in/pds4/isda/v1"}

PDS4_DTYPE = {
    "UnsignedByte":     "u1",  "SignedByte":       "i1",
    "UnsignedLSB2":     "<u2", "SignedLSB2":       "<i2",
    "UnsignedMSB2":     ">u2", "SignedMSB2":       ">i2",
    "UnsignedLSB4":     "<u4", "SignedLSB4":       "<i4",
    "IEEE754LSBSingle": "<f4", "IEEE754LSBDouble": "<f8",
}

def load_pds4(img_path, xml_path, name="PRODUCT", allow_truncated=False, refuse_polar=True):
    """(raster_memmap, corners_deg, (lines, samples), gsd_m, meta).

    The raster stays in the label's own dtype; 16-bit products are stretched to uint8 only
    after cropping (see stretch_to_uint8)."""
    root = ET.parse(xml_path).getroot()

    axes = root.findall(".//pds:Array_2D_Image/pds:Axis_Array", NS)
    dims = {a.find("pds:axis_name", NS).text: int(a.find("pds:elements", NS).text) for a in axes}
    n_lines, n_samples = dims["Line"], dims["Sample"]

    _dt = root.find(".//pds:Element_Array/pds:data_type", NS)
    dt_raw = _dt.text.strip() if _dt is not None else "UnsignedByte"
    if dt_raw not in PDS4_DTYPE:
        raise ValueError(f"{name}: unmapped PDS4 data_type {dt_raw!r}.")
    dtype = np.dtype(PDS4_DTYPE[dt_raw])

    geom = root.find(".//isda:Refined_Corner_Coordinates", NS)
    if geom is None:
        geom = root.find(".//isda:System_Level_Coordinates", NS)
    if geom is None:
        raise ValueError(f"{name}: the label carries no footprint corners. This is probably a "
                         f"BROWSE product (*_b_brw_*); use the data product (*_d_img_*).")
    corners_deg = {k: {"lat": float(geom.find(f"isda:{t}_latitude", NS).text),
                       "lon": float(geom.find(f"isda:{t}_longitude", NS).text) % 360.0}
                   for k, t in (("UL", "upper_left"), ("UR", "upper_right"),
                                ("LL", "lower_left"), ("LR", "lower_right"))}

    def _txt(path, cast=str, default=None):
        el = root.find(path, NS)
        if el is None or el.text is None: return default
        try: return cast(el.text.strip())
        except (TypeError, ValueError): return default

    meta = {
        "area":       _txt(".//isda:area", str, "?"),
        "projection": _txt(".//isda:projection", str, "?"),
        "level":      _txt(".//pds:Primary_Result_Summary/pds:processing_level", str, "?"),
        "sun_elev":   _txt(".//isda:sun_elevation", float),
        "sun_azim":   _txt(".//isda:sun_azimuth", float),
        "roll":       _txt(".//isda:roll", float),
        "pitch":      _txt(".//isda:pitch", float),
        "dtype_pds4": dt_raw,
    }

    if refuse_polar and ("pole" in str(meta["area"]).lower()
                         or str(meta["projection"]).lower().startswith("polar")):
        raise RuntimeError(f"{name} is a POLAR product (area={meta['area']}, projection="
                           f"{meta['projection']}); the plane-affine corner model does not "
                           f"hold there. Pick a Selenographic/Equatorial product.")

    gsd = _txt(".//isda:pixel_resolution", float)          # the authoritative field
    if gsd is None:                                         # loose fallback scan
        for el in root.iter():
            tag = el.tag.split('}')[-1].lower()
            if ('resolution' in tag or 'pixel_size' in tag) and el.text:
                try: v = float(el.text.strip().split()[0])
                except ValueError: continue
                if 0.05 < v < 500.0:
                    gsd = v
                    break

    expected = n_lines * n_samples * dtype.itemsize
    actual = os.path.getsize(img_path)
    if actual < expected:
        avail = actual // (n_samples * dtype.itemsize)
        if not allow_truncated:
            raise ValueError(f"{name} file truncated: {avail:,}/{n_lines:,} lines present.")
        frac = avail / float(n_lines)
        for top, bot in (("UL", "LL"), ("UR", "LR")):
            for k in ("lat", "lon"):
                corners_deg[bot][k] = corners_deg[top][k] + frac * (corners_deg[bot][k] - corners_deg[top][k])
        n_lines = avail

    raster = np.memmap(img_path, dtype=dtype, mode="r", shape=(n_lines, n_samples))
    return raster, corners_deg, (n_lines, n_samples), gsd, meta

def stretch_to_uint8(a, lo_pct=2.0, hi_pct=98.0, sample_rows=2000):
    """Percentile stretch to 8-bit, block-wise so a large memmap never lands in RAM whole.
    Zeros (strip padding) are excluded from the percentiles."""
    if a.dtype == np.uint8:
        return np.asarray(a)
    step = max(1, a.shape[0] // sample_rows)
    s = np.asarray(a[::step]).ravel()
    s = s[s > 0]
    if s.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(s, [lo_pct, hi_pct])
    if hi <= lo: hi = lo + 1.0
    out = np.empty(a.shape, np.uint8)
    for r0 in range(0, a.shape[0], 4096):
        r1 = min(a.shape[0], r0 + 4096)
        blk = (np.asarray(a[r0:r1], np.float32) - lo) * (255.0 / (hi - lo))
        out[r0:r1] = np.clip(blk, 0, 255).astype(np.uint8)
    return out

def level_view_antialiased(raster, native_gsd, target_gsd, band_rows=4096):
    """Whole raster resampled to target_gsd, anti-aliased and memory-bounded (v5-tmc).

    level_view decimates with raster[::step, ::step] first, which folds high frequencies
    back in as noise. Harmless at NAC's 3.3x ratio, destructive at TMC's ~19x."""
    f = native_gsd / target_gsd
    h, w = raster.shape[:2]
    if abs(f - 1.0) < 1e-6:
        return np.asarray(raster)
    out_h, out_w = max(1, int(round(h * f))), max(1, int(round(w * f)))
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
    if raster.dtype != np.uint8:
        raise TypeError(f"level_view_antialiased needs uint8, got {raster.dtype}.")
    out = np.zeros((out_h, out_w), np.uint8)
    for r0 in range(0, h, band_rows):
        r1 = min(h, r0 + band_rows)
        y0, y1 = int(round(r0 * f)), min(out_h, int(round(r1 * f)))
        if y1 <= y0: continue
        out[y0:y1] = cv2.resize(np.asarray(raster[r0:r1]), (out_w, y1 - y0), interpolation=interp)
    return out
