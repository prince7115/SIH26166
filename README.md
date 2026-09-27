# Trinetra

Registers Chandrayaan-2 OHRC imagery to LRO NAC and Chandrayaan-2 TMC-2 reference frames. A
website runs the 18-step pipeline on a GPU backend and shows the output of every step.

```
┌──────────────────┐                        ┌──────────────────────┐
│  frontend/       │ ◄═══════════════════►  │  backend/ (GPU)      │
│  Vite dev server │     REST API + SSE     │  Flask API +         │
│  localhost:5173  │                        │  registration        │
└──────────────────┘                        └──────────────────────┘
```

## Repository layout

```
backend/
  trinetra/
    api/            Flask endpoints and the server-sent-event run stream
    sensors/        steps 1-2 for each sensor pair: ohrc_nac.py, ohrc_tmc.py
    registration/   steps 3-18, shared by both pairs
    matching/       keypoint, dense and crater matching; robust model fitting
    products/       PDS4 / PDS3 readers, file discovery, sun geometry, label metadata
    geometry/       lunar geodesy and homogeneous transforms
    imaging/        contrast stretching and resampling
    reporting/      step previews and result figures sent to the website
  tests/
frontend/
  index.html
  src/              ES modules (main.js wires them together) and styles/
notebooks/          research notebooks the two pipelines were ported from
colab_server.py     starts the API from the repository root (used on Colab)
data/raw/           input products (not in git)
outputs/            homographies and metrics.json written by runs (not in git)
```

| Website option | Pipeline | Reads |
|---|---|---|
| OHRC ↔ LRO NAC | `sensors/ohrc_nac.py` (v4) | `data/raw/ohrc/<pair>/` + `data/raw/nac/<pair>/*.IMG` |
| OHRC ↔ TMC-2 | `sensors/ohrc_tmc.py` (v5) | `data/raw/ohrc/<pair>/` (or `LINK.json`) + `data/raw/tmc/<pair>/` |

Both hand the loaded products to the same registration (`registration/runner.py`). The TMC
settings (16-bit stretch, sub-strip refit of the corner model, anti-aliased resampling, 0.5×
fine GSD, adaptive tile size) live in that pipeline's `PROFILE`.

## Backend

Tested with Python 3.13, PyTorch 2.6 (CUDA), transformers 5.17, OpenCV 5.0 and NumPy 2.5.

### Run locally (GPU on this machine)

```bash
pip install -e backend
LOCAL_MODE=true python colab_server.py
```

Connect the website to `http://localhost:5000`.

### Run on Google Colab with ngrok

1. Get an ngrok auth token from [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).
2. Upload or clone this repository to Colab and set the runtime to **T4 GPU**.
3. Run:

```python
from google.colab import drive; drive.mount('/content/drive')
!pip install -q -e backend
%env PROJECT_ROOT=/content/drive/MyDrive/SIH26166
%env NGROK_AUTH_TOKEN=your_token_here
!python colab_server.py
```

Paste the printed `https://xxxx.ngrok-free.app` URL into the website.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PROJECT_ROOT` | this repository | folder that holds `data/` and `outputs/` |
| `LOCAL_MODE` | `false` | `true` skips ngrok |
| `NGROK_AUTH_TOKEN` | *(empty)* | ngrok token; never commit it |
| `PORT` | `5000` | server port |

### Adding data

- **New NAC pair**: put the `.IMG` in `data/raw/nac/pairNN_x/` and the OHRC `.img` + `.xml` in
  `data/raw/ohrc/pairNN_x/`. NAC EDR labels carry no footprint, so add the product's corners and
  GSD to `NAC_FOOTPRINTS` in `backend/trinetra/sensors/ohrc_nac.py`.
- **New TMC pair**: put the TMC `.img` + `.xml` in `data/raw/tmc/pairNN_x/`. For the OHRC, copy
  the files in or add a `LINK.json` pointing at an existing OHRC product. Both labels carry their
  own corners, so nothing else is needed.

### Sun geometry (`sun.json`)

The pair-difficulty score uses the sun elevation and azimuth of both images. OHRC and TMC-2
labels carry them; **LROC NAC EDR labels do not**, so copy them from the LROC product page into
`data/raw/nac/<pair>/sun.json`:

```json
{ "incidence_deg": 81.6, "azimuth_deg": 123.4, "azimuth_convention": "north_cw" }
```

- `elevation_deg` may be given instead of `incidence_deg` (elevation = 90 − incidence).
- `azimuth_convention`: `north_cw` (clockwise from north, as the ISRO labels use), `image_cw`
  (clockwise in the image frame, the ISIS/LROC convention) or `unknown`. The azimuth gap is only
  scored when both images use `north_cw`.
- The website pre-fills its sun fields from this file (or the label) and lets you override them
  per run. A `sun.json` next to a TMC product replaces the values from its label.
- Sun values only feed the difficulty score; they never change the registration.

### Robustness options (website → Run configuration → Advanced)

All are off by default except ECC, which is gated. The rescues are in
`backend/trinetra/matching/rescue.py`, the ECC modes in `registration/ecc.py`.

| Option | What it does | Cost |
|---|---|---|
| ECC: Gated / Legacy / Off | Gated tries ECC from coarse to fine and keeps a result only if the fine-tile ECC score improves. Legacy always keeps it. | Gated adds a few seconds |
| Shadow suppression | For tiles that failed: fills cast shadows from nearby lit ground, then re-matches | failed tiles only |
| Multi-scale | For tiles that failed: re-matches at 1.5× and 2× coarser scale | failed tiles only |
| Intensity polarity | For tiles that failed: matches on brightness, normal and inverted | failed tiles only |
| Crater matching | Crater-anchored matches pooled with the dense matches before the final fit | a few seconds |
| Auto-retry | If the result is not trusted, reruns steps 12–14 with rescues on, at most twice, and keeps the best attempt | up to 3× the fine stage |
| Coarse matcher | SuperGlue or LightGlue for steps 8 and 11. LightGlue downloads on first use. | — |

**Trusted** requires every check to pass: ≥ 15 inliers, ≥ 5% coverage, spatial held-out RMSE
≤ 5 px, and NMI ≥ 1.5× the misaligned baseline. The older count-only flag is kept as
`trusted_legacy` in `metrics.json`.

### Tests

```bash
cd backend
python -m unittest
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, paste the backend URL, click **Connect**, choose the sensor pair and
image pair, and click **Run pipeline**. `npm run build` writes a static site to `frontend/dist/`.

## Troubleshooting

| Issue | Fix |
|---|---|
| "Connection failed" | Make sure the Colab cell is still running. The ngrok URL changes on every restart. |
| "No NAC .IMG found" | Check that `PROJECT_ROOT` matches your Drive folder structure. |
| CORS errors in the console | The server allows all origins. On the ngrok free tier, open the URL once and accept the warning page. |
| GPU not available | Runtime → Change runtime type → T4 GPU. |
| Slow first run | SuperGlue downloads its weights on first use (~300 MB). |
| Result says LOW CONFIDENCE | Open **Trust checks** under the metric cards to see which check failed. Try Auto-retry, or switch on the rescues one at a time. |
