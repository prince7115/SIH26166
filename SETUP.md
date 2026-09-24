# SIH26166 — Setup Guide

## Architecture

```
┌──────────────────┐      ngrok tunnel        ┌──────────────────────┐
│  Frontend (here) │ ◄═══════════════════►    │  Google Colab (GPU)  │
│  Vite dev server │    REST API + SSE        │  Flask + pipeline    │
│  localhost:5173   │                          │  ngrok public URL    │
└──────────────────┘                          └──────────────────────┘
```

---

## Part 1: Backend Setup

The backend is the `backend/` package; `colab_server.py` is just its launcher. The website
offers two sensor pairs, each handled by its own pipeline module:

| Website option | Module | Source | Data it reads |
|---|---|---|---|
| OHRC ↔ LRO NAC | `backend/pipelines/ohrc_nac.py` | v4 server pipeline | `data/raw/ohrc/<pair>/` + `data/raw/nac/<pair>/*.IMG` |
| OHRC ↔ TMC-2 | `backend/pipelines/ohrc_tmc.py` | `sih_final_v5_ohrc_tmc_1.ipynb` | `data/raw/ohrc/<pair>/` (or `LINK.json`) + `data/raw/tmc/<pair>/` |

Both modules hand the loaded products to the shared engine (`backend/engine.py`, steps 3–18).
TMC-specific settings: 16-bit stretch, a sub-strip refit of the corner model, anti-aliased
resampling, 0.5× fine GSD and adaptive tile size. They are set in the module's `PROFILE`.

### Option A — run locally (GPU on this machine)

```bash
pip install flask flask-cors pyngrok planetaryimage pvl scikit-image scikit-learn transformers torch torchvision matplotlib opencv-python
LOCAL_MODE=true python colab_server.py
```

Connect the website to `http://localhost:5000`.

### Option B — Google Colab + ngrok

1. Get an ngrok auth token from [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).
2. Upload or clone this folder to Colab, and set the runtime to **T4 GPU**.
3. Run:

```python
!pip install flask flask-cors pyngrok planetaryimage pvl scikit-image scikit-learn transformers
from google.colab import drive; drive.mount('/content/drive')
%env PROJECT_ROOT=/content/drive/MyDrive/SIH26166
%env NGROK_AUTH_TOKEN=your_token_here
!python colab_server.py
```

Copy the printed `https://xxxx.ngrok-free.app` URL into the website.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PROJECT_ROOT` | this repo | folder that holds `data/` and `outputs/` |
| `LOCAL_MODE` | `false` | `true` skips ngrok |
| `NGROK_AUTH_TOKEN` | *(empty)* | ngrok token; never commit it |
| `PORT` | `5000` | server port |

### Adding data

- **New NAC pair**: put the `.IMG` in `data/raw/nac/pairNN_x/` and the OHRC `.img` + `.xml` in
  `data/raw/ohrc/pairNN_x/`. Add the NAC corners and GSD to `KNOWN_NAC_CORNERS` / `KNOWN_NAC_GSD`
  in `ohrc_nac.py`.
- **New TMC pair**: put the TMC `.img` + `.xml` in `data/raw/tmc/pairNN_x/`. For the OHRC, either
  copy the files in or add a `LINK.json` pointing at an existing OHRC product. Nothing else is
  needed, because both labels carry their own corners.

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
- Sun values only feed the difficulty score. They never change the registration itself.

### Robustness options (website → Run configuration → Advanced)

All of these are off by default, so a default run matches the previous pipeline. The one
exception is ECC, which is now **gated**. The code is in `backend/robust.py`.

| Option | What it does | Cost |
|---|---|---|
| ECC: Gated / Legacy / Off | Gated tries ECC from coarse to fine and keeps a result only if the fine-tile ECC score improves. Legacy is the old always-keep behaviour. | Gated adds ~seconds |
| Shadow suppression | For tiles that failed: fills cast shadows from nearby lit ground, then re-matches | only failed tiles |
| Multi-scale | For tiles that failed: re-matches at 1.5× and 2× coarser scale | only failed tiles |
| Intensity polarity | For tiles that failed: matches on brightness, normal and inverted (lit/shadowed slopes swapped) | only failed tiles |
| Crater matching | Crater-anchored matches pooled with the dense matches before the final fit | a few seconds |
| Auto-retry | If the result is not trusted, reruns steps 12–14 with rescues on, at most twice, and keeps the best attempt | up to 3× the fine stage |
| Coarse matcher | SuperGlue or LightGlue for steps 8 and 11. LightGlue downloads on first use. | — |

**Trusted** now requires every check to pass: ≥ 15 inliers, ≥ 5% coverage, spatial held-out
RMSE ≤ 5 px, and NMI ≥ 1.5× the misaligned baseline. The old count-only flag is kept as
`trusted_legacy` in `metrics.json`.

---

## Part 2: Frontend Setup

### 1. Install Dependencies
```bash
cd c:\ddrive\SIH26Frontend
npm install
```

### 2. Start Dev Server
```bash
npm run dev
```

Opens at `http://localhost:5173`

### 3. Connect to Colab
1. Paste the ngrok URL from Colab into the connection panel
2. Click **Connect** → should show "Connected · GPU ✓"
3. Choose the **Sensor Pair** (OHRC ↔ LRO NAC or OHRC ↔ TMC-2), then the image pair
4. Click **Run Pipeline**

---

## Troubleshooting

| Issue | Fix |
|---|---|
| "Connection failed" | Make sure the Colab cell is still running. The ngrok URL changes if you restart. |
| "No NAC .IMG found" | Verify `PROJECT_ROOT` path matches your Drive folder structure. |
| CORS errors in console | The server has CORS enabled. If using ngrok free tier, click the ngrok warning page first. |
| GPU not available | Go to Runtime → Change runtime type → Select T4 GPU. |
| Slow first run | SuperGlue model downloads on first use (~300 MB). Subsequent runs are faster. |
| Result says LOW CONFIDENCE | Open **Trust checks** under the metric cards to see which check failed. Try Auto-retry, or switch on the rescues individually. |

## File Structure

```
SIH26Frontend/
├── index.html / index.css / app.js    ← website (Vite)
├── colab_server.py                    ← server launcher
├── backend/
│   ├── app.py                         ← Flask API + SSE
│   ├── config.py                      ← paths / env vars
│   ├── common.py                      ← shared geometry, preprocessing, matching, PDS4 loader
│   ├── engine.py                      ← shared registration steps 3–18
│   ├── robust.py                      ← gated ECC, held-out/trust, difficulty, tile rescues, craters, retries
│   └── pipelines/
│       ├── ohrc_nac.py                ← OHRC ↔ LRO NAC (v4)
│       └── ohrc_tmc.py                ← OHRC ↔ TMC-2 (v5-tmc)
├── ohrc_nac.ipynb                     ← older NAC notebook (reference only)
├── sih_final_v5_ohrc_tmc_1.ipynb      ← TMC notebook (source of ohrc_tmc.py)
├── data/raw/{ohrc,nac,tmc}/<pair>/    ← input products
└── outputs/                           ← <pair>_v4_* (NAC), <pair>_v5_* (TMC)
```
