# Codogno Digital Twin — Free, Always-On Setup

Follow these in order. Part A shrinks your models (the real fix). Part B puts
the pollution predictions online for free, forever. Part C is optional (a live
API). Part D is the cleaned backend if you'd rather keep a server.

**I already trained the models on your `historical.csv`** — they're in the
`models/` folder (5 files, ~1.9 MB total, down from ~4 GB). You can skip
straight to Part B and drop them into your repo. Only re-run Part A if you get
new data.

Files in this folder:

| File | What it is |
|------|------------|
| `models/` | **Trained, ready-to-use** LightGBM models (`rf_O3_model.pkl`, …) |
| `train_lightgbm.py` | Re-trains the 5 models from a CSV (only needed for new data) |
| `predict_job.py` | The worker GitHub runs hourly → writes `predictions.json` |
| `predict.yml` | GitHub Actions schedule (goes in `.github/workflows/`) |
| `pollution_api.py` | Cleaned, bug-fixed FastAPI (optional live server) |
| `frontend_snippet.html` | JS to read `predictions.json` from your site |
| `requirements.txt` | Python dependencies |

### What changed from your originals

- **Size:** ~4 GB → **1.9 MB total**. This is what makes free hosting possible.
- **Features (8):** the 4 weather columns **plus 4 time features** (Hour, Month,
  Weekday, DayOfYear) derived from the timestamp. Time is free at prediction
  time and roughly doubled accuracy for the traffic-driven pollutants.
- **Precipitation** in your data is **rainfall in mm** (not pressure), so the
  setting is `PRECIP_FEATURE_KIND=RAIN_MM` everywhere. Already set as default.
- **Radiation** was tested and added almost nothing (~+0.02 R²) and isn't
  available from live weather, so it's left out.

Accuracy on a 20% held-out test set (R² = fraction of variance explained):

| Pollutant | R² (4 weather only) | R² (with time) | Test MAE |
|-----------|--------------------|----------------|----------|
| O3 | 0.825 | **0.892** | 9.96 |
| NO | 0.297 | **0.615** | 21.7 |
| NO2 | 0.315 | **0.608** | 8.63 |
| CO | 0.330 | **0.505** | 0.17 |
| NH3 | 0.128 | **0.400** | 11.3 |

O3 is strong. NO/NO2/CO are solid for weather-only inputs. NH3 is the weakest —
if you want it better, the biggest lever is adding a **traffic or day-type
signal**; tell me and I'll wire it in.

---

## Part A — (Re)train the models  *(optional — already done for you)*

Only needed if you get new data.

1. Install deps once:
   ```bash
   pip install pandas numpy scikit-learn lightgbm joblib requests
   ```
2. Retrain from your CSV:
   ```bash
   python train_lightgbm.py --csv /path/to/historical.csv --out ./models
   ```
3. You get `models/rf_O3_model.pkl`, `rf_CO_model.pkl`, etc. Same filenames and
   feature contract as the included ones, so they stay drop-in compatible.

> Your CSV needs the `DateTime` column (used to build the time features) plus
> Temperature, Humidity, Precipitation, WindSpeed_m_s and the 5 pollutant
> columns. Precipitation is treated as rainfall mm (`RAIN_MM`).

---

## Part B — Free, always-on predictions (recommended)

No server. GitHub Actions runs the models hourly and commits a JSON file your
static site reads.

1. Create a **new GitHub repo** (e.g. `codogno-twin`) and add:
   ```
   predict_job.py
   requirements.txt
   models/                       ← the small .pkl files from Part A
   .github/workflows/predict.yml ← the predict.yml file, renamed into this path
   docs/                         ← create it; predictions.json lands here
   ```
2. Get a free **OpenWeather API key** at openweathermap.org (free tier is fine).
3. In the repo: **Settings → Secrets and variables → Actions → New repository
   secret**. Name it `OPENWEATHER_API_KEY`, paste your key.
4. Go to the **Actions** tab, open “Codogno pollution prediction”, click **Run
   workflow** to test it now. After ~1 min you should see `docs/predictions.json`
   appear in the repo.
5. From then on it runs **every hour automatically**, free.
6. Point your frontend at the JSON: copy `frontend_snippet.html` into your
   `index.html`, replace `YOUR_USER/YOUR_REPO`, and wire the values into your
   Cesium overlays / Chart.js where the TODO is.
7. (Optional) Host the frontend free too: **Settings → Pages → deploy from
   `docs/`**, or drag the folder into Netlify. Both are free and always on.

That's the whole "runs 24/7 for free" answer — GitHub keeps the schedule alive,
so there is no server to pay for or keep awake.

> Note: GitHub disables scheduled workflows after 60 days of **zero repo
> activity**. The hourly commit counts as activity, so this keeps itself alive.

---

## Part C — (Optional) Keep a live API instead

If you specifically want a live `/predict` endpoint (e.g. for what-if sliders):

1. With the small LightGBM models, free hosts now work. Easiest:
   - **Render / Fly.io free tier** — fine now that models are a few MB.
   - **Hugging Face Spaces** (16 GB RAM, 50 GB disk) — sleeps after 48h idle.
   - **Oracle Cloud Always Free VM** (4 ARM cores, 24 GB RAM) — truly always on,
     but ARM capacity is often “out of stock”; keep retrying region/zone.
2. Run locally to test first:
   ```bash
   pip install -r requirements.txt
   export OPENWEATHER_API_KEY=your_key
   export MODELS_DIR=./models
   export ALLOWED_ORIGINS="https://your-frontend-domain"
   uvicorn pollution_api:app --host 0.0.0.0 --port 8000
   ```
3. Test: open `http://localhost:8000/health` and `http://localhost:8000/docs`.

---

## Part D — What I fixed in the backend

`pollution_api.py` replaces `pollution_api2.py`. Bugs removed:

- **Duplicate definitions** — you had `IngestPayload`, `/ingest`, `/history`,
  `DB_PATH` and `db()` defined twice; the second silently overrode the first and
  used a *different table* (`points` vs `readings`). Now one schema (`readings`).
- **Missing `import time`** — the first `/history` called `time.time()` with no
  import and would crash. Fixed.
- **CORS `*`** — now driven by `ALLOWED_ORIGINS` env so you can lock it to your
  domain in production.
- **Added `/predict-batch`** — send a whole timeline of weather rows, get all
  predictions in one request (faster charts).

---

## Suggested next steps (from our discussion)

Once this is live, the high-value additions are: a **calibration loop**
(prediction vs. real sensor, track drift), **what-if sliders** (perturb wind/
temp/traffic and re-predict), **spatial dispersion** (one reading → a field over
the town), and new targets using the *same* pipeline — **fog** (very predictable
in the Po Valley), **UHI intensity**, **noise**, and **river/flood level**.
Tell me which and I'll build it next.
