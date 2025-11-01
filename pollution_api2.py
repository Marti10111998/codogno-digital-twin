# pollution_api2.py
# -------------------------------------------------------------
# FastAPI that serves predictions from RF models downloaded
# automatically from Google Drive or Dropbox.
# -------------------------------------------------------------

from __future__ import annotations
import os, joblib, logging, requests
from pathlib import Path
from typing import Dict, List
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# --------------------------
# Config
# --------------------------
TARGETS: List[str] = ["O3", "CO", "NO2", "NO", "NH3"]
FEATURES: List[str] = ["Temperature", "Humidity", "Precipitation", "WindSpeed_m_s"]
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ❶ Put your direct download links here
MODEL_URLS = {
    "rf_CO_model.pkl":  "https://drive.google.com/uc?export=download&id=1ukLqBo8aMsf8Z9f1-bQlHI5C3fQ6JedU",
    "rf_NH3_model.pkl": "https://drive.google.com/uc?export=download&id=173i4y5fyfb3sVe663CYVbliCAcJAzUfu",
    "rf_NO_model.pkl":  "https://drive.google.com/uc?export=download&id=13HOvPrJk7a4oJnXbYr4jkB9txIFd0g4E",
    "rf_NO2_model.pkl": "https://drive.google.com/uc?export=download&id=1T1Yh17VYC_vcjwN0RuteZRpUNi1uANL9",
    "rf_O3_model.pkl":  "https://drive.google.com/uc?export=download&id=1q7oKyiW73EtGmtzv1EZDJTw2GRaWjutV",  
}


OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_KEY")
PRECIP_FEATURE_KIND = os.getenv("PRECIP_FEATURE_KIND", "PRESSURE_HPA").upper()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("pollution_api2")

# --------------------------
# Download helper
# --------------------------
def download_model(url: str, filename: str) -> Path:
    """Download model from URL only if not present locally."""
    path = MODEL_DIR / filename
    if path.exists():
        return path
    log.info("⬇️  Downloading %s ...", filename)
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        log.info("✅ Saved %s", filename)
    except Exception as e:
        log.error("❌ Failed to download %s: %s", filename, e)
        raise
    return path

# --------------------------
# Model loading
# --------------------------
def load_models() -> Dict[str, object]:
    models = {}
    for target in TARGETS:
        fname = f"rf_{target}_model.pkl"
        url = MODEL_URLS.get(fname)
        if not url:
            raise FileNotFoundError(f"No URL for {fname}. Please set MODEL_URLS.")
        pkl_path = download_model(url, fname)
        models[target] = joblib.load(pkl_path)
        log.info("Loaded %s", fname)
    return models

MODELS = load_models()

# --------------------------
# FastAPI setup
# --------------------------
app = FastAPI(title="Codogno Pollution API (Cloud Models)", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --------------------------
# Schemas
# --------------------------
class Weather(BaseModel):
    Temperature: float = Field(..., description="Air temperature in °C")
    Humidity: float = Field(..., ge=0, le=100, description="Relative humidity %")
    Precipitation: float = Field(..., description="Pressure hPa or rain mm")
    WindSpeed_m_s: float = Field(..., ge=0, description="Wind speed m/s")

    @field_validator("Temperature", "Humidity", "Precipitation", "WindSpeed_m_s")
    @classmethod
    def finite(cls, v: float) -> float:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            raise ValueError("feature must be a finite number")
        return float(v)

class Prediction(BaseModel):
    O3: float
    CO: float
    NO2: float
    NO: float
    NH3: float

# --------------------------
# Core logic
# --------------------------
def features_to_row(w: Weather) -> np.ndarray:
    return np.array([[w.Temperature, w.Humidity, w.Precipitation, w.WindSpeed_m_s]], dtype=float)

def predict_row(X: np.ndarray) -> Dict[str, float]:
    out = {}
    for t, model in MODELS.items():
        out[t] = float(model.predict(X)[0])
    return out

# --------------------------
# Routes
# --------------------------
@app.get("/health")
def health():
    return {"ok": True, "targets": TARGETS, "features": FEATURES}

@app.post("/predict", response_model=Prediction)
def predict_now(w: Weather):
    X = features_to_row(w)
    return predict_row(X)

# --------------------------
# Entry point for local run
# --------------------------

# ==== STORAGE & CRON: add this block to pollution_api2.py ====
import sqlite3, time

# Persistent DB path (on Render we'll point this to /var/data/data.db)
DB_PATH = os.getenv("DB_PATH", "data.db")

# Optional weather for /cron
LAT = float(os.getenv("LAT", "45.165"))
LON = float(os.getenv("LON", "9.703"))
OWM_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# The same pollutant names your models serve:
POLLUTANTS = TARGETS  # ["O3","CO","NO2","NO","NH3"]

def _db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def _init_db():
    con = _db()
    con.execute("CREATE TABLE IF NOT EXISTS readings (ts INTEGER PRIMARY KEY)")
    # add columns if missing
    have = {r["name"] for r in con.execute("PRAGMA table_info(readings)").fetchall()}
    for p in POLLUTANTS:
        if p not in have:
            con.execute(f"ALTER TABLE readings ADD COLUMN {p} REAL")
    con.execute("CREATE TABLE IF NOT EXISTS devices (device TEXT PRIMARY KEY, last_seen_ts INTEGER)")
    con.commit(); con.close()

_init_db()

def _upsert_row(ts:int, kv:dict):
    con = _db()
    con.execute("INSERT OR IGNORE INTO readings (ts) VALUES (?)", (ts,))
    sets, vals = [], []
    for k,v in kv.items():
        if k in POLLUTANTS and isinstance(v,(int,float)):
            sets.append(f"{k}=?"); vals.append(float(v))
    if sets:
        vals.append(ts)
        con.execute(f"UPDATE readings SET {', '.join(sets)} WHERE ts=?", vals)
    con.commit(); con.close()

def _row_to_obj(row: sqlite3.Row) -> dict:
    o = {"ts": int(row["ts"])}
    for p in POLLUTANTS:
        if p in row.keys() and row[p] is not None:
            o[p] = float(row[p])
    return o

@app.post("/ingest")
def ingest(payload: dict):
    """
    Body: {"device":"abc","rows":[{"ts":<ms>,"pollutant":"O3","value":88.2}, ...]}
    This matches your front-end sync format.
    """
    device = payload.get("device") or "anon"
    rows = payload.get("rows") or []
    now = int(time.time()*1000)
    con = _db()
    con.execute("INSERT OR REPLACE INTO devices (device, last_seen_ts) VALUES (?,?)", (device, now))
    con.commit(); con.close()
    for r in rows:
        ts  = int(r.get("ts", now))
        pol = r.get("pollutant")
        val = r.get("value")
        if pol in POLLUTANTS and isinstance(val,(int,float)):
            _upsert_row(ts, {pol: float(val)})
    return {"ok": True}

@app.get("/history")
def history(days: int = 7):
    since = int(time.time()*1000) - int(days)*24*3600*1000
    con = _db()
    cur = con.execute("SELECT * FROM readings WHERE ts>=? ORDER BY ts ASC", (since,))
    out = [_row_to_obj(r) for r in cur.fetchall()]
    con.close()
    return out

@app.get("/latest")
def latest():
    con = _db()
    cur = con.execute("SELECT * FROM readings ORDER BY ts DESC LIMIT 1")
    row = cur.fetchone()
    con.close()
    return _row_to_obj(row) if row else {}

def _fetch_weather():
    # If no key, use safe defaults; still lets cron store rows
    if not OWM_KEY:
        return dict(Temperature=28.0, Humidity=50.0, Precipitation=0.0, WindSpeed_m_s=1.5)
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&units=metric&appid={OWM_KEY}"
    r = requests.get(url, timeout=10); r.raise_for_status()
    j = r.json()
    return dict(
        Temperature     = float(j.get("main",{}).get("temp", 28.0)),
        Humidity        = float(j.get("main",{}).get("humidity", 50.0)),
        Precipitation   = float((j.get("rain",{}) or {}).get("1h", 0.0)),
        WindSpeed_m_s   = float(j.get("wind",{}).get("speed", 1.5))
    )

@app.get("/cron")
def cron():
    # Called by a scheduler every 10 minutes. Predicts + stores a row.
    w = _fetch_weather()
    X = features_to_row(Weather(**w))
    preds = predict_row(X)
    ts = int(time.time()*1000)
    _upsert_row(ts, preds)
    return {"stored": {"ts": ts, **preds}}
# ==== END STORAGE & CRON block ====


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pollution_api2:app", host="0.0.0.0", port=8000, reload=True)
