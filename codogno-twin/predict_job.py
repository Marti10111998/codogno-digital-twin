"""
predict_job.py
-------------------------------------------------------------
The "always-on, for free" worker.

Instead of keeping a server alive 24/7, GitHub Actions runs this
script on a schedule (e.g. every hour). It:
  1. pulls current + forecast weather from OpenWeather for Codogno
  2. runs the LightGBM pollution models
  3. writes docs/predictions.json

Your static frontend (GitHub Pages / Netlify) just fetches that
JSON. No live server, no RAM limits, no cold starts -> truly free.

Env vars required:
  OPENWEATHER_API_KEY   your OpenWeather key
Optional:
  CITY                  default "Codogno"
  MODELS_DIR            where the .pkl files are (default ".")
  OUT_PATH              output json (default "docs/predictions.json")
  PRECIP_FEATURE_KIND   PRESSURE_HPA (default) or RAIN_MM
-------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import requests

CITY = os.getenv("CITY", "Codogno")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "."))
OUT_PATH = Path(os.getenv("OUT_PATH", "docs/predictions.json"))
API_KEY = os.environ["OPENWEATHER_API_KEY"]  # fail loudly if missing
PRECIP_FEATURE_KIND = os.getenv("PRECIP_FEATURE_KIND", "RAIN_MM").upper()

TARGETS = ["O3", "CO", "NO2", "NO", "NH3"]
# Order must match train_lightgbm.py: 4 weather + 4 time features.
FEATURES = ["Temperature", "Humidity", "Precipitation", "WindSpeed_m_s",
            "Hour", "Month", "Weekday", "DayOfYear"]


def load_models():
    return {t: joblib.load(MODELS_DIR / f"rf_{t}_model.pkl") for t in TARGETS}


def precip(entry: dict) -> float:
    if PRECIP_FEATURE_KIND == "RAIN_MM":
        return float((entry.get("rain", {}) or {}).get("1h", 0.0) or 0.0)
    return float(entry.get("main", {}).get("pressure", 1013))


def feature_row(entry: dict) -> np.ndarray:
    # OpenWeather 'dt' is unix seconds; derive the time features from it.
    dt = datetime.fromtimestamp(int(entry["dt"]), tz=timezone.utc) if "dt" in entry \
        else datetime.now(timezone.utc)
    return np.array([[
        float(entry["main"]["temp"]),
        float(entry["main"]["humidity"]),
        precip(entry),
        float(entry["wind"]["speed"]),
        dt.hour,
        dt.month,
        dt.weekday(),
        dt.timetuple().tm_yday,
    ]], dtype=float)


def predict(models, X: np.ndarray) -> dict:
    return {t: round(float(m.predict(X)[0]), 3) for t, m in models.items()}


def get_json(url: str) -> dict:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def main() -> None:
    models = load_models()

    base = "https://api.openweathermap.org/data/2.5"
    now = get_json(f"{base}/weather?q={CITY}&appid={API_KEY}&units=metric")
    fc = get_json(f"{base}/forecast?q={CITY}&appid={API_KEY}&units=metric")

    current = {
        "weather": {
            "temp": now["main"]["temp"],
            "humidity": now["main"]["humidity"],
            "pressure": now["main"].get("pressure"),
            "wind_m_s": now["wind"]["speed"],
        },
        "pollution": predict(models, feature_row(now)),
    }

    forecast = []
    for e in fc.get("list", [])[:16]:  # ~48h of 3-hourly steps
        forecast.append({
            "dt_txt": e.get("dt_txt"),
            "pollution": predict(models, feature_row(e)),
        })

    payload = {
        "city": CITY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_ts": int(time.time() * 1000),
        "current": current,
        "forecast": forecast,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}  ({len(forecast)} forecast steps)")


if __name__ == "__main__":
    main()
