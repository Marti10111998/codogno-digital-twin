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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pollution_api2:app", host="0.0.0.0", port=8000, reload=True)
