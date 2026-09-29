from contextlib import asynccontextmanager
from pathlib import Path

import joblib,  pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as mae

from fastapi import FastAPI
from pydantic import BaseModel, Field

class Features(BaseModel):
    season: int = Field(ge=1, le=4)
    mnth: int = Field(ge=1, le=12)
    hr: int = Field(ge=0, le=23)
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6)
    workingday: int = Field(ge=0, le=1)
    weathersit: int = Field(ge=1, le=4)
    temp: float; atemp: float; hum: float; windspeed: float

STATE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load("model/hgb.joblib")
    STATE["model"], STATE["meta"] = bundle["model"], bundle["meta"]
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE, **STATE.get("meta", {})}

@app.post("/predict")
def predict(x: Features):
    y = STATE["model"].predict(pd.DataFrame([x.model_dump()]))[0]
    return {"cnt": max(0.0, float(y))}