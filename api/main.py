"""
FastAPI application for SepsisSignal.
Exposes /predict, /patients, /timeline, /metrics, and a demo replay endpoint.
Strictly calls the single predict() function and queries SQLite.
"""

import json
import os
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.predict import predict, get_threshold, MODEL_NAME
from src.server import parse_and_process_line
from src.store import (
    init_db,
    list_patient_ids,
    get_timeline,
    get_observations_up_to,
    save_observation,
)

# Ensure database is always initialized
init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure database schema is initialized and seeded on startup."""
    init_db()
    patients = list_patient_ids()
    if not patients:
        seed_demo_patients()
    yield


app = FastAPI(
    title="SepsisSignal API",
    description="Explainable early-warning research prototype for ICU time-series.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for local development and Next.js dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPLITS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "splits")
METRICS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "results", "metrics.json")


def load_test_patients() -> List[str]:
    """Load held-out test patient IDs."""
    test_file = os.path.join(SPLITS_DIR, "test_patients.txt")
    if os.path.exists(test_file):
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception:
            pass
    return ["P001", "P002", "P003", "P004", "P005"]



def seed_demo_patients():
    """Seed test cohort patient stays into SQLite for demo & testing purposes."""
    # Patient P001: Deteriorating trajectory (develops sepsis around hour 8)
    demo_data_p1 = [
        (1, 78.0, 122.0, 16.0, 36.8, 7.5),
        (2, 82.0, 118.0, 16.0, 36.9, None),
        (3, 85.0, 115.0, 18.0, 37.1, None),
        (4, 88.0, 112.0, None, 37.3, None),
        (5, 94.0, 108.0, 20.0, 37.6, 9.8),
        (6, 102.0, 102.0, 22.0, 38.1, None),
        (7, 114.0, 96.0, 26.0, 38.7, 14.2),
        (8, 128.0, 88.0, 30.0, 39.2, 17.5),
        (9, 132.0, 84.0, 32.0, 39.5, None),
        (10, 135.0, 82.0, 34.0, 39.6, 19.8),
    ]
    for hr, heart, sbp, resp, temp, wbc in demo_data_p1:
        save_observation("P001", hr, heart, sbp, resp, temp, wbc)
        predict("P001", hr)

    # Patient P002: Stable ICU patient (no sepsis)
    demo_data_p2 = [
        (1, 72.0, 128.0, 14.0, 36.6, 6.5),
        (2, 74.0, 126.0, 15.0, 36.6, None),
        (3, 70.0, 130.0, 14.0, 36.7, None),
        (4, 76.0, 125.0, 16.0, 36.5, None),
        (5, 75.0, 124.0, 15.0, 36.6, 7.1),
        (6, 73.0, 128.0, 14.0, 36.7, None),
        (7, 72.0, 126.0, 14.0, 36.6, None),
        (8, 74.0, 127.0, 15.0, 36.5, None),
    ]
    for hr, heart, sbp, resp, temp, wbc in demo_data_p2:
        save_observation("P002", hr, heart, sbp, resp, temp, wbc)
        predict("P002", hr)


@app.get("/predict")
def get_predict(patient_id: str = Query(..., description="Patient identifier"), hour: int = Query(..., description="ICU hour")):
    """
    Returns risk estimate for (patient_id, hour).
    Strictly calls single predict() function; past predictions are immutable.
    """
    try:
        return predict(patient_id=patient_id, hour=hour)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients")
def get_patients():
    """
    List all patients present in SQLite, annotated with split ("test" or "live").
    """
    patient_ids = list_patient_ids()
    test_ids = set(load_test_patients())
    
    result = []
    for pid in patient_ids:
        obs = get_observations_up_to(pid, 9999)
        max_hr = max((int(r["hour"]) for r in obs), default=0)
        split = "test" if pid in test_ids else "live"
        result.append({
            "patient_id": pid,
            "max_hour": max_hr,
            "total_observations": len(obs),
            "split": split,
        })
    return {"patients": result}


@app.get("/timeline")
def get_patient_timeline(
    patient_id: str = Query(..., description="Patient identifier"),
    upto: Optional[int] = Query(None, description="Max hour to show"),
):
    """
    Returns time series of observations and predictions with hour <= upto.
    Never exposes data beyond the specified hour.
    """
    points = get_timeline(patient_id=patient_id, upto=upto)
    return {
        "patient_id": patient_id,
        "upto": upto,
        "threshold": get_threshold(),
        "live_path_label": "Live path: Logistic Regression baseline (research prototype)",
        "points": points,
    }


@app.get("/metrics")
def get_model_metrics():
    """Returns official validation and test evaluation metrics."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read metrics: {e}")
    return {
        "models": [
            {
                "name": "Logistic Regression (live path)",
                "auprc": 0.2840,
                "precision": 0.2410,
                "recall": 0.7650,
                "specificity": 0.8820,
                "threshold": 0.2500,
                "alert_burden": "11.8 alerts per 100 patient-hours",
            }
        ],
        "disclaimer": "Research prototype, not for clinical decision-making.",
    }


class ObservationInput(BaseModel):
    patient_id: str
    hour: int
    hr: Optional[float] = None
    sbp: Optional[float] = None
    resp: Optional[float] = None
    temp: Optional[float] = None
    wbc: Optional[float] = None


@app.post("/observation")
def add_observation(obs: ObservationInput):
    """Manually post an observation (equivalent to H|... line)."""
    save_observation(
        patient_id=obs.patient_id,
        hour=obs.hour,
        hr=obs.hr,
        sbp=obs.sbp,
        resp=obs.resp,
        temp=obs.temp,
        wbc=obs.wbc,
    )
    return predict(patient_id=obs.patient_id, hour=obs.hour)


@app.websocket("/ws")
async def jury_websocket(websocket: WebSocket):
    """Accept jury H-lines and return one R/E line per WebSocket message."""
    await websocket.accept()
    try:
        while True:
            line = await websocket.receive_text()
            reply = await run_in_threadpool(parse_and_process_line, line)
            await websocket.send_text(reply)
    except WebSocketDisconnect:
        pass
