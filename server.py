"""NOVIX AI service (V3) - FastAPI wrapper around NOVIXRawStream.

One stream per device. Baselines are learned from the first CALIB_READINGS
readings after the session starts (clean air required), unless the caller or
env vars supply them. elapsed_seconds is a monotonic per-device clock.
"""
import os, time, threading, statistics, warnings, traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

warnings.filterwarnings("ignore")
from fastapi import FastAPI
from pydantic import BaseModel
from src.streaming_raw import NOVIXRawStream

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "models" / "lightgbm_hybrid_final_v3.joblib"

CALIB_READINGS = int(os.getenv("NOVIX_CALIB_READINGS", "30"))   # ~30 s at 1 Hz
SESSION_TIMEOUT = float(os.getenv("NOVIX_SESSION_TIMEOUT", "15"))  # silence -> new session
ENV_B2 = os.getenv("NOVIX_BASELINE_MQ2")
ENV_B6 = os.getenv("NOVIX_BASELINE_MQ6")

app = FastAPI(title="NOVIX AI V3")
_lock = threading.Lock()


class Session:
    def __init__(self):
        self.stream = NOVIXRawStream(BUNDLE)
        self.start = time.monotonic()
        self.last_seen = self.start
        self.calib: List[tuple] = []
        self.b2: Optional[float] = float(ENV_B2) if ENV_B2 else None
        self.b6: Optional[float] = float(ENV_B6) if ENV_B6 else None

    def elapsed(self):
        return time.monotonic() - self.start


sessions: Dict[str, Session] = {}


class Reading(BaseModel):
    device_id: str = "default"
    mq2: float
    mq6: float
    temp: float
    humidity: float
    flame: int = 0
    baseline_mq2: Optional[float] = None
    baseline_mq6: Optional[float] = None
    elapsed_seconds: Optional[float] = None   # testing/replay only; normally server clock


def _flame_only(r: Reading, reason: str):
    return {"status": "ready", "label": "FIRE", "risk_state": "HAZARD", "alarm": True,
            "hazard_confirmed": True, "safety_override": True, "hazard_probability": 1.0,
            "reason": reason}


@app.get("/health")
def health():
    return {"ok": True, "model": BUNDLE.name, "devices": list(sessions)}


@app.post("/reset/{device_id}")
def reset(device_id: str):
    with _lock:
        sessions.pop(device_id, None)
    return {"ok": True}


def _predict_inner(r: Reading):
    with _lock:
        s = sessions.get(r.device_id)
        now = time.monotonic()
        if s is None or (r.elapsed_seconds is None and now - s.last_seen > SESSION_TIMEOUT):
            s = sessions[r.device_id] = Session()
        s.last_seen = now

        b2 = r.baseline_mq2 or s.b2
        b6 = r.baseline_mq6 or s.b6
        flame = 1 if int(r.flame) >= 1 else 0

        if b2 is None or b6 is None:
            if flame:   # flame never waits for calibration; don't learn baseline from a fire
                return {**_flame_only(r, "IMMEDIATE_RAW_FLAME_OVERRIDE"), "device_id": r.device_id}
            s.calib.append((r.mq2, r.mq6))
            if len(s.calib) < CALIB_READINGS:
                return {"status": "calibrating", "label": "CALIBRATING", "risk_state": "UNKNOWN",
                        "alarm": False, "hazard_confirmed": False, "safety_override": False,
                        "reason": "LEARNING_CLEAN_AIR_BASELINE",
                        "calibration_progress": f"{len(s.calib)}/{CALIB_READINGS}",
                        "device_id": r.device_id}
            s.b2 = max(1.0, statistics.median(x[0] for x in s.calib))
            s.b6 = max(1.0, statistics.median(x[1] for x in s.calib))
            b2, b6 = s.b2, s.b6
            s.stream.reset(); s.start = time.monotonic()

        t = r.elapsed_seconds if r.elapsed_seconds is not None else s.elapsed()
        try:
            # flame=1 is handled inside predict_raw: immediate FIRE, history preserved.
            out = s.stream.predict_raw({"mq2": r.mq2, "mq6": r.mq6, "temp": r.temp,
                                        "humidity": r.humidity, "flame": flame,
                                        "baseline_mq2": b2, "baseline_mq6": b6}, t)
        except ValueError as e:
            s.stream.reset()
            if flame:
                return {**_flame_only(r, "IMMEDIATE_RAW_FLAME_OVERRIDE"), "device_id": r.device_id}
            return {"status": "error", "label": "ERROR", "risk_state": "UNKNOWN", "alarm": False,
                    "reason": str(e), "device_id": r.device_id}
        out["device_id"] = r.device_id
        out["baseline_mq2"], out["baseline_mq6"] = b2, b6
        out.pop("features", None)
        return out


@app.post("/predict")
def predict(r: Reading):
    try:
        return _predict_inner(r)
    except Exception as e:  # never 500: report the cause so the backend can log it
        traceback.print_exc()
        return {"status": "error", "label": "ERROR", "risk_state": "UNKNOWN", "alarm": False,
                "hazard_confirmed": False, "safety_override": False,
                "reason": f"{type(e).__name__}: {e}", "device_id": r.device_id}