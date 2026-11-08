"""Calibrated inference from a complete engineered feature row."""
import joblib
import numpy as np
import pandas as pd
from .config import FEATURES, N_CONSECUTIVE, MAX_GAP_SECONDS

class NOVIXDetector:
    def __init__(self,bundle_path):
        self.bundle=joblib.load(bundle_path)
        self.stage1=self.bundle["stage1_hazard_gate"]
        self.safe_head=self.bundle["stage2_safe_classifier"]
        self.hazard_head=self.bundle["stage2_hazard_classifier"]
        self.gate_cal=self.bundle.get("hazard_gate_calibrator")
        self.subtype_cal=self.bundle.get("hazard_subtype_calibrator")
        self.threshold=float(self.bundle["threshold"])
        self.n_consecutive=int(self.bundle.get("n_consecutive",N_CONSECUTIVE))
        self.max_gap=float(self.bundle.get("max_gap_seconds",MAX_GAP_SECONDS))
        self.min_score=float(self.bundle.get("min_subtype_score",0.50))
        self.min_margin=float(self.bundle.get("min_subtype_margin",0.10))
        self.run=0; self.previous_elapsed=None
    def reset(self): self.run=0; self.previous_elapsed=None
    def predict_one(self,feature_row,elapsed_seconds):
        missing=[f for f in FEATURES if f not in feature_row]
        if missing: raise ValueError(f"Missing engineered features: {missing}")
        now=float(elapsed_seconds)
        if self.previous_elapsed is not None:
            gap=now-self.previous_elapsed
            if gap>self.max_gap or gap<=0: self.run=0
        x=pd.DataFrame([[float(feature_row[f]) for f in FEATURES]],columns=FEATURES)
        classes=list(self.stage1.classes_)
        raw_gate=float(self.stage1.predict_proba(x)[0][classes.index(1)]) if 1 in classes else 0.0
        p_h=float(self.gate_cal.predict([raw_gate])[0]) if self.gate_cal is not None else raw_gate
        self.run=self.run+1 if p_h>=self.threshold else 0
        confirmed=self.run>=self.n_consecutive
        flame=float(feature_row.get("f_flame",0))>=1.0
        raw_sub=self.hazard_head.predict_proba(x)[0]
        hclasses=list(self.hazard_head.classes_)
        raw_sub=raw_sub[[hclasses.index(c) for c in ["GAS_LEAK","SMOKE_LIKE","FIRE"]]]
        probs=self.subtype_cal.predict(raw_sub.reshape(1,-1))[0] if self.subtype_cal is not None else raw_sub
        order=np.argsort(probs); best=int(order[-1]); score=float(probs[best])
        second=float(probs[order[-2]]) if len(order)>1 else 0.0
        margin=score-second
        label=str(["GAS_LEAK","SMOKE_LIKE","FIRE"][best])
        if score<self.min_score or margin<self.min_margin: label="HAZARD_UNKNOWN"
        if flame: label="FIRE"; reason="FLAME_SENSOR_OVERRIDE"; score=1.0; margin=1.0
        elif confirmed: reason="CALIBRATED_AI_HAZARD_CLASSIFICATION" if getattr(self.subtype_cal,"calibrated",False) else "UNCALIBRATED_AI_HAZARD_CLASSIFICATION"
        else:
            label=str(np.asarray(self.safe_head.predict(x)).reshape(-1)[0]); reason="SAFE_HEAD_BEFORE_HAZARD_CONFIRMATION"; score=None; margin=None
        self.previous_elapsed=now
        return {"label":label,"risk_state":"HAZARD" if label in {"GAS_LEAK","SMOKE_LIKE","FIRE","HAZARD_UNKNOWN"} else "SAFE",
          "hazard_probability":p_h,"hazard_score_raw":raw_gate,"hazard_threshold":self.threshold,
          "subtype_score":score,"subtype_margin":margin,"subtype_calibrated":bool(getattr(self.subtype_cal,"calibrated",False)),
          "consecutive_count":self.run,"hazard_confirmed":bool(confirmed or flame),"alarm":bool(confirmed or flame),
          "safety_override":bool(flame),"reason":reason}
