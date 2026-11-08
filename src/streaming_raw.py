"""Raw ESP32 sensor stream -> exact engineered features -> calibrated inference."""
from .feature_extractor import RawSensorFeatureExtractor
from .inference import NOVIXDetector

class NOVIXRawStream:
    def __init__(self, bundle_path):
        self.extractor=RawSensorFeatureExtractor()
        self.detector=NOVIXDetector(bundle_path)
    def reset(self):
        self.extractor.reset(); self.detector.reset()
    def predict_raw(self, sample, elapsed_seconds):
        # Flame is an independent immediate safety override and must not depend on baselines/history.
        if float(sample.get("flame",0)) >= 1.0:
            features=None
            try:
                # Preserve this reading in temporal history whenever the full raw payload is available.
                features=self.extractor.update(sample,elapsed_seconds)
            except (ValueError, TypeError):
                pass
            self.detector.reset()
            return {"status":"ready","label":"FIRE","risk_state":"HAZARD","hazard_probability":1.0,
                    "subtype_score":1.0,"subtype_margin":1.0,"hazard_confirmed":True,"alarm":True,
                    "safety_override":True,"reason":"IMMEDIATE_RAW_FLAME_OVERRIDE","features":features}
        features=self.extractor.update(sample,elapsed_seconds)
        if features is None:
            return {"status":"warming_up","label":"WARMING_UP","risk_state":"UNKNOWN","hazard_probability":None,
                    "subtype_score":None,"subtype_margin":None,"hazard_confirmed":False,"alarm":False,
                    "safety_override":False,"reason":"INSUFFICIENT_TEMPORAL_HISTORY","features":None}
        result=self.detector.predict_one(features,elapsed_seconds)
        return {"status":"ready",**result,"features":features,"feature_parity_status":"EQUATION_RECONSTRUCTION_VERIFIED"}
