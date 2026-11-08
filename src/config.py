from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "NOVIX_FINAL_training_data.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "cleaned_training_data.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "label_final"
GROUP_ENV = "environment_id"
GROUP_SESSION = "session_id"

SAFE_LABELS = {"NORMAL", "BENIGN_INTERFERENCE"}
HAZARD_LABELS = {"GAS_LEAK", "SMOKE_LIKE", "FIRE"}
ALL_LABELS = ["NORMAL", "BENIGN_INTERFERENCE", "GAS_LEAK", "SMOKE_LIKE", "FIRE"]
HAZARD_CLASSES = ["GAS_LEAK", "SMOKE_LIKE", "FIRE"]

# Deliberately exclude raw sensor counts, absolute baselines, IDs, timestamps,
# labels/confidence metadata and post-event annotations to reduce leakage.
FEATURES = [
    "f_mq2_ratio", "f_mq6_ratio",
    "f_mq2_slope_10s", "f_mq2_slope_30s", "f_mq2_accel", "f_mq2_var_10s",
    "f_mq6_slope_10s", "f_mq6_slope_30s", "f_mq6_accel", "f_mq6_var_10s",
    "f_mq_agreement", "f_temp_change_30s", "f_humidity_change_30s",
    "f_flame", "f_flame_frac_30s", "f_sensor_saturation",
]

N_CONSECUTIVE = 10
TARGET_HAZARD_RECALL = 0.95
MAX_GAP_SECONDS = 3.0
CALIBRATION_SECONDS = 3600
