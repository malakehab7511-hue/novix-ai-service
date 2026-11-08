from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Small, regularized search space. The outer environment fold is never used
# during this selection; only grouped OOF predictions from its training data are.
LIGHTGBM_CANDIDATES = [
    {"n_estimators": 350, "learning_rate": 0.035, "num_leaves": 7, "max_depth": 3,
     "min_child_samples": 30, "subsample": 0.85, "colsample_bytree": 0.85,
     "reg_alpha": 0.5, "reg_lambda": 5.0},
    {"n_estimators": 500, "learning_rate": 0.025, "num_leaves": 7, "max_depth": 3,
     "min_child_samples": 45, "subsample": 0.9, "colsample_bytree": 0.8,
     "reg_alpha": 1.0, "reg_lambda": 8.0},
    {"n_estimators": 350, "learning_rate": 0.03, "num_leaves": 15, "max_depth": 4,
     "min_child_samples": 35, "subsample": 0.8, "colsample_bytree": 0.9,
     "reg_alpha": 0.5, "reg_lambda": 8.0},
    {"n_estimators": 450, "learning_rate": 0.025, "num_leaves": 15, "max_depth": -1,
     "min_child_samples": 50, "subsample": 0.85, "colsample_bytree": 0.85,
     "reg_alpha": 1.0, "reg_lambda": 10.0},
]


def make_binary_model(name, random_state=42, params=None):
    params = dict(params or {})
    if name == "CatBoost":
        return CatBoostClassifier(
            iterations=500, depth=5, learning_rate=0.04, loss_function="Logloss",
            auto_class_weights="Balanced", verbose=False, random_seed=random_state,
            l2_leaf_reg=5
        )
    if name == "LightGBM":
        base = {
            "n_estimators": 400, "learning_rate": 0.035, "num_leaves": 15,
            "max_depth": -1, "class_weight": "balanced", "reg_lambda": 3,
            "min_child_samples": 25, "verbosity": -1, "random_state": random_state,
            "n_jobs": -1,
        }
        base.update(params)
        return LGBMClassifier(**base)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features=0.85,
            class_weight="balanced", n_jobs=-1, random_state=random_state
        )
    raise ValueError(name)


def make_multiclass_model(name, random_state=42, params=None):
    params = dict(params or {})
    if name == "CatBoost":
        return CatBoostClassifier(
            iterations=500, depth=5, learning_rate=0.04, loss_function="MultiClass",
            auto_class_weights="Balanced", verbose=False, random_seed=random_state,
            l2_leaf_reg=5
        )
    if name == "LightGBM":
        base = {
            "n_estimators": 400, "learning_rate": 0.035, "num_leaves": 15,
            "class_weight": "balanced", "reg_lambda": 3, "min_child_samples": 20,
            "verbosity": -1, "random_state": random_state, "n_jobs": -1,
        }
        # Use the selected tree complexity for the specialist heads too.
        for key in ("n_estimators", "learning_rate", "num_leaves", "max_depth",
                    "min_child_samples", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
            if key in params:
                base[key] = params[key]
        return LGBMClassifier(**base)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features=0.85,
            class_weight="balanced", n_jobs=-1, random_state=random_state
        )
    raise ValueError(name)


def available_models():
    found = []
    for name in ["CatBoost", "LightGBM", "ExtraTrees"]:
        try:
            make_binary_model(name)
            found.append(name)
        except Exception as exc:
            print(f"Skipping {name}: package/import issue: {exc}")
    return found
