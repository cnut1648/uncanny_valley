from xgboost import XGBClassifier

MODEL_CONFIG = {
    "xgboost": {
        "model": XGBClassifier,
        "params": {
            "objective": "multi:softprob",
            "n_estimators": (100, 1000),
            "max_depth": (20, 100),
            "subsample": (0.5, 1),
            "reg_alpha": [0, 1],
            "reg_lambda": [0, 1]
        }
    }
}

NEPTUNE_API="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwOGU3OTA3Mi1mMGYzLTQ1MzgtYjU3OC04NGUzNTkxMjYwM2EifQ=="