"""
model_loader.py
Singleton loader untuk model XGBoost, scaler, dan konfigurasi.
"""
import os
import json
import threading
import warnings
import joblib
import xgboost as xgb
from preprocessing import InferencePipeline, EXPECTED_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_JSON = os.path.join(BASE_DIR, 'model', 'xgboost_model.json')
MODEL_PATH_PKL = os.path.join(BASE_DIR, 'model', 'xgboost_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
CONFIG_PATH = os.path.join(BASE_DIR, 'model', 'model_config.json')

DEFAULT_CONFIG = {
    'confidence_threshold': 0.20,
    'thresh_ergo': 0.70,
    'thresh_non': 0.60,
    'buzzer_duration': 1.5,
    'prediction_interval_ms': 300,
    'buzzer_enabled': True,
    'sensor_stale_timeout_ms': 5000,
}

_lock = threading.RLock()
_model = None
_scaler = None
_config = None
_pipeline = None

def load_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
            
        if os.path.exists(MODEL_PATH_PKL):
            _model = joblib.load(MODEL_PATH_PKL)
        elif os.path.exists(MODEL_PATH_JSON):
            booster = xgb.Booster()
            booster.load_model(MODEL_PATH_JSON)
            _model = booster
        else:
            raise FileNotFoundError(
                f"Model file not found. Expected {MODEL_PATH_PKL} or {MODEL_PATH_JSON}"
            )
        return _model

def load_scaler():
    global _scaler
    if _scaler is not None:
        return _scaler
    with _lock:
        if _scaler is not None:
            return _scaler
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
        _scaler = joblib.load(SCALER_PATH)
        return _scaler

def load_config():
    global _config
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        _config = DEFAULT_CONFIG.copy()
        return _config
    try:
        with open(CONFIG_PATH, 'r') as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        _config = cfg
        return _config
    except Exception as e:
        warnings.warn(f"Failed to load config: {e}. Using default.")
        return DEFAULT_CONFIG.copy()

def save_config(config: dict):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        warnings.warn(f"Failed to save config: {e}")
        return False

def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        model = load_model()
        scaler = load_scaler()
        _pipeline = InferencePipeline(
            model=model,
            scaler=scaler,
            expected_features=EXPECTED_FEATURES
        )
        return _pipeline

def reload_all():
    global _model, _scaler, _config, _pipeline
    with _lock:
        _model = None
        _scaler = None
        _config = None
        _pipeline = None
    get_pipeline()