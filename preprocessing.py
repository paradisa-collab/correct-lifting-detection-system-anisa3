"""
preprocessing.py
Pipeline preprocessing untuk deteksi postur menggunakan XGBoost.
Menggunakan buffer per session_id untuk windowed aggregation.
DISESUAIKAN UNTUK 48 FITUR (6 Acc + 6 Gyro x 4 Agregasi).
"""
import warnings
import threading
import numpy as np
import pandas as pd
from collections import deque

WINDOW_SIZE = 15
STEP_SIZE = 15

# 12 Kolom sensor mentah (sesuai dengan 48 fitur yang dilatih model)
RAW_SENSOR_COLS = [
    'ax1', 'ay1', 'az1', 'gx1', 'gy1', 'gz1',
    'ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2'
]

# GENERATE 48 EXPECTED FEATURES SECARA OTOMATIS
# Ini menjamin urutan dan nama fitur 100% sama dengan saat training di Colab
AGGREGATIONS = ['mean', 'std', 'min', 'max']
EXPECTED_FEATURES = [f"{col}_{agg}" for col in RAW_SENSOR_COLS for agg in AGGREGATIONS]

def validate_and_normalize_raw(data: dict) -> dict:
    """Validasi input mentah, pastikan 12 kolom sensor ada dan bertipe float."""
    cleaned = {}
    for col in RAW_SENSOR_COLS:
        raw_val = data.get(col, 0.0)
        try:
            val = float(raw_val) if raw_val not in (None, '') else 0.0
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            cleaned[col] = val
        except (TypeError, ValueError):
            cleaned[col] = 0.0
            
    # Opsional: tangani pressure/altitude jika ada, agar tidak error, meski tidak dipakai model
    for col in ['pressure_hpa', 'altitude_m']:
        if col in data:
            try:
                val = float(data[col]) if data[col] not in (None, '') else 0.0
                cleaned[col] = 0.0 if np.isnan(val) or np.isinf(val) else val
            except (TypeError, ValueError):
                cleaned[col] = 0.0
                
    return cleaned

def create_windowed_features(X_df, window_size=WINDOW_SIZE, step_size=STEP_SIZE, y_series=None):
    """
    Agregasi window (mean/std/min/max) HANYA pada 12 kolom sensor mentah,
    persis seperti kondisi training di Colab (derived/rolling features diabaikan).
    """
    # Ambil HANYA 12 kolom yang relevan untuk model ini
    cols_to_keep = [c for c in RAW_SENSOR_COLS if c in X_df.columns]
    X_processed = X_df[cols_to_keep].apply(pd.to_numeric, errors='coerce').fillna(0).reset_index(drop=True)
    
    if len(X_processed) < window_size:
        if y_series is not None:
            return pd.DataFrame(), pd.Series()
        return pd.DataFrame()
        
    features_list = []
    labels_list = []
    
    for i in range(0, len(X_processed) - window_size + 1, step_size):
        window_X = X_processed.iloc[i : i + window_size]
        agg_row = {}
        
        for col in window_X.columns:
            agg_row[f'{col}_mean'] = float(window_X[col].mean())
            agg_row[f'{col}_std'] = float(np.nan_to_num(window_X[col].std()))
            agg_row[f'{col}_min'] = float(window_X[col].min())
            agg_row[f'{col}_max'] = float(window_X[col].max())
            
        features_list.append(agg_row)
        
        if y_series is not None:
            window_y = y_series.iloc[i : i + window_size]
            labels_list.append(pd.Series(window_y).mode()[0])
            
    df_features = pd.DataFrame(features_list)
    if y_series is not None:
        return df_features, pd.Series(labels_list)
    return df_features

def align_to_expected(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """Pastikan DataFrame memiliki 48 kolom yang tepat dan urutannya benar."""
    # Tambahkan kolom yang hilang (misal gyro tidak dikirim realtime) dengan nilai 0
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        warnings.warn(f"[alignment] Missing columns filled with 0: {missing}", RuntimeWarning)
        for col in missing:
            df[col] = 0.0
            
    # Hapus kolom ekstra dan pastikan urutan sesuai EXPECTED_FEATURES
    return df[expected_features].copy()

class InferencePipeline:
    """
    Pipeline inferensi real-time dengan buffer per session_id.
    Thread-safe menggunakan threading.Lock.
    """
    def __init__(self, model, scaler, expected_features: list):
        self.model = model
        self.scaler = scaler
        self.expected_features = expected_features
        self._buffers = {}
        self._lock = threading.Lock()
        
    def predict(self, raw_dict: dict, session_id: str = 'default') -> dict:
        cleaned = validate_and_normalize_raw(raw_dict)
        
        with self._lock:
            if session_id not in self._buffers:
                self._buffers[session_id] = deque(maxlen=WINDOW_SIZE)
            self._buffers[session_id].append(cleaned)
            buf_len = len(self._buffers[session_id])
            
            if buf_len < WINDOW_SIZE:
                return {
                    'status': 'buffering',
                    'buffered': buf_len,
                    'required': WINDOW_SIZE,
                    'prediction': None,
                    'label': 'menunggu_data'
                }
                
            buf_data = list(self._buffers[session_id])
            self._buffers[session_id].clear()
            
        # Jalankan pipeline: agregasi window → align → scaling
        # (Derived & Rolling di-skip karena model tidak dilatih dengannya)
        df_buf = pd.DataFrame(buf_data)
        df_agg = create_windowed_features(df_buf, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
        
        if df_agg.empty:
            return {'status': 'error', 'message': 'Aggregation failed'}
            
        df_current = df_agg.iloc[[-1]].copy()
        df_aligned = align_to_expected(df_current, self.expected_features)
        
        # Scaling
        X_scaled = self.scaler.transform(df_aligned)
        
        # Prediksi
        if hasattr(self.model, 'predict_proba'):
            probas_arr = self.model.predict_proba(X_scaled)[0]
            classes = list(self.model.classes_)
            # Asumsi: 0 = ergonomis, 1 = non-ergonomis (sesuaikan jika terbalik)
            prob_non = float(probas_arr[classes.index(0)]) if 0 in classes else 0.0
            prob_ergo = float(probas_arr[classes.index(1)]) if 1 in classes else 0.0
            pred_class = int(self.model.predict(X_scaled)[0])
        else:
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X_scaled, feature_names=self.expected_features)
            prob_non = float(self.model.predict(dmatrix)[0])
            prob_ergo = 1.0 - prob_non
            pred_class = 1 if prob_non >= 0.5 else 0
            
        return {
            'status': 'ready',
            'prediction': pred_class,
            'label': 'ergonomis' if pred_class == 0 else 'non-ergonomis',
            'probabilities': {'ergonomis': round(prob_ergo, 4), 'non-ergonomis': round(prob_non, 4)},
            'feature_count': len(self.expected_features),
            'extracted_features': df_aligned.iloc[0].to_dict(),
        }

    def reset_session(self, session_id: str = 'default') -> None:
        with self._lock:
            self._buffers.pop(session_id, None)

    def reset_all_sessions(self) -> None:
        with self._lock:
            self._buffers.clear()