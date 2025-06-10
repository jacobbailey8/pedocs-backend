import joblib
import os
from darts import TimeSeries

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/pedocs_model_24hr.pkl')

_model = None  # global cache

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_pedocs(target_df, feature_df):
    model = load_model()

    feature_series = TimeSeries.from_dataframe(feature_df, time_col='timestamp')
    past_covariates = feature_series[['pedocs_score_rolling_3h',
                         'pedocs_score_rolling_6h', 'pedocs_score_rolling_12h', 'pedocs_score_rolling_24h',
                           'pedocs_score_rolling_48h']]
    future_covariates = feature_series[['is_peak_hour', 'temperature_2m (°C)', 'wind_speed_10m (km/h)', 'is_valley_hour', 'is_weekend']]


    target_series = TimeSeries.from_dataframe(target_df, time_col='timestamp')
    target = target_series['pedocs_score']

    return model.predict(n=24, series=target, past_covariates=past_covariates, future_covariates=future_covariates)
    

