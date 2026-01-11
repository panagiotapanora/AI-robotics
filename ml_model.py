import joblib
import numpy as np

class MLModel:
    def __init__(self, model_path="model.joblib", scaler_path="scaler.joblib"):
        # load with safe fallback if files are missing or invalid
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        except Exception:
            self.model = None
            self.scaler = None

    def predict(self, sensor_values: dict):
        # If model/scaler not available, return safe default
        if not self.model or not self.scaler:
            return {"probability": 0.0, "level": "Low"}

        # order must match training: temperature, humidity, co2
        X = np.array([[sensor_values.get("Temperature", 0),
                       sensor_values.get("Humidity", 0),
                       sensor_values.get("CO2", 0)]])
        X_scaled = self.scaler.transform(X)
        pred_prob = float(self.model.predict_proba(X_scaled)[0][1])  # probability of danger
        level = "High" if pred_prob > 0.75 else "Medium" if pred_prob > 0.5 else "Low"
        return {"probability": round(pred_prob, 3), "level": level}
