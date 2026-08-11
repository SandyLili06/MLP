import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def ruta_recurso(ruta):
    # PyInstaller
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    # Nuitka
    elif getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    ruta_final = os.path.join(base, ruta)
    print("Buscando:", ruta_final)
    return ruta_final


def calcular_energia(ruta_archivo):
    from tensorflow.keras.models import load_model
    modelo = load_model(ruta_recurso("modelos/modelo_nanotubo.keras"))
    scaler_X = joblib.load(ruta_recurso("modelos/scaler_X.pkl"))
    scaler_y = joblib.load(ruta_recurso("modelos/scaler_y.pkl"))
    historial_entrenamiento = joblib.load(ruta_recurso("modelos/historial_entrenamiento.pkl"))

    datos =pd.read_csv(ruta_archivo)

    if "energia" in datos.columns:
        X = datos.drop(columns=["energia"])
    else:
        X = datos.copy()

    X_scaled = scaler_X.transform(X)

    energia_scaled = modelo.predict(X_scaled)

    energia = scaler_y.inverse_transform(
        energia_scaled
    )

    # Agregar resultado
    datos["energia_predicha"] = energia.flatten()

    r2 = r2_score(datos["energia"], datos["energia_predicha"])
    mae = mean_absolute_error(datos["energia"], datos["energia_predicha"])
    mse = mean_squared_error(datos["energia"], datos["energia_predicha"])
    rmse = np.sqrt(mse)

    metricas = {
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }

    print("\nRESULTADOS SOBRE EL CSV CARGADO")
    print("------------------------------")
    print(f"R²   = {r2:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")

    return datos, historial_entrenamiento, metricas
