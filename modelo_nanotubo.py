import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Input


import tensorflow as tf
import random

import matplotlib.pyplot as plt

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ========================== 1. CARGAR DATOS=====================================================
archivo = "datos_nanotubo.csv"
df = pd.read_csv(archivo)
print("Dimensiones:", df.shape)
df.head()

# ===========================2. SEPARAR VARIABLES=====================================================
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("X =", X.shape)
print("y =", y.shape)


# ==========================3. DIVIDIR LOS DATOS=====================================================
#    80% entrenamiento
#    10% validación
#    10% prueba

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

print("Entrenamiento:", X_train.shape)
print("Validación:", X_val.shape)
print("Prueba:", X_test.shape)

# ===========================4. NORMALIZAR ENTRADAS================================================
scaler_X = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

# =======================5. NORMALIZAR SALIDA=====================================================
#    (SOLO con entrenamiento)
scaler_y = StandardScaler()

y_train = scaler_y.fit_transform(
    y_train.reshape(-1,1)
)

y_val = scaler_y.transform(
    y_val.reshape(-1,1)
)

y_test_scaled = scaler_y.transform(
    y_test.reshape(-1,1)
)


# =========================6. CREAR RED NEURONAL=====================================================
model = Sequential()

model.add(Input(shape=(X_train.shape[1],)))

model.add(Dense(
    128,
    activation='relu'
))

model.add(Dropout(0.2))

model.add(Dense(
    64,
    activation='relu'
))

model.add(Dropout(0.2))

model.add(Dense(
    32,
    activation='relu'
))

model.add(Dense(
    1,
    activation='linear'
))

# ==========================7. COMPILAR=====================================================
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ========================8. EARLY STOPPING=====================================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)

# ========================REDUCIR LEARNING RATE=====================================================
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# ======================= 9. ENTRENAR =====================================================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=1000,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    label='Entrenamiento'
)

plt.plot(
    history.history['val_loss'],
    label='Validación'
)

plt.xlabel('Época')
plt.ylabel('MSE')
plt.title('Curva de pérdida')
plt.legend()
plt.grid(True)

plt.savefig(
    "curva_perdida.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()


plt.figure(figsize=(8,5))

plt.plot(
    history.history['mae'],
    label='MAE entrenamiento'
)

plt.plot(
    history.history['val_mae'],
    label='MAE validación'
)

plt.xlabel('Época')
plt.ylabel('MAE')
plt.title('Curva de MAE')
plt.legend()
plt.grid(True)

plt.savefig(
    "curva_mae.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ====================== 10. PREDICCIONES =====================================================
# Predicciones
y_pred_scaled = model.predict(X_test)

y_pred = scaler_y.inverse_transform(
    y_pred_scaled
)

# ======================Tabla de resultados
tabla_resultados = pd.DataFrame({
    'Real': y_test,
    'Predicción': y_pred.flatten()
})

tabla_resultados['Error absoluto'] = (
    abs(
        tabla_resultados['Real']
        - tabla_resultados['Predicción']
    )
)

tabla_resultados.head(10)

# Mejores
tabla_resultados.sort_values(
    by='Error absoluto'
).head(10)

# Peores
tabla_resultados.sort_values(
    by='Error absoluto',
    ascending=False
).head(10)


tabla_resultados.to_csv(
    "predicciones_modelo.csv",
    index=False
)

print("Tabla guardada.")

# ======================== 11. MÉTRICAS =====================================================
mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("RESULTADOS")
print("----------------------")
print(f"R²   = {r2:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")


plt.figure(figsize=(7,7))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7,
    color='blue'
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)

plt.xlabel('Valores reales')
plt.ylabel('Valores predichos')
plt.title('Valores reales vs valores predichos')

plt.grid(True)

plt.savefig(
    "real_vs_predicho.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ==================== 12. GUARDAR MODELO =====================================================
model.save(
    "modelo_nanotubo.keras"
)

print("Modelo guardado correctamente.")
