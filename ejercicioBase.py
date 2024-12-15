import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generación de datos ficticios de ventas (pueden ser reemplazados por datos reales)
data = {
    'Semana': np.arange(1, 11),
    'Ventas_Pasadas': [50, 55, 53, 60, 62, 65, 70, 75, 80, 85],
    'Demanda': [52, 57, 55, 63, 65, 68, 72, 78, 83, 88]
}
df = pd.DataFrame(data)

# División de los datos en características (X) y etiquetas (y)
X = df[['Ventas_Pasadas']]  # Variable independiente
y = df['Demanda']           # Variable dependiente

# Separación de datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creación del modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio (MSE):", mse)

# Predicción para un nuevo dato
nueva_venta = [[90]]  # Predicción para un producto con 90 ventas pasadas
prediccion = model.predict(nueva_venta)
print(f"Predicción de demanda para ventas pasadas {nueva_venta[0][0]}: {prediccion[0]:.2f}")

# Visualización básica (opcional si hay tiempo en el laboratorio)
import matplotlib.pyplot as plt
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, model.predict(X), color='red', label='Modelo ajustado')
plt.xlabel('Ventas Pasadas')
plt.ylabel('Demanda')
plt.legend()
plt.show()
