import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'origen': ['A', 'A', 'B', 'B', 'C', 'C', 'D'],
    'destino': ['B', 'C', 'A', 'C', 'A', 'D', 'B'],
    'distancia': [1, 4, 1, 2, 4, 1, 5]
}

df = pd.DataFrame(data)

# Codificar las columnas categóricas
label_encoder = LabelEncoder()
df['origen'] = label_encoder.fit_transform(df['origen'])
df['destino'] = label_encoder.fit_transform(df['destino'])

# Definir las características (X) y la variable objetivo (y)
X = df[['origen', 'destino']]
y = df['distancia']

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Hacer predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio: {mse}')
