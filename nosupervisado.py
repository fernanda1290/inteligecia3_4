import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generar un dataset ficticio
data = {
    'origen': ['A', 'A', 'B', 'B', 'C', 'C', 'D'],
    'destino': ['B', 'C', 'A', 'C', 'A', 'D', 'B'],
    'distancia': [1, 4, 1, 2, 4, 1, 5]
}

df = pd.DataFrame(data)

# Preparar los datos para K-means
X = df[['distancia']]

# Implementar el modelo K-means
kmeans = KMeans(n_clusters=2, random_state=42)  # Elegimos 2 clústeres para este ejemplo
df['cluster'] = kmeans.fit_predict(X)

# Mostrar los resultados
print("Clústeres asignados:")
print(df)

# Visualización
plt.scatter(df['distancia'], [0] * len(df), c=df['cluster'], cmap='viridis')
plt.title('Clústeres de Distancias')
plt.xlabel('Distancia')
plt.yticks([])  # Ocultar el eje y para claridad
plt.show()
