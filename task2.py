import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("dataset_phishing.csv")

print(dataset.info())
print(dataset.head())
print("\n" + "=" * 30)

# a
columnas_a_eliminar = ['url', 'length_url']

print(f"\nColumnas identificadas para eliminar: {columnas_a_eliminar}")

for columna in columnas_a_eliminar:
    if columna in dataset.columns:
        print(f"  Columna '{columna}' encontrada y sera eliminada")
    else:
        print(f"  Advertencia: Columna '{columna}' no encontrada")

dataset = dataset.drop(columns=columnas_a_eliminar, errors='ignore')

print(f"\nResultado despues de eliminar columnas:")
print(f"Columnas restantes: {dataset.shape[1]}")
print(f"Filas: {dataset.shape[0]}")

# b
print("\n" + "=" * 30)
print("\nvariables categoricas")

print(dataset["status"].dtype)
print(dataset["status"].value_counts())

print("\nCodificando variables categoricas")

dataset["status"] = (
    dataset["status"]
    .str.strip()
    .str.lower()
    .map({
        "phishing": 1,
        "legitimate": 0
    })
    .astype(int)
)

print(dataset["status"].dtype)
print(dataset["status"].value_counts())

# INCISO 2: SELECCION DE LAS 2 MEJORES CARACTERISTICAS

print("\n" + "=" * 60)
print("INCISO 2: SELECCION DE LAS 2 MEJORES CARACTERISTICAS")

# Separar la variable objetivo (y) de las caracteristicas (X)
print("\nPaso 1: Separando variable objetivo y caracteristicas...")

# La variable objetivo es 'status' (0 = legitimate, 1 = phishing)
y = dataset["status"]

# Las caracteristicas son todas las demas columnas (excepto 'status')
X = dataset.drop("status", axis=1)

print(f"Variable objetivo (y): '{y.name}'")
print(f"Forma de y: {y.shape}")  # Deberia ser (11430,)
print(f"Forma de X: {X.shape}")  # Deberia ser (11430, 87) aproximadamente

# Entender que es la correlacion
print("\n" + "-" * 50)

"""
En este caso:
Si correlacion es POSITIVA: cuando la caracteristica aumenta,
es mas probable que sea phishing (status=1)
Si correlacion es NEGATIVA: cuando la caracteristica aumenta,
es mas probable que sea legitimate (status=0)
"""

# Calcular correlaciones de cada caracteristica con 'status'
print("\n" + "-" * 50)
print("Calculando correlaciones")

# Crear un diccionario para guardar las correlaciones
correlaciones = {}

print("Calculando correlacion para cada caracteristica...")
for i, columna in enumerate(X.columns):
    # Calcular la correlacion de Pearson entre esta columna y 'status'
    correlacion = X[columna].corr(y)
    
    # Guardar el valor absoluto (nos interesa la fuerza)
    correlaciones[columna] = abs(correlacion)
    
    # Mostrar progreso cada 20 columnas
    if (i + 1) % 20 == 0:
        print(f"  Procesadas {i + 1} de {len(X.columns)} columnas...")

print(f"\nTotal de correlaciones calculadas: {len(correlaciones)}")

# Paso 4: Ordenar las correlaciones de mayor a menor
print("\n" + "-" * 50)
print("Ordenando las correlaciones")

# Convertir el diccionario a lista de tuplas (caracteristica, correlacion)
lista_correlaciones = list(correlaciones.items())

# Ordenar de mayor a menor correlacion (usando el segundo elemento de la tupla)
lista_correlaciones.sort(key=lambda x: x[1], reverse=True)

print("Correlaciones ordenadas de mayor a menor.")

# Mostrar las 10 mejores correlaciones
print("\n" + "-" * 50)
print("Top 10 caracteristicas mas correlacionadas")

print("\nRanking | Caracteristica" + " " * 25 + "| Correlacion")

for i, (caracteristica, correlacion) in enumerate(lista_correlaciones[:10], 1):
    # Mostrar las 10 mejores
    print(f"{i:7d} | {caracteristica:35s} | {correlacion:.4f}")

# Seleccionar las 2 mejores caracteristicas
print("\n" + "-" * 50)
print("las 2 mejores caracteristicas")

# Las dos primeras de la lista ordenada son las mejores
mejor_1 = lista_correlaciones[0][0]  # Caracteristica en posicion 0
correlacion_1 = lista_correlaciones[0][1]

mejor_2 = lista_correlaciones[1][0]  # Caracteristica en posicion 1
correlacion_2 = lista_correlaciones[1][1]

print(f"\nLas 2 caracteristicas seleccionadas son:")
print(f"\n1. {mejor_1}")
print(f"   Correlacion con 'status': {correlacion_1:.4f}")
print(f"   (La mas alta correlacion)")

print(f"\n2. {mejor_2}")
print(f"   Correlacion con 'status': {correlacion_2:.4f}")
print(f"   (Segunda mas alta correlacion)")

# Crear un nuevo dataset solo con estas 2 caracteristicas
print("\n" + "-" * 50)
print("dataset con las 2 caracteristicas seleccionadas")

# Seleccionar solo las 2 mejores caracteristicas del dataset original
X_2d = dataset[[mejor_1, mejor_2]]

print(f"\nNuevo conjunto de caracteristicas (X_2d):")
print(f"Forma: {X_2d.shape}")
print(f"\nPrimeras 5 filas de X_2d:")
print(X_2d.head())

print(f"\nPrimeras 5 valores de la variable objetivo (y):")
print(y.head().values)

# Analizar las caracteristicas seleccionadas
print("\n" + "-" * 50)
print("caracteristicas seleccionadas")

print(f"\nAnalisis de '{mejor_1}':")
print(f"  Tipo de dato: {X_2d[mejor_1].dtype}")
print(f"  Valores unicos: {X_2d[mejor_1].nunique()}")
print(f"  Minimo: {X_2d[mejor_1].min():.2f}")
print(f"  Maximo: {X_2d[mejor_1].max():.2f}")
print(f"  Promedio: {X_2d[mejor_1].mean():.2f}")

print(f"\nAnalisis de '{mejor_2}':")
print(f"  Tipo de dato: {X_2d[mejor_2].dtype}")
print(f"  Valores unicos: {X_2d[mejor_2].nunique()}")
print(f"  Minimo: {X_2d[mejor_2].min():.2f}")
print(f"  Maximo: {X_2d[mejor_2].max():.2f}")
print(f"  Promedio: {X_2d[mejor_2].mean():.2f}")

# ====================================================================
# GRAFICAR LOS DATOS EN 2D
print("\n" + "=" * 60)
print("GRAFICANDO LOS DATOS EN 2D")

print("\nCreando grafico de dispersion 2D")
print(f"Eje X: {mejor_1}")
print(f"Eje Y: {mejor_2}")
print("Colores: Azul = legitimate (0), Rojo = phishing (1)")

# Crear la figura
plt.figure(figsize=(10, 8))

# Separar los datos por clase para graficar con colores diferentes
# Clase 0: legitimate (azul)
x_clase0 = X_2d[y == 0][mejor_1]
y_clase0 = X_2d[y == 0][mejor_2]

# Clase 1: phishing (rojo)
x_clase1 = X_2d[y == 1][mejor_1]
y_clase1 = X_2d[y == 1][mejor_2]

# Graficar puntos de cada clase
plt.scatter(x_clase0, y_clase0, 
           color='blue', alpha=0.6, s=20,
           label='Legitimate (0)', edgecolors='w', linewidth=0.5)

plt.scatter(x_clase1, y_clase1, 
           color='red', alpha=0.6, s=20,
           label='Phishing (1)', edgecolors='w', linewidth=0.5)

# Configurar el grafico
plt.xlabel(mejor_1, fontsize=12)
plt.ylabel(mejor_2, fontsize=12)
plt.title(f'Visualización 2D para Detección de Phishing\n{mejor_1} vs {mejor_2}', 
          fontsize=14, fontweight='bold')

plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()

# Mostrar el grafico
plt.show()

# ====================================================================
# GRAFICA ADICIONAL: HISTOGRAMAS POR CLASE
# ====================================================================
print("\n" + "=" * 60)
print("GRAFICAS ADICIONALES: HISTOGRAMAS POR CLASE")
print("=" * 60)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histograma de mejor_1 por clase
axes[0, 0].hist(x_clase0, bins=30, alpha=0.7, color='blue', label='Legitimate')
axes[0, 0].hist(x_clase1, bins=30, alpha=0.7, color='red', label='Phishing')
axes[0, 0].set_xlabel(mejor_1)
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title(f'Distribución de {mejor_1} por Clase')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Histograma de mejor_2 por clase
axes[0, 1].hist(y_clase0, bins=30, alpha=0.7, color='blue', label='Legitimate')
axes[0, 1].hist(y_clase1, bins=30, alpha=0.7, color='red', label='Phishing')
axes[0, 1].set_xlabel(mejor_2)
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].set_title(f'Distribución de {mejor_2} por Clase')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Boxplot de mejor_1 por clase
boxplot_data1 = [x_clase0, x_clase1]
axes[1, 0].boxplot(boxplot_data1, labels=['Legitimate', 'Phishing'])
axes[1, 0].set_ylabel(mejor_1)
axes[1, 0].set_title(f'Boxplot de {mejor_1} por Clase')
axes[1, 0].grid(True, alpha=0.3)

# Boxplot de mejor_2 por clase
boxplot_data2 = [y_clase0, y_clase1]
axes[1, 1].boxplot(boxplot_data2, labels=['Legitimate', 'Phishing'])
axes[1, 1].set_ylabel(mejor_2)
axes[1, 1].set_title(f'Boxplot de {mejor_2} por Clase')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
