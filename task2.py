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

# 2
# a
print("\n" + "=" * 30)
print("\nvariables independientes y objetivo")
"""
X = variables independientes
y = variable objetivo
"""
X = dataset.drop(columns=["status"])
y = dataset["status"]


print("\n" + "=" * 30)
print("\nCorrelacion con variable objetivo")
# correlacion con var obj, como la pusimos estandar 0 1 puedo usar pearson
correlations = X.corrwith(y)
print(correlations)

# como no importa si es neg o pos solo que tan fuerte esta correlacion
# puedo usar el valor absoluto
correlations = correlations.abs()
print(correlations)

# ordenar por correlacion
correlations = correlations.sort_values(ascending=False)
print(correlations)

# mostrar las 10 mas correlacionadas
print("\nLas 10 mas correlacionadas:")
print(correlations.head(10))

# seleccionar las 2 variables mas correlacionadas
print("\nSeleccionando las 2 variables mas correlacionadas:")
X_selected = X[correlations.head(2).index]
print(X_selected.head())
selected_features = [correlations.index[0], correlations.index[1]]
print(f"\nFeatures seleccionadas: {selected_features}")




