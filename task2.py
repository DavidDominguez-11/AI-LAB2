import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



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

selected_features = [correlations.index[2], correlations.index[3]]
#selected_features = ["ratio_digits_url", "nb_dots"]

print(f"\nFeatures seleccionadas: {selected_features}")

# dataset con las 2 variables mas correlacionadas
X_2d = dataset[selected_features]
y = dataset["status"]

# scatter plot
plt.figure(figsize=(6, 5))

plt.scatter(
    X_2d[y == 0].iloc[:, 0],
    X_2d[y == 0].iloc[:, 1],
    label="Legitimate",
    alpha=0.5
)

plt.scatter(
    X_2d[y == 1].iloc[:, 0],
    X_2d[y == 1].iloc[:, 1],
    label="Phishing",
    alpha=0.5
)

plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.legend()
plt.title("Distribución de Features Seleccionadas")
plt.show()

# 3
# a
print("\n" + "=" * 30)
print("\n Escalado MUST")

scaler = StandardScaler()

# Ajustar y transformar SOLO las features
X_scaled = scaler.fit_transform(X_2d)

# pasarlo a dataframe pq me daba errores
X_scaled = pd.DataFrame(
    X_scaled,
    columns=X_2d.columns
)

# para verrificar mean ≈ 0, std ≈ 1
print("\nVerificacion de escalado mean ≈ 0, std ≈ 1: ")
print(X_scaled.describe())

# 4 80/20

print("\n" + "=" * 30)
print("Split 80/20")

# hacer el split 80 / 20
X_train, X_test, y_train, y_test = train_test_split(
    X_2d,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# comprobaciones
print("Training size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

print("\nDistribución en training:")
print(y_train.value_counts(normalize=True))

print("\nDistribución en test:")
print(y_test.value_counts(normalize=True))

# escalar los datos solo del training set
scaler2 = StandardScaler()
scaler2.fit(X_train)

# Transformar training y test
X_train_scaled = scaler2.transform(X_train)
X_test_scaled = scaler2.transform(X_test)

# pasarlo a dataframe pq me daba errores
print("\nX_train_scaled: ")
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_2d.columns
)
print("\nVerificacion de escalado mean ≈ 0, std ≈ 1: ")
print(X_train_scaled.describe())

print("\nX_test_scaled: ")
X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_2d.columns
)
print("\nVerificacion de escalado mean ≈ 0, std ≈ 1: ")
print(X_test_scaled.describe())

