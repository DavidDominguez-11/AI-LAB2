# ====================================================================
# TASK 2 - PREPARACIÓN DE DATOS PARA DETECCIÓN DE PHISHING
# ====================================================================

# Importamos solo pandas, que es esencial para trabajar con datos tabulares
import pandas as pd

print("=" * 70)
print("TASK 2 - PREPARACION DE DATOS PARA DETECCION DE PHISHING")
print("=" * 70)

# ====================================================================
# INCISO 1: CARGA Y LIMPIEZA DE DATOS
# ====================================================================
print("\n--- INCISO 1: CARGA Y LIMPIEZA DE DATOS ---")
print("\n1a. Cargando el dataset dataset_phishin.csv...")

# Cargar el archivo CSV en un DataFrame de pandas
dataset = pd.read_csv("dataset_phishin.csv")

print("Dataset cargado exitosamente.")
print(f"Dimensiones del dataset: {dataset.shape[0]} filas y {dataset.shape[1]} columnas")

# Mostrar un vistazo inicial de los datos
print("\nPrimeras 3 filas del dataset:")
print(dataset.head(3))

# Analizar las columnas que tenemos
print("\nAnalizando las columnas disponibles:")
print(f"Primera columna: '{dataset.columns[0]}' (URL)")
print(f"Ultima columna: '{dataset.columns[-1]}' (Status)")
print(f"Total de columnas: {len(dataset.columns)}")

# 1a. Eliminar columnas irrelevantes segun las instrucciones
print("\n1a. Eliminando columnas irrelevantes...")
print("Razon: Las instrucciones indican eliminar columnas que sean IDs unicos")
print("       o que puedan introducir ruido en el analisis.")

# Columnas a eliminar:
# 1. 'url' - Es un identificador unico para cada sitio web
# 2. 'length_url' - Las instrucciones mencionan especificamente que 
#    si no esta normalizada puede meter ruido
columnas_a_eliminar = ['url', 'length_url']

print(f"\nColumnas identificadas para eliminar: {columnas_a_eliminar}")
print("Explicacion:")
print("- 'url': Cada URL es unica, no aporta informacion predictiva")
print("- 'length_url': Variable con alta varianza que necesita normalizacion")

# Verificar que las columnas existan antes de eliminarlas
for columna in columnas_a_eliminar:
    if columna in dataset.columns:
        print(f"  Columna '{columna}' encontrada y sera eliminada")
    else:
        print(f"  Advertencia: Columna '{columna}' no encontrada")

# Eliminar las columnas
dataset = dataset.drop(columns=columnas_a_eliminar, errors='ignore')

print(f"\nResultado despues de eliminar columnas:")
print(f"Columnas restantes: {dataset.shape[1]}")
print(f"Filas: {dataset.shape[0]}")

# ====================================================================
# INCISO 1b: CODIFICACION DE VARIABLES CATEGORICAS
# ====================================================================
print("\n--- INCISO 1b: CODIFICACION DE VARIABLES CATEGORICAS ---")
print("\nProposito: Los algoritmos de machine learning trabajan con numeros,")
print("           no con texto. Necesitamos convertir las categorias a valores numericos.")

# Identificar variables categoricas (de tipo texto)
print("\nBuscando variables de tipo texto en el dataset...")

variables_categoricas = []
for columna in dataset.columns:
    if dataset[columna].dtype == 'object':  # 'object' usualmente indica texto en pandas
        variables_categoricas.append(columna)

print(f"Variables categoricas encontradas: {variables_categoricas}")

# En nuestro dataset, solo 'status' es categorica
if 'status' in dataset.columns:
    print("\nAnalizando la variable 'status':")
    
    # Mostrar los valores unicos que tiene
    valores_unicos = dataset['status'].unique()
    print(f"Valores unicos en 'status': {valores_unicos}")
    print(f"Cantidad de valores unicos: {len(valores_unicos)}")
    
    # Ver la distribucion de clases
    print("\nDistribucion actual de clases:")
    conteo_clases = dataset['status'].value_counts()
    print(conteo_clases)
    
    # Convertir a valores numericos
    print("\nConvirtiendo a valores numericos:")
    print("  'legitimate' -> 0")
    print("  'phishing' -> 1")
    
    # Usamos un diccionario de mapeo
    mapeo = {'legitimate': 0, 'phishing': 1}
    dataset['status'] = dataset['status'].map(mapeo)
    
    # Verificar la conversion
    print("\nVerificando la conversion:")
    valores_despues = dataset['status'].unique()
    print(f"Valores unicos despues de conversion: {valores_despues}")
    
    # Verificar que no hayan valores NaN (no mapeados)
    if dataset['status'].isnull().any():
        print("Advertencia: Hay valores no mapeados en 'status'")
    else:
        print("Conversion completada exitosamente")
else:
    print("Error: Columna 'status' no encontrada en el dataset")

# ====================================================================
# INCISO 2: SELECCION DE 2 CARACTERISTICAS
# ====================================================================
print("\n--- INCISO 2: SELECCION DE 2 CARACTERISTICAS ---")
print("\nProposito: Segun las instrucciones, necesitamos seleccionar 2 variables")
print("           para poder visualizar la frontera de decision en 2D.")

print("\nCriterio de seleccion: Las 2 variables con mayor correlacion")
print("                       con la variable objetivo (status).")

# Separar la variable objetivo (y) y las caracteristicas (X)
print("\nSeparando variables:")
y = dataset['status']  # Variable objetivo
X = dataset.drop('status', axis=1)  # Todas las demas variables

print(f"Variable objetivo (y): {y.name}")
print(f"Caracteristicas disponibles (X): {X.shape[1]} columnas")

# Calcular correlaciones con la variable objetivo
print("\nCalculando correlaciones entre cada caracteristica y 'status'...")

# Diccionario para almacenar correlaciones
correlaciones = {}

for columna in X.columns:
    # Calcular correlacion de Pearson
    # Esta mide la relacion lineal entre dos variables
    # Valor entre -1 (correlacion negativa perfecta) y 1 (correlacion positiva perfecta)
    correlacion = X[columna].corr(y)
    
    # Usamos el valor absoluto porque nos interesa la fuerza de la relacion,
    # no la direccion (positiva o negativa)
    fuerza_correlacion = abs(correlacion)
    
    correlaciones[columna] = fuerza_correlacion

# Convertir a lista y ordenar de mayor a menor correlacion
lista_correlaciones = list(correlaciones.items())
lista_correlaciones.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 caracteristicas mas correlacionadas con 'status':")
print("=" * 60)
for i, (caracteristica, correlacion) in enumerate(lista_correlaciones[:10], 1):
    print(f"{i:2d}. {caracteristica:30s}: {correlacion:.4f}")

# Seleccionar las 2 caracteristicas con mayor correlacion
caracteristica_1 = lista_correlaciones[0][0]
caracteristica_2 = lista_correlaciones[1][0]
correlacion_1 = lista_correlaciones[0][1]
correlacion_2 = lista_correlaciones[1][1]

print("\n" + "=" * 60)
print("CARACTERISTICAS SELECCIONADAS PARA TRABAJAR EN 2D:")
print("=" * 60)
print(f"1. {caracteristica_1}")
print(f"   Correlacion con status: {correlacion_1:.4f}")
print(f"   (Mas alta correlacion)")

print(f"\n2. {caracteristica_2}")
print(f"   Correlacion con status: {correlacion_2:.4f}")
print(f"   (Segunda mas alta correlacion)")

# Crear un nuevo DataFrame con solo estas 2 caracteristicas
X_2d = df[[caracteristica_1, caracteristica_2]]

print(f"\nNuevo conjunto de caracteristicas (X_2d): {X_2d.shape}")
print(f"Variable objetivo (y): {y.shape}")

# ====================================================================
# INCISO 3: ESCALADO DE CARACTERISTICAS
# ====================================================================
print("\n--- INCISO 3: ESCALADO DE CARACTERISTICAS ---")
print("\nImportancia del escalado (segun las instrucciones):")
print("1. El Descenso del Gradiente tarda mucho en converger si")
print("   los datos no estan escalados")
print("2. KNN no funciona bien si una dimension tiene magnitudes")
print("   mucho mayores que otra")

print("\nMetodo seleccionado: Normalizacion Min-Max")
print("Formula: X_escalado = (X - min(X)) / (max(X) - min(X))")
print("Resultado: Todos los valores entre 0 y 1")

# Guardar los valores originales para referencia
print("\nValores originales de las caracteristicas:")
for columna in [caracteristica_1, caracteristica_2]:
    min_val = X_2d[columna].min()
    max_val = X_2d[columna].max()
    media_val = X_2d[columna].mean()
    print(f"\n{columna}:")
    print(f"  Minimo: {min_val:.4f}")
    print(f"  Maximo: {max_val:.4f}")
    print(f"  Media:  {media_val:.4f}")
    print(f"  Rango:  {max_val - min_val:.4f}")

# Aplicar escalado Min-Max
print("\nAplicando escalado Min-Max...")
X_2d_original = X_2d.copy()  # Guardar copia de los datos originales

for columna in [caracteristica_1, caracteristica_2]:
    min_val = X_2d[columna].min()
    max_val = X_2d[columna].max()
    
    # Aplicar la formula de normalizacion Min-Max
    X_2d[columna] = (X_2d[columna] - min_val) / (max_val - min_val)

print("\nValores despues del escalado:")
for columna in [caracteristica_1, caracteristica_2]:
    min_val_esc = X_2d[columna].min()
    max_val_esc = X_2d[columna].max()
    print(f"\n{columna}:")
    print(f"  Minimo despues de escalar: {min_val_esc:.4f}")
    print(f"  Maximo despues de escalar: {max_val_esc:.4f}")
    print(f"  Verificacion: {min_val_esc == 0.0 and max_val_esc == 1.0}")

print("\nEscalado completado. Todas las caracteristicas estan ahora en el rango [0, 1]")

# ====================================================================
# INCISO 4: DIVISION DE DATOS (80% TRAINING, 20% TESTING)
# ====================================================================
print("\n--- INCISO 4: DIVISION DE DATOS ---")
print("\nProposito: Separar los datos en conjunto de entrenamiento y prueba")
print("           para evaluar correctamente el rendimiento del modelo.")

print("\nProporcion: 80% para entrenamiento, 20% para prueba")

# Calcular los tamanos de cada conjunto
n_total = len(X_2d)
n_entrenamiento = int(n_total * 0.8)  # 80% para entrenamiento
n_prueba = n_total - n_entrenamiento  # 20% para prueba

print(f"\nCalculos:")
print(f"Total de muestras: {n_total}")
print(f"80% para entrenamiento: {n_entrenamiento} muestras")
print(f"20% para prueba: {n_prueba} muestras")

# Dividir los datos manualmente
print("\nDividiendo los datos...")

# Tomar las primeras n_entrenamiento muestras para entrenamiento
X_train = X_2d.iloc[:n_entrenamiento]
y_train = y.iloc[:n_entrenamiento]

# Tomar las restantes para prueba
X_test = X_2d.iloc[n_entrenamiento:]
y_test = y.iloc[n_entrenamiento:]

print("Division completada.")
print(f"\nConjunto de entrenamiento:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")

print(f"\nConjunto de prueba:")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

# Verificar la distribucion de clases en cada conjunto
print("\nDistribucion de clases en cada conjunto:")

print("\nEntrenamiento (y_train):")
conteo_train = y_train.value_counts()
for clase, cantidad in conteo_train.items():
    porcentaje = (cantidad / len(y_train)) * 100
    nombre_clase = "legitimate" if clase == 0 else "phishing"
    print(f"  Clase {clase} ({nombre_clase}): {cantidad} muestras ({porcentaje:.1f}%)")

print("\nPrueba (y_test):")
conteo_test = y_test.value_counts()
for clase, cantidad in conteo_test.items():
    porcentaje = (cantidad / len(y_test)) * 100
    nombre_clase = "legitimate" if clase == 0 else "phishing"
    print(f"  Clase {clase} ({nombre_clase}): {cantidad} muestras ({porcentaje:.1f}%)")

# ====================================================================
# RESUMEN FINAL
# ====================================================================
print("\n" + "=" * 70)
print("RESUMEN FINAL DE LA PREPARACION DE DATOS")
print("=" * 70)

print("\n1. CARGA Y LIMPIEZA:")
print(f"   - Dataset original: 11430 muestras, 89 caracteristicas")
print(f"   - Columnas eliminadas: url, length_url")
print(f"   - Dataset despues de limpieza: {df.shape[1]} caracteristicas")

print("\n2. CODIFICACION:")
print("   - Variable 'status' convertida de texto a numeros")
print("   - legitimate -> 0")
print("   - phishing -> 1")
print(f"   - Distribucion: {conteo_clases[0]} legitimate, {conteo_clases[1]} phishing")

print("\n3. SELECCION DE CARACTERISTICAS:")
print(f"   - Caracteristica 1: {caracteristica_1}")
print(f"     Correlacion con status: {correlacion_1:.4f}")
print(f"   - Caracteristica 2: {caracteristica_2}")
print(f"     Correlacion con status: {correlacion_2:.4f}")

print("\n4. ESCALADO:")
print("   - Metodo: Normalizacion Min-Max")
print("   - Rango final: [0, 1] para ambas caracteristicas")
print("   - Importante para: Descenso del Gradiente y KNN")

print("\n5. DIVISION DE DATOS:")
print(f"   - Entrenamiento: {X_train.shape[0]} muestras (80%)")
print(f"   - Prueba: {X_test.shape[0]} muestras (20%)")
print("   - Proporcion de clases mantenida en ambos conjuntos")

print("\n" + "=" * 70)
print("DATOS PREPARADOS PARA LOS SIGUIENTES ALGORITMOS:")
print("=" * 70)
print("\nVariables disponibles para usar:")
print("  X_train, y_train - Para entrenar modelos")
print("  X_test, y_test   - Para evaluar modelos")
print(f"\nCaracteristicas usadas: {caracteristica_1}, {caracteristica_2}")
print("\nLos datos estan listos para:")
print("  - Regresion Lineal")
print("  - Regresion Logistica")
print("  - K-Nearest Neighbors (KNN)")

# Guardar una copia de los datos preparados si es necesario
print("\nNota: Los datos estan listos en memoria.")
print("      Puedes acceder a ellos usando las variables:")
print("      X_train, X_test, y_train, y_test")