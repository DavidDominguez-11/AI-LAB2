# 🔒 Detección de Phishing con Machine Learning

**Laboratorio 2 - CC3045 Inteligencia Artificial**

Implementación desde cero de algoritmos de clasificación para detectar sitios web maliciosos (phishing).

---

## 📋 Descripción

Este proyecto implementa **manualmente** (sin librerías de ML) algoritmos de clasificación binaria para detectar phishing en sitios web, demostrando comprensión profunda de:

- Regresión Logística con Descenso del Gradiente
- K-Nearest Neighbors (KNN)
- Funciones de costo y optimización
- Evaluación y comparación de modelos

---

## 🗂️ Estructura del Proyecto

```
.
├── tasks_2_3_4.ipynb          # Notebook principal con implementaciones
├── dataset_phishing_processed.csv  # Dataset preprocesado
└── README.md
```

---

## 🚀 Implementación

### **Task 2: Preparación de Datos**
- Carga y limpieza del dataset
- Selección de features por correlación
- Normalización con StandardScaler
- Split 80/20 (train/test)

### **Task 3: Implementación Manual**

#### Regresión Logística
- ✅ Función sigmoide
- ✅ Log Loss (Binary Cross-Entropy)
- ✅ Gradient Descent
- ✅ Visualización de curva de aprendizaje
- ✅ Decision boundary

#### K-Nearest Neighbors (KNN)
- ✅ Distancia euclidiana
- ✅ Votación por mayoría
- ✅ Visualización de regiones de decisión

### **Task 4: Benchmark con sklearn**
- Comparación con implementaciones profesionales
- Métricas: Accuracy, Precision, Recall
- Análisis de Falsos Positivos vs Falsos Negativos
- Recomendación de modelo óptimo

---

## 📊 Dataset

**Web Page Phishing Detection Dataset** (Kaggle)

- **Muestras:** 11,430
- **Features seleccionados:** 
  - `nb_www`: Número de ocurrencias de "www"
  - `ratio_digits_url`: Ratio de dígitos en la URL
- **Clases:** 
  - 0 = Legítimo (50%)
  - 1 = Phishing (50%)

---

## 🛠️ Tecnologías

- Python 3.12
- NumPy (operaciones matriciales)
- Pandas (manipulación de datos)
- Matplotlib (visualizaciones)
- Seaborn (matrices de confusión)
- scikit-learn (solo para benchmark)

---

## 📈 Resultados

Los modelos implementados manualmente alcanzaron **precisión comparable** a las implementaciones de sklearn, validando la correctitud de las implementaciones.

**Métrica prioritaria:** RECALL
- En phishing, minimizar Falsos Negativos es crítico
- Es preferible bloquear sitios legítimos que dejar pasar ataques

---

## 🎯 Conceptos Clave

### Regresión Logística
- ❌ No usar MSE (función no convexa con sigmoide)
- ✅ Usar Log Loss (convexa, garantiza convergencia)
- Gradiente descendente encuentra mínimo global

### KNN
- ⚠️ Sensible a datos desbalanceados
- ⚠️ K > clases minoritarias = sesgo hacia clase mayoritaria
- ✅ Requiere escalado de features (distancias)

### Overfitting
- Polinomios de alto grado → pérdida ~0 en train
- Pero error alto en producción → modelando ruido
- Solución: validación cruzada, regularización

---

## 🔧 Instalación

```bash
# Clonar repositorio
git clone https://github.com/DavidDominguez-11/AI-LAB2.git

# Instalar dependencias
pip install numpy pandas matplotlib seaborn scikit-learn

# Abrir notebook
jupyter notebook tasks_2_3_4.ipynb
```

---

## 📝 Uso

1. Ejecutar celdas en orden secuencial
2. Task 2: Preparación de datos
3. Task 3: Entrenar modelos manuales
4. Task 4: Comparar con sklearn

---

## 👥 Autores

- David Domínguez- 23712
- Gabriel Bran - 23590

**Curso:** CC3045 - Inteligencia Artificial  
**Universidad del Valle de Guatemala**  
**Año:** 2026

---

## 📄 Licencia

Proyecto académico - Universidad del Valle de Guatemala

---

## 🙏 Agradecimientos

- Dataset: Kaggle - Web Page Phishing Detection
- Material del curso: Samuel Chávez
- Referencias: CS221 Stanford, Machine Learning @ Berkeley