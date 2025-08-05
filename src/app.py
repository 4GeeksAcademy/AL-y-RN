"""
Proyecto: Predicción de Diabetes
Modelos: Árbol de Decisión, Regresión Logística, Red Neuronal, Random Forest, AdaBoost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report


# 1. Cargar datos
url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
df = pd.read_csv(url)
print("Datos cargados. Shape:", df.shape)

# 2. Preprocesamiento básico
df = df.drop_duplicates().reset_index(drop=True)

# Función para detectar variables binarias
def is_binary(df, col):
    unique = df[col].dropna().unique()
    return set(unique).issubset({0, 1})

# Clasificar tipos de variables
normal_cols = []
non_normal_cols = []
binary_cols = []

for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        if is_binary(df, col):
            binary_cols.append(col)
        else:
            _, p = shapiro(df[col])
            if p > 0.05:
                normal_cols.append(col)
            else:
                non_normal_cols.append(col)

print(f"Variables binarias: {binary_cols}")
print(f"Variables no normales: {non_normal_cols}")

# 3. Separar características y objetivo
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# 4. Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Función para evaluar modelos y guardar resultados
def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    return {
        'model': name,
        'f1_class_0': report['0']['f1-score'],
        'f1_class_1': report['1']['f1-score'],
        'accuracy': report['accuracy']
    }

# Lista para almacenar resultados
results = []

# --- 7. Árbol de Decisión ---
print("\nEntrenando Árbol de Decisión...")
dt = DecisionTreeClassifier(random_state=666)
params_dt = {
    'class_weight': [{0:0.05, 1:0.95}, {0:0.1, 1:0.9}, {0:0.2, 1:0.8}],
    'max_depth': [None, 3, 5, 10],
    'min_samples_leaf': [5, 10, 20, 50],
    'criterion': ['gini', 'entropy']
}
grid_dt = GridSearchCV(dt, params_dt, cv=5, scoring='f1', n_jobs=-1)
results.append(evaluate_model(grid_dt, X_train, X_test, y_train, y_test, "Decision Tree"))

import pickle
with open("Modelo_ArbolD.pkl", 'wb') as j:
    pickle.dump("best_dt", j)
