"""
Speed benchmark for candidate algorithms on Forest Cover Type subsample.
Measures 1 training pass on N=40,000 samples with 16 CPU cores.
"""
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

# Load data
df = pd.read_csv(r"c:\Users\DELL\Downloads\ANN\covtype.data.gz", header=None)
X = df.iloc[:, :54].values
y = df.iloc[:, 54].values

# Stratified subsample of N=50,000
X_sub, _, y_sub, _ = train_test_split(X, y, train_size=50000, stratify=y, random_state=42)

# 80/20 train/val split
X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, train_size=40000, stratify=y_sub, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

models = {
    "LogisticRegression": (LogisticRegression(max_iter=200, random_state=42, n_jobs=-1), X_train_scaled, X_val_scaled),
    "DecisionTree": (DecisionTreeClassifier(max_depth=15, random_state=42), X_train, X_val),
    "RandomForest": (RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1), X_train, X_val),
    "LightGBM": (LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1), X_train, X_val),
    "MLP": (MLPClassifier(hidden_layer_sizes=(100,), max_iter=40, random_state=42, early_stopping=True), X_train_scaled, X_val_scaled),
}

print(f"{'Algorithm':20s} | {'Train Time (s)':15s} | {'Val Accuracy':12s} | {'Val Macro-F1':12s}")
print("-" * 65)

from sklearn.metrics import accuracy_score, f1_score

for name, (clf, X_tr, X_va) in models.items():
    t0 = time.time()
    clf.fit(X_tr, y_train)
    t_train = time.time() - t0
    y_pred = clf.predict(X_va)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    print(f"{name:20s} | {t_train:15.2f} | {acc:12.4f} | {f1:12.4f}")
