import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv").values.ravel()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


k_values = range(1, 31, 2)  # 1, 3, 5, ..., 29
acc_scores = []
macro_f1_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_val_scaled)

    acc_scores.append(accuracy_score(y_val, y_pred))
    macro_f1_scores.append(f1_score(y_val, y_pred, average='macro'))

plt.figure(figsize=(10, 6))
plt.plot(k_values, acc_scores, marker='o', label='Accuracy')
plt.plot(k_values, macro_f1_scores, marker='x', label='Macro F1')
plt.xlabel("K Value")
plt.ylabel("Score")
plt.title("KNN Model Performance for Different K")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_k_search_performance.png")

