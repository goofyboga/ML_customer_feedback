import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load datasets
X_train = pd.read_csv('C:/Users/Arum Titan/Downloads/X_train.csv')
y_train = pd.read_csv('C:/Users/Arum Titan/Downloads/y_train.csv')
X_test_1 = pd.read_csv('C:/Users/Arum Titan/Downloads/X_test_1.csv')
X_test_2 = pd.read_csv('C:/Users/Arum Titan/Downloads/X_test_2.csv')

# Handle missing values
missing_counts = X_train.isnull().sum()
print("Missing Values:")
print(missing_counts[missing_counts > 0])

# SMOTE to balance classes
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train.values.ravel())
print(f"New class distribution after SMOTE: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")

# Correlation filtering
corr_matrix = X_train_resampled.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
X_train_filtered = X_train_resampled.drop(to_drop, axis=1)
X_test_1_filtered = X_test_1.drop(to_drop, axis=1)
X_test_2_filtered = X_test_2.drop(to_drop, axis=1)
print(f"Features after correlation filter: {X_train_filtered.shape[1]}")

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_test_1_scaled = scaler.transform(X_test_1_filtered)
X_test_2_scaled = scaler.transform(X_test_2_filtered)

# PCA (keep 95% variance)
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_1_pca = pca.transform(X_test_1_scaled)
X_test_2_pca = pca.transform(X_test_2_scaled)
print(f"Number of PCA components: {X_train_pca.shape[1]}")

# Train-validation split
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_pca, y_train_resampled, test_size=0.2, stratify=y_train_resampled, random_state=42
)

# RFE with Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=log_reg, n_features_to_select=50)
X_train_rfe = rfe.fit_transform(X_train_split, y_train_split)
X_val_rfe = rfe.transform(X_val)
X_test_1_rfe = rfe.transform(X_test_1_pca)
X_test_2_rfe = rfe.transform(X_test_2_pca)
print(f"Features after RFE: {X_train_rfe.shape[1]}")

# Train final Logistic Regression
log_reg_final = LogisticRegression(max_iter=1000, random_state=42)
log_reg_final.fit(X_train_rfe, y_train_split)

# Validation performance
y_val_pred = log_reg_final.predict(X_val_rfe)
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# Predictions for X_test_1 and X_test_2
y_test_1_pred = log_reg_final.predict(X_test_1_rfe)
y_test_2_pred = log_reg_final.predict(X_test_2_rfe)

# Print predictions nicely
print("\nPredictions for X_test_1:")
print(pd.DataFrame({'Id': X_test_1.index, 'Predicted_department_id': y_test_1_pred}))

print("\nPredictions for X_test_2:")
print(pd.DataFrame({'Id': X_test_2.index, 'Predicted_department_id': y_test_2_pred}))

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=5)
grid.fit(X_train_rfe, y_train_split)
print("Best C:", grid.best_params_)
