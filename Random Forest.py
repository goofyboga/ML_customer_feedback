import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# The Code should be used directly after the EDA.py
# The Code should be used directly after the EDA.py
# The Code should be used directly after the EDA.py

# Data Preprocessing

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #X_train is defined in EDA.py
X_test_2_scaled = scaler.transform(X_test_2) #X_test_2 is defined in EDA.py

# Feature Selection
# Use best features selected from previous Logistic Regression step
current_features = best_features  #best_features is defined in EDA.py


# Validation Set Split
X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(
    X_train_scaled[:, current_features],
    y_train.values.ravel(), #y_train is defined in EDA.py
    test_size=0.2,
    random_state=42
)


# Apply SMOTE to training data only

smote = SMOTE(random_state=42, k_neighbors=1, sampling_strategy='not majority')
X_train_rf_resampled, y_train_rf_resampled = smote.fit_resample(X_train_rf, y_train_rf)


# Train Random Forest Model

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=3,
    class_weight='balanced',
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_rf_resampled, y_train_rf_resampled)


# Validation Set Evaluation
y_pred_val = rf_model.predict(X_val_rf)

print("\n📊 Random Forest Performance on Validation Set:")
print("Accuracy:", accuracy_score(y_val_rf, y_pred_val))
print("Macro F1 Score:", f1_score(y_val_rf, y_pred_val, average='macro'))

# Print full classification report
report_val = classification_report(y_val_rf, y_pred_val, output_dict=True)
report_val_df = pd.DataFrame(report_val).transpose().round(4)
pd.set_option('display.max_rows', None)
print("\nClassification Report (Validation Set):")
print(report_val_df.to_string())


# Test Set 2 Evaluation
X_test_2_eval = X_test_2_scaled[:, current_features][:202]
y_pred_test2 = rf_model.predict(X_test_2_eval)

print("\n📊 Random Forest Performance on Test Set 2 (First 202 samples):")
print("Accuracy:", accuracy_score(y_test_2_reduced, y_pred_test2))
print("Macro F1 Score:", f1_score(y_test_2_reduced, y_pred_test2, average='macro'))

# Print full classification report
report_test2 = classification_report(y_test_2_reduced, y_pred_test2, output_dict=True)
report_test2_df = pd.DataFrame(report_test2).transpose().round(4)
print("\nClassification Report (Test Set 2):")
print(report_test2_df.to_string())