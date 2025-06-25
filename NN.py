import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# DATA LOADING & PREP
# ======================
X_train = pd.read_csv('X_train.csv').values
y_train = pd.read_csv('y_train.csv').values.ravel()  # Convert to 1D array
X_test_1 = pd.read_csv('X_test_1.csv').values
X_test_2 = pd.read_csv('X_test_2.csv').values
y_test_2_reduced = pd.read_csv('y_test_2_reduced.csv').values.ravel()

# ======================
# FEATURE SELECTION
# ======================
top_features = [
    111, 141, 47, 187, 221, 76, 23, 239, 88, 68,
    142, 160, 132, 58, 21, 45, 53, 39, 214, 34,
    29, 126, 13, 117, 97, 206, 71, 60, 247, 290,
    223, 48, 69, 122, 190, 86, 268, 5, 10, 6,
    8, 12, 24, 17, 26, 21, 14, 4, 25, 19,
    20, 27, 3, 9, 15, 16, 18, 22, 30, 31,
    32, 33, 35, 36, 37, 38, 40, 41, 42, 43,
    44, 46, 49, 50, 51, 52, 54, 55, 56, 57,
    59, 61, 62, 63, 64, 65, 66, 67, 70, 72
]

# Process features
top_features = list(set(top_features))
max_feature_index = X_train.shape[1] - 1
top_features = [f for f in top_features if f <= max_feature_index][:100]
print(f"\nUsing {len(top_features)} features in the model")

# Apply feature selection
X_train = X_train[:, top_features]
X_test_1 = X_test_1[:, top_features]
X_test_2 = X_test_2[:, top_features]

# ======================
# DATA PROCESSING
# ======================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test_1 = scaler.transform(X_test_1)
X_test_2 = scaler.transform(X_test_2)

# Prepare labels - find all existing classes first
all_classes = np.unique(np.concatenate([y_train, y_test_2_reduced]))
num_classes = len(all_classes)
print(f"\nDetected {num_classes} classes in all datasets: {sorted(all_classes)}")

# Convert to categorical using all classes
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_2_reduced = tf.keras.utils.to_categorical(y_test_2_reduced, num_classes)

# Split test set 2
X_test_2_labelled = X_test_2[:202]
y_test_2_reduced = y_test_2_reduced[:202]

# Train/val split (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# ======================
# IMPROVED CLASS WEIGHTS
# ======================
class_counts = np.bincount(np.argmax(y_train, axis=1))
total_samples = len(y_train)
class_weights = {
    i: (total_samples / (num_classes * count)) ** 0.5  # Smoother weighting
    for i, count in enumerate(class_counts)
}

# ======================
# ENHANCED MODEL ARCHITECTURE
# ======================
def create_model(input_dim, output_dim):
    model = Sequential([
        Dense(512, activation='swish', input_shape=(input_dim,), kernel_regularizer=l2(0.002)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='swish', kernel_regularizer=l2(0.002)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='swish'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(output_dim, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc', multi_label=True),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    return model

# Initialise model with correct dimensions
input_dim = X_train.shape[1]
model = create_model(input_dim=input_dim, output_dim=num_classes)

# ======================
# IMPROVED CALLBACKS
# ======================
callbacks = [
    EarlyStopping(monitor='val_auc', patience=20, mode='max', 
                 restore_best_weights=True, min_delta=0.002),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=8, min_lr=1e-6),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_auc')
]

# ======================
# TRAINING WITH CLASS WEIGHTS
# ======================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ======================
# IMPROVED EVALUATION
# ======================
def evaluate_model(X_test, y_test, set_name):
    results = {}
    
    # Basic metrics
    test_loss, test_acc, test_auc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)
    results['metrics'] = {
        'loss': test_loss,
        'accuracy': test_acc,
        'auc': test_auc,
        'precision': test_prec,
        'recall': test_rec
    }
    
    # Classification report
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Get only classes present in this dataset
    present_classes = np.unique(y_true_classes)
    class_labels = [f'Class {i}' for i in range(num_classes) if i in present_classes]
    
    print(f"\n=== {set_name} Evaluation ===")
    print(f"Accuracy: {test_acc:.4f}")
    
    report = classification_report(
        y_true_classes, y_pred_classes,
        labels=present_classes,
        target_names=class_labels,
        digits=4,
        output_dict=True,
        zero_division=0  # Handle division by zero
    )
    
    print(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_true_classes, y_pred_classes,
        labels=present_classes,
        target_names=class_labels,
        digits=4,
        zero_division=0
    ))
    
    # Confusion matrix
    plt.figure(figsize=(12,10))
    cm = confusion_matrix(y_true_classes, y_pred_classes, labels=present_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present_classes,
                yticklabels=present_classes)
    plt.title(f'Confusion Matrix ({set_name})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return results

# Evaluate on validation and test sets
print("\n" + "="*50)
val_results = evaluate_model(X_val, y_val, "Validation Set")
print("\n" + "="*50)
test_results = evaluate_model(X_test_2_labelled, y_test_2_reduced, "Test Set 2 (Labelled)")

# ======================
# PREDICTIONS FOR TEST SET 1
# ======================
test1_pred = model.predict(X_test_1)
test1_classes = np.argmax(test1_pred, axis=1)
pd.DataFrame(test1_classes).to_csv('test1_predictions.csv', index=False)
print(f"\nSaved predictions for Test Set 1 (1000 samples) to 'test1_predictions.csv'")

# ======================
# TRAINING HISTORY PLOTS
# ======================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()

# ======================
# PREDICTIONS SUBMISSION FOR BOTH TEST SETS
# ======================
def generate_predictions_submission():
    """
    Generates predictions for both test sets in the required format for submission.
    Returns predictions as NumPy arrays of shape (1000, 28) and (1818, 28).
    """
    # Generate predictions for test set 1 (1000 samples)
    preds_1 = model.predict(X_test_1, verbose=0)
    
    # Generate predictions for test set 2 (1818 unlabelled samples)
    X_test_2_unlabelled = X_test_2[202:]  # Extract unlabelled portion
    preds_2 = model.predict(X_test_2_unlabelled, verbose=0)
    
    # Ensure predictions have correct dimensions
    if preds_1.shape[1] < 28:
        # Pad with zeros if needed
        padding = np.zeros((preds_1.shape[0], 28 - preds_1.shape[1]))
        preds_1 = np.hstack([preds_1, padding])
    
    if preds_2.shape[1] < 28:
        # Pad with zeros if needed
        padding = np.zeros((preds_2.shape[0], 28 - preds_2.shape[1]))
        preds_2 = np.hstack([preds_2, padding])
    
    return preds_1, preds_2

# Generate and save predictions
preds_1, preds_2 = generate_predictions_submission()

# Save to .npy files
np.save('preds_1.npy', preds_1)
np.save('preds_2.npy', preds_2)

# Create zip file
import zipfile
with zipfile.ZipFile('Gazebo.zip', 'w') as zipf:
    zipf.write('preds_1.npy')
    zipf.write('preds_2.npy')

print("\nPredictions submission files created successfully:")
print("- preds_1.npy (1000x28 predictions for Test Set 1)")
print("- preds_2.npy (1818x28 predictions for Test Set 2 unlabelled portion)")