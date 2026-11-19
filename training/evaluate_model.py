# Comprehensive Model Evaluation

import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
)
from preprocess_data import test_generator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "resnet50_pneumonia.h5")

print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# -------------------------
# Evaluate overall accuracy
# -------------------------
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss:     {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

# -------------------------
# Per-class metrics
# -------------------------
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = test_generator.classes
class_names = list(test_generator.class_indices.keys())

print(f"\nClass mapping: {test_generator.class_indices}")
print(f"\n{'='*50}")
print("CLASSIFICATION REPORT")
print('='*50)
print(classification_report(true_classes, predicted_classes, target_names=class_names))

print(f"{'='*50}")
print("CONFUSION MATRIX")
print('='*50)
cm = confusion_matrix(true_classes, predicted_classes)
print(f"\n              Predicted")
print(f"              {'  '.join(class_names)}")
for i, row in enumerate(cm):
    print(f"Actual {class_names[i]:>10s}  {'  '.join(str(v).rjust(6) for v in row)}")

# -------------------------
# Per-class accuracy
# -------------------------
print(f"\n{'='*50}")
print("PER-CLASS ACCURACY")
print('='*50)
for i, name in enumerate(class_names):
    mask = true_classes == i
    class_acc = np.mean(predicted_classes[mask] == i)
    total = np.sum(mask)
    correct = np.sum(predicted_classes[mask] == i)
    print(f"  {name:>10s}: {class_acc*100:.1f}% ({correct}/{total})")

# -------------------------
# F1 Score & Recall
# -------------------------
print(f"\n{'='*50}")
print("F1 SCORE & RECALL")
print('='*50)

# Per-class
f1_per_class = f1_score(true_classes, predicted_classes, average=None)
recall_per_class = recall_score(true_classes, predicted_classes, average=None)
precision_per_class = precision_score(true_classes, predicted_classes, average=None)

for i, name in enumerate(class_names):
    print(f"  {name:>10s}  ->  F1: {f1_per_class[i]:.4f}  |  Recall: {recall_per_class[i]:.4f}  |  Precision: {precision_per_class[i]:.4f}")

# Weighted averages (accounts for class imbalance)
f1_weighted = f1_score(true_classes, predicted_classes, average='weighted')
recall_weighted = recall_score(true_classes, predicted_classes, average='weighted')
precision_weighted = precision_score(true_classes, predicted_classes, average='weighted')

# Macro averages
f1_macro = f1_score(true_classes, predicted_classes, average='macro')
recall_macro = recall_score(true_classes, predicted_classes, average='macro')
precision_macro = precision_score(true_classes, predicted_classes, average='macro')

print(f"\n  {'Weighted':>10s}  ->  F1: {f1_weighted:.4f}  |  Recall: {recall_weighted:.4f}  |  Precision: {precision_weighted:.4f}")
print(f"  {'Macro':>10s}  ->  F1: {f1_macro:.4f}  |  Recall: {recall_macro:.4f}  |  Precision: {precision_macro:.4f}")

# -------------------------
# Save metrics to JSON
# -------------------------
metrics = {
    "accuracy": float(accuracy),
    "loss": float(loss),
    "f1_score": {
        "weighted": float(f1_weighted),
        "macro": float(f1_macro),
        "per_class": {name: float(f1_per_class[i]) for i, name in enumerate(class_names)},
    },
    "recall": {
        "weighted": float(recall_weighted),
        "macro": float(recall_macro),
        "per_class": {name: float(recall_per_class[i]) for i, name in enumerate(class_names)},
    },
    "precision": {
        "weighted": float(precision_weighted),
        "macro": float(precision_macro),
        "per_class": {name: float(precision_per_class[i]) for i, name in enumerate(class_names)},
    },
    "confusion_matrix": cm.tolist(),
    "class_names": class_names,
}

METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved to: {METRICS_PATH}")
print("Evaluation complete!")