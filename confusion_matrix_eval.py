import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

# Load the confusion matrix
conf_mat = np.load("confusion_matrix.npy")
print(f"Confusion matrix shape: {conf_mat.shape}")

# Load class labels from the model config
config_path = "videomae_model/config.json"  # or "vivit_model/config.json" depending on which model you used
with open(config_path, 'r') as f:
    config = json.load(f)

# Get class names from id2label mapping
id2label = config['id2label']
class_names = [id2label[str(i)] for i in range(len(id2label))]

# Calculate overall accuracy
accuracy = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
print(f"Overall accuracy: {accuracy * 100:.2f}%")

# Calculate per-class metrics
class_accuracy = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
class_precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)

# Handle division by zero
class_accuracy = np.nan_to_num(class_accuracy)
class_precision = np.nan_to_num(class_precision)

# Get top and bottom performing classes
top_classes_idx = np.argsort(class_accuracy)[-10:]
bottom_classes_idx = np.argsort(class_accuracy)[:10]

print("\nTop 10 classes by accuracy:")
for idx in reversed(top_classes_idx):
    print(f"  {class_names[idx]}: {class_accuracy[idx] * 100:.2f}%")

print("\nBottom 10 classes by accuracy:")
for idx in bottom_classes_idx:
    print(f"  {class_names[idx]}: {class_accuracy[idx] * 100:.2f}%")

# Create a heatmap of the confusion matrix (for the most confused classes)
# First, find the most confused pairs
confusion = conf_mat.copy()
np.fill_diagonal(confusion, 0)  # Zero out the diagonal
most_confused_flat = np.argsort(confusion.flatten())[-20:]  # Top 20 confusions
most_confused = [(idx // conf_mat.shape[0], idx % conf_mat.shape[0]) for idx in most_confused_flat]

# Create a smaller confusion matrix with just the most confused classes
confused_classes = set()
for i, j in most_confused:
    confused_classes.add(i)
    confused_classes.add(j)
confused_classes = sorted(list(confused_classes))

# Extract the submatrix
sub_conf_mat = conf_mat[np.ix_(confused_classes, confused_classes)]
sub_class_names = [class_names[i] for i in confused_classes]

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(sub_conf_mat, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sub_class_names, yticklabels=sub_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Most Confused Classes')
plt.tight_layout()
plt.savefig('confusion_matrix_most_confused.png', dpi=300)

# Also create a normalized version
plt.figure(figsize=(12, 10))
row_sums = sub_conf_mat.sum(axis=1)
norm_sub_conf_mat = sub_conf_mat / row_sums[:, np.newaxis]
sns.heatmap(norm_sub_conf_mat, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=sub_class_names, yticklabels=sub_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix for Most Confused Classes')
plt.tight_layout()
plt.savefig('confusion_matrix_most_confused_normalized.png', dpi=300)

print("\nVisualization saved to confusion_matrix_most_confused.png")
print("Normalized visualization saved to confusion_matrix_most_confused_normalized.png")

# If you want to see the full confusion matrix (warning: might be very large)
if conf_mat.shape[0] <= 50:  # Only show full matrix if it's reasonably sized
    plt.figure(figsize=(20, 16))
    sns.heatmap(conf_mat, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Full Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_full.png', dpi=300)
    print("Full confusion matrix saved to confusion_matrix_full.png")