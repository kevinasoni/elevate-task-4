import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('data.csv')

# Drop unwanted columns
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')

# Convert target to binary (Malignant=1, Benign=0)
df['diagnosis'] = df['diagnosis'].map(lambda x: 1 if x.upper().startswith('M') else 0)

# Split features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit logistic regression model
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train_scaled, y_train)

# Predict probabilities on test set
y_probs = logreg.predict_proba(X_test_scaled)[:, 1]

# Predict with default threshold 0.5
y_pred = (y_probs >= 0.5).astype(int)

# Evaluation metrics at default threshold
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)

print("Confusion Matrix (threshold=0.5):\n", cm)
print(f"Precision (threshold=0.5): {precision:.3f}")
print(f"Recall (threshold=0.5): {recall:.3f}")
print(f"ROC-AUC Score: {roc_auc:.3f}")

# Find best threshold by maximizing F1 score
precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-6)
best_idx = np.argmax(f1_scores)
best_threshold = pr_thresholds[best_idx if best_idx < len(pr_thresholds) else -1]

# Predict with best threshold
y_pred_best = (y_probs >= best_threshold).astype(int)
cm_best = confusion_matrix(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)

print(f"\nBest Threshold by F1 Score: {best_threshold:.3f}")
print("Confusion Matrix (best threshold):\n", cm_best)
print(f"Precision (best threshold): {precision_best:.3f}")
print(f"Recall (best threshold): {recall_best:.3f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1],'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png')
plt.show()

# Sigmoid function plot (for explanation)
sigmoid_x = np.linspace(-10, 10, 100)
sigmoid_y = 1 / (1 + np.exp(-sigmoid_x))
plt.figure(figsize=(8,6))
plt.plot(sigmoid_x, sigmoid_y)
plt.title('Sigmoid Function')
plt.xlabel('Input (z)')
plt.ylabel('Output σ(z)')
plt.grid(True)
plt.savefig('sigmoid_function.png')
plt.show()
