# Remove the unnamed column and re-run pipeline (skip features with all NaN)
from sklearn.impute import SimpleImputer

# Drop 'Unnamed: 32' and 'id' columns
if 'Unnamed: 32' in df.columns:
    df = df.drop(['Unnamed: 32'], axis=1)
if 'id' in df.columns:
    df = df.drop(['id'], axis=1)

# Convert target
label_col = 'diagnosis'
if df[label_col].dtype == 'O':
    df[label_col] = df[label_col].map(lambda x: 1 if x.upper().startswith('M') else 0)

feature_cols = [col for col in df.columns if col != label_col]
X = df[feature_cols]
y = df[label_col]

# Impute any missing values (mean imputation for simplicity, should be safe)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve
import numpy as np

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Logistic Regression
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train_scaled, y_train)

# Predict probabilities
y_probs = logreg.predict_proba(X_test_scaled)[:, 1]
# Default threshold = 0.5
y_pred = (y_probs >= 0.5).astype(int)

# Evaluation metrics
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)

# Compute ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_probs)
# Compute Precision-Recall curve
precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_probs)

# Find best threshold by maximizing F1
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_f1_idx = np.argmax(f1s)
best_threshold = pr_thresholds[best_f1_idx if best_f1_idx < len(pr_thresholds) else -1]

# Evaluate at best threshold
y_pred_best = (y_probs >= best_threshold).astype(int)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
cm_best = confusion_matrix(y_test, y_pred_best)

# Sigmoid function explanation
sigmoid_inputs = np.linspace(-10, 10, 100)
sigmoid_outputs = 1 / (1 + np.exp(-sigmoid_inputs))

results = {
    'confusion_matrix_default': cm.tolist(),
    'precision_default': precision,
    'recall_default': recall,
    'roc_auc': roc_auc,
    'best_threshold': best_threshold,
    'confusion_matrix_best': cm_best.tolist(),
    'precision_best': precision_best,
    'recall_best': recall_best,
    'fpr': fpr.tolist(),
    'tpr': tpr.tolist(),
    'roc_thresholds': roc_thresholds.tolist(),
    'precisions': precisions.tolist(),
    'recalls': recalls.tolist(),
    'pr_thresholds': pr_thresholds.tolist(),
    'sigmoid_x': sigmoid_inputs.tolist(),
    'sigmoid_y': sigmoid_outputs.tolist(),
    'coef': logreg.coef_.tolist(),
    'intercept': logreg.intercept_.tolist(),
    'feature_names': feature_cols
}
results
