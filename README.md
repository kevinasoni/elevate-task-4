# elevate-task-4
# Logistic Regression Binary Classifier

This project builds a binary classifier using Logistic Regression on the Breast Cancer Wisconsin dataset to classify tumors as malignant or benign.

## Tools & Libraries
- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib
- NumPy

## Project Overview
- Dataset loaded and preprocessed (removing irrelevant columns, imputing missing values).
- Target variable converted into binary (Malignant=1, Benign=0).
- Train/test split with stratification to preserve class distribution.
- Features standardized using StandardScaler.
- Logistic Regression model trained with max iterations set to 200.
- Model evaluated using confusion matrix, precision, recall, and ROC-AUC score.
- Threshold tuning performed by maximizing F1 score for better balance between precision and recall.
- ROC curve and Sigmoid function plotted for visualization.

## Usage Instructions

1. Clone this repository.
2. Place `data.csv` in the project directory.
3. Install dependencies:
4. Run `logistic_regression_classifier.py` or use the Jupyter notebook to reproduce the results and visualizations.

## Results

- Default threshold (0.5) results:
- High precision (~0.975)
- High recall (~0.93)
- ROC-AUC close to perfect (0.996)

- Tuned threshold (~0.315) results:
- Balanced precision and recall (~0.976)
- Improved recall with almost no loss in precision

## Explanation

- The **sigmoid function** is used to convert the linear model output into a probability between 0 and 1:
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]
- ROC curve shows the model's ability to differentiate classes across thresholds. A nearly vertical curve near the top-left corner means excellent performance.

## Visuals

- `roc_curve.png`: ROC curve of the classifier.

<img width="2400" height="1600" alt="roc_curve" src="https://github.com/user-attachments/assets/d01db962-ac25-4848-aeba-e6cfe30b4d80" />

