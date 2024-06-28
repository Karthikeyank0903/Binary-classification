import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('heart1.csv')

# Split the data
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)

# Evaluate the Models
# Logistic Regression Metrics
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg)
recall_log_reg = recall_score(y_test, y_pred_log_reg)
roc_auc_log_reg = roc_auc_score(y_test, y_pred_log_reg)

# Decision Tree Metrics
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
precision_decision_tree = precision_score(y_test, y_pred_decision_tree)
recall_decision_tree = recall_score(y_test, y_pred_decision_tree)
roc_auc_decision_tree = roc_auc_score(y_test, y_pred_decision_tree)

# Print Metrics
print(f"Logistic Regression - Accuracy: {accuracy_log_reg}, Precision: {precision_log_reg}, Recall: {recall_log_reg}, ROC AUC: {roc_auc_log_reg}")
print(f"Decision Tree - Accuracy: {accuracy_decision_tree}, Precision: {precision_decision_tree}, Recall: {recall_decision_tree}, ROC AUC: {roc_auc_decision_tree}")

# Compute ROC curve and ROC area
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
fpr_decision_tree, tpr_decision_tree, _ = roc_curve(y_test, decision_tree.predict_proba(X_test)[:, 1])

plt.figure()
plt.plot(fpr_log_reg, tpr_log_reg, color='blue', lw=2, label='Logistic Regression (area = %0.2f)' % roc_auc_log_reg)
plt.plot(fpr_decision_tree, tpr_decision_tree, color='green', lw=2, label='Decision Tree (area = %0.2f)' % roc_auc_decision_tree)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Create a DataFrame with metrics
metrics = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [accuracy_log_reg, accuracy_decision_tree],
    'Precision': [precision_log_reg, precision_decision_tree],
    'Recall': [recall_log_reg, recall_decision_tree],
    'ROC AUC': [roc_auc_log_reg, roc_auc_decision_tree]
})

# Save to CSV
metrics.to_csv('model_metrics.csv', index=False)

# Create a DataFrame with ROC data
roc_data = pd.DataFrame({
    'fpr_log_reg': fpr_log_reg,
    'tpr_log_reg': tpr_log_reg,
    'fpr_decision_tree': fpr_decision_tree,
    'tpr_decision_tree': tpr_decision_tree
})

# Save to CSV
roc_data.to_csv('roc_data.csv', index=False)
