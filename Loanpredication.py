import pandas as pd
import numpy as np
import os
import re
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix

try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("LightGBM not installed. Run: pip install lightgbm")

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError("imbalanced-learn not installed. Run: pip install imbalanced-learn")

# Generate Sample Data
if not os.path.exists("loan.csv"):
    print("loan.csv not found. Generating sample data...")
    np.random.seed(42)

    num_rows = 500
    loan_data = {
        'loan_amnt': np.random.randint(5000, 40000, num_rows),
        'int_rate': np.round(np.random.uniform(6.0, 25.0, num_rows), 2),
        'annual_inc': np.random.randint(30000, 150000, num_rows),
        'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'other'], num_rows),
        'addr_state': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL', 'PA'], num_rows),
        'dti': np.round(np.random.uniform(0, 35, num_rows), 2),
        'loan_status': np.random.choice(['Fully Paid', 'Charged Off'], num_rows, p=[0.7, 0.3]),
        'emp_length': np.random.choice(['< 1 year', '1 year', '2 years', '3 years', '5 years',
                                        '7 years', '10+ years', np.nan], num_rows)
    }

    pd.DataFrame(loan_data).to_csv('loan.csv', index=False)
    print("Sample loan.csv generated.\n")

# Load and Preprocess
df = pd.read_csv('loan.csv')
df = df[['loan_amnt', 'int_rate', 'annual_inc', 'purpose', 'addr_state', 'dti', 'loan_status', 'emp_length']]
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['target'] = (df['loan_status'] == 'Charged Off').astype(int)
df.drop('loan_status', axis=1, inplace=True)

# Clean emp_length
def parse_emp_length(val):
    if pd.isna(val):
        return np.nan
    if '10+' in str(val): return 10.0
    if '< 1' in str(val): return 0.5
    match = re.search(r'(\d+)', str(val))
    return float(match.group(1)) if match else np.nan

df['emp_length'] = df['emp_length'].apply(parse_emp_length)

# Encode categorical variables
for col in ['purpose', 'addr_state']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = df.drop('target', axis=1)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = df['target']

# Train/Test Split + SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Train LightGBM
model = LGBMClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Confusion Matrix Plot
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance Plot
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("Feature Importance - Loan Default Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
