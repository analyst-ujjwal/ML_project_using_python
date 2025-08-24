import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.impute import SimpleImputer
from lightgbm import early_stopping
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

# =====================
# 1. CONFIGURATION - CHANGE THESE FOR OTHER FILES
# =====================
TRAIN_FILE = input("ENTER THE PATH OF TRAIN FILE:")   # Change to your train file
TEST_FILE = input("ENTER THE PATH OF TEST FILE:")     # Change to your test file
TARGET_COL = input("ENTER THE TARETED COLUMN:")           # Change to your target column

# =====================
# 2. Load Data
# =====================
train = pd.read_csv(TRAIN_FILE)
test  = pd.read_csv(TEST_FILE)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

# =====================
# 3. Check Target Column
# =====================
if TARGET_COL not in train.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in train file. Available columns: {train.columns.tolist()}")

print(f"Selected target column: {TARGET_COL}")

# =====================
# 4. Explore Target Variable
# =====================
if train[TARGET_COL].nunique() <= 20:
    sns.countplot(x=TARGET_COL, data=train)
    plt.title(f"Target Distribution ({TARGET_COL})")
    # plt.show()
else:
    print(f"Target column '{TARGET_COL}' has more than 20 unique values, skipping plot.")

# =====================
# 5. Preprocessing
# =====================
# Drop ID column if exists
id_col = None
for possible_id in ['id', 'ID', 'Id', 'PassengerId']:
    if possible_id in train.columns:
        id_col = possible_id
        train.drop(possible_id, axis=1, inplace=True)
        if possible_id in test.columns:
            test_id = test[possible_id]
            test.drop(possible_id, axis=1, inplace=True)
        else:
            test_id = pd.Series(np.arange(len(test)), name="id")
        break
if id_col is None:
    test_id = pd.Series(np.arange(len(test)), name="id")

# Drop target column from test if present (should not be, but just in case)
if TARGET_COL in test.columns:
    test.drop(TARGET_COL, axis=1, inplace=True)

# Encode categorical features (excluding target)
cat_cols = [col for col in train.select_dtypes(include=['object']).columns if col != TARGET_COL]
for col in cat_cols:
    le = LabelEncoder()
    all_vals = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(all_vals)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Encode target if it's not numeric
if train[TARGET_COL].dtype == 'object'  or train[TARGET_COL].dtype == 'bool':
    le_target = LabelEncoder()
    train[TARGET_COL] = le_target.fit_transform(train[TARGET_COL].astype(str))

# Ensure target is integer type (important for bool columns)
train[TARGET_COL] = train[TARGET_COL].astype(int)

# Split features and target
X = train.drop(TARGET_COL, axis=1)
y = train[TARGET_COL]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Impute missing values in features
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
test = imputer.transform(test)

# Convert back to DataFrame to preserve feature names
X_train = pd.DataFrame(X_train, columns=X.columns)
X_val = pd.DataFrame(X_val, columns=X.columns)
test = pd.DataFrame(test, columns=X.columns)

# =====================
# 6. Baseline Logistic Regression
# =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_val_scaled)

print("\nLogistic Regression Performance:")
print(classification_report(y_val, y_pred_lr))
if len(np.unique(y_val)) == 2 or len(np.unique(y_val)) == 1:
    print("ROC-AUC:", roc_auc_score(y_val, log_reg.predict_proba(X_val_scaled)[:,1]))

# =====================
# 7. LightGBM Model
# =====================
lgbm = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    random_state=42,
    class_weight='balanced'
)

lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[early_stopping(stopping_rounds=1000)]
)
y_pred_lgb = lgbm.predict(X_val)

print("\nLightGBM Performance:")
print(classification_report(y_val, y_pred_lgb))
if len(np.unique(y_val)) == 2 or len(np.unique(y_val)) == 1:
    print("ROC-AUC:", roc_auc_score(y_val, lgbm.predict_proba(X_val)[:,1]))

# =====================
# 8. Generate Submission
# =====================
final_preds = lgbm.predict(test)
final_preds = np.round(final_preds, 2)  # Ensures two decimals

submission = pd.DataFrame({
    "Passengerid": test_id,
    TARGET_COL: final_preds
})

submission.to_csv("submis.csv", index=False, float_format="%.2f")
print("Submission file created!")