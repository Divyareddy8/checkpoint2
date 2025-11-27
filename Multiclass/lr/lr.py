# ============================================================
# logreg_multiclass.py — Preprocessing + Samplers + Logistic Regression
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ML imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# ============================================================
# Load Data
# ============================================================
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ============================================================
# Preprocessing Setup
# ============================================================
X = train.drop(columns=['participant_id', 'personality_cluster'])
y = train['personality_cluster']

X_test_final = test.drop(columns=['participant_id'])

numerical_cols = ['focus_intensity', 'consistency_score']

nominal_cols = [
    'identity_code', 'cultural_background', 'upbringing_influence', 
    'external_guidance_usage', 'support_environment_score',
    'hobby_engagement_level', 'physical_activity_index',
    'creative_expression_index', 'altruism_score'
]

ordinal_cols = ['age_group']

# Pipelines
numerical_pipeline = Pipeline([('scaler', StandardScaler())])
nominal_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
ordinal_pipeline = Pipeline([('ordinal', OrdinalEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('nom', nominal_pipeline, nominal_cols),
        ('ord', ordinal_pipeline, ordinal_cols),
    ],
    remainder='drop'
)

# Fit-transform
X_processed = preprocessor.fit_transform(X)
X_test_final_processed = preprocessor.transform(X_test_final)

# Encode multi-class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print("Target classes mapping:", dict(zip(label_encoder.classes_, y_encoded)))

# ============================================================
# Sampling Strategies
# ============================================================
samplers = {
    'No_Sampling': None,
    'ROS': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42)
}

best_models = {}
results = []
test_predictions = {}

# ============================================================
# Logistic Regression Model + GridSearchCV
# ============================================================
param_grid = {
    'C': [0.01, 0.1, 1, 8, 10, 100],
    'solver': ['liblinear', 'saga', 'newton-cholesky', 'lbfgs', 'sag', 'newton-cg'],
    'max_iter': [1000]
}

for name, sampler in samplers.items():
    print(f"\n--- Training Logistic Regression with {name} ---")

    # Apply sampler
    if sampler:
        X_res, y_res = sampler.fit_resample(X_processed, y_encoded)
    else:
        X_res, y_res = X_processed, y_encoded

    # Train/Test split
    X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    # GridSearchCV for Logistic Regression
    log_reg = LogisticRegression(random_state=42, multi_class='auto')
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    best_models[name] = best_model

    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    results.append({
        "Sampling Strategy": name,
        "Val_Accuracy": val_acc
    })

    # Test predictions
    test_pred_numeric = best_model.predict(X_test_final_processed)
    test_predictions[name] = test_pred_numeric

# ============================================================
# Summary of all samplers
# ============================================================
summary_df = pd.DataFrame(results).sort_values(by="Val_Accuracy", ascending=False)
print("\n===== SUMMARY OF LOGISTIC REGRESSION RESULTS =====")
print(summary_df)

# ============================================================
# Save predictions for ALL samplers
# ============================================================
for name, preds_numeric in test_predictions.items():
    preds_labels = label_encoder.inverse_transform(preds_numeric)
    submission = pd.DataFrame({
        "participant_id": test["participant_id"],
        "personality_cluster": preds_labels
    })
    filename = f"logreg_multiclass_submission_{name}.csv"
    submission.to_csv(filename, index=False)
    print(f"Saved → {filename}")
    print(submission.head(), "\n")
