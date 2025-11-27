# ============================================================
# ensemble_stacking_voting.py — Train Stacking & Voting with preprocessing
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Sklearn imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Ensemble models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# ============================================================
# Load data
# ============================================================
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

X = train.drop(columns=['participant_id', 'personality_cluster'])
y = train['personality_cluster']
X_test_final = test.drop(columns=['participant_id'])
test_ids = test['participant_id']

# ============================================================
# Preprocessing
# ============================================================
numerical_cols = ['focus_intensity', 'consistency_score']
nominal_cols = [
    'identity_code', 'cultural_background', 'upbringing_influence',
    'external_guidance_usage', 'support_environment_score',
    'hobby_engagement_level', 'physical_activity_index',
    'creative_expression_index', 'altruism_score'
]
ordinal_cols = ['age_group']

numerical_pipeline = Pipeline([('scaler', StandardScaler())])
nominal_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
ordinal_pipeline = Pipeline([('ordinal', OrdinalEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('nom', nominal_pipeline, nominal_cols),
        ('ord', ordinal_pipeline, ordinal_cols)
    ],
    remainder='drop'
)

# Encode target
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# Train/Validation split
X_train, X_val, y_train_enc, y_val_enc = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ============================================================
# Base estimators for stacking / voting
# ============================================================
rfc = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
gbc = GradientBoostingClassifier(n_estimators=200, random_state=42)
xgbc = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="mlogloss", random_state=42, n_jobs=-1)

estimators = [
    ('rf', rfc),
    ('gbc', gbc),
    ('xgb', xgbc)
]

# ============================================================
# Preprocessing + Model Pipelines
# ============================================================
# Transform features first
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test_final)

# ============================================================
# 1️⃣ Stacking Classifier
# ============================================================
stack_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3,
    n_jobs=-1
)

stack_clf.fit(X_train_processed, y_train_enc)
y_val_pred_stack = stack_clf.predict(X_val_processed)

print("\n=== Stacking Classifier Validation ===")
print(classification_report(le_y.inverse_transform(y_val_enc), le_y.inverse_transform(y_val_pred_stack)))
print("Validation Accuracy:", accuracy_score(y_val_enc, y_val_pred_stack))
print("Validation F1 (macro):", f1_score(y_val_enc, y_val_pred_stack, average='macro'))

# Predict on test
test_preds_stack = stack_clf.predict(X_test_processed)
test_preds_stack_labels = le_y.inverse_transform(test_preds_stack)

submission_stack = pd.DataFrame({
    "participant_id": test_ids,
    "personality_cluster": test_preds_stack_labels
})
submission_stack.to_csv("stacking_submission.csv", index=False)
print("\nSaved → stacking_submission.csv")

# ============================================================
# 2️⃣ Voting Classifier
# ============================================================
voting_clf = VotingClassifier(
    estimators=estimators,
    voting='soft',  # soft voting usually better for multi-class
    n_jobs=-1
)
voting_clf.fit(X_train_processed, y_train_enc)
y_val_pred_vote = voting_clf.predict(X_val_processed)

print("\n=== Voting Classifier Validation ===")
print(classification_report(le_y.inverse_transform(y_val_enc), le_y.inverse_transform(y_val_pred_vote)))
print("Validation Accuracy:", accuracy_score(y_val_enc, y_val_pred_vote))
print("Validation F1 (macro):", f1_score(y_val_enc, y_val_pred_vote, average='macro'))

# Predict on test
test_preds_vote = voting_clf.predict(X_test_processed)
test_preds_vote_labels = le_y.inverse_transform(test_preds_vote)

submission_vote = pd.DataFrame({
    "participant_id": test_ids,
    "personality_cluster": test_preds_vote_labels
})
submission_vote.to_csv("voting_submission.csv", index=False)
print("\nSaved → voting_submission.csv")
