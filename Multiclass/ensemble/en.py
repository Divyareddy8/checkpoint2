# ============================================================
# ensemble_models.py — Train multiple ensemble models, save CSV
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Ensemble models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# ============================================================
# Load Data
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

y_val = y.iloc[y_val_enc.index] if hasattr(y_val_enc, 'index') else y.iloc[y_val_enc]

# ============================================================
# Ensemble Models & Grid
# ============================================================
models_and_grids = {
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10]
        }
    },

    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5]
        }
    },

    "AdaBoost": {
        "estimator": AdaBoostClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.5, 1.0, 1.5]
        }
    },

    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 5]
        }
    },

    "XGBoost": {
        "estimator": XGBClassifier(
            objective="multi:softprob",
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ),
        "param_grid": {
            "clf__n_estimators": [200, 400],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 5, 7],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.7, 1.0],
            "clf__reg_alpha": [0, 0.1],
            "clf__reg_lambda": [1, 1.5]
        }
    }
}

# ============================================================
# Function to evaluate & save CSV
# ============================================================
def evaluate_and_save(best_estimator, model_name, X_val, y_val_enc, X_test, test_ids, out_dir="submissions"):
    y_val_pred_enc = best_estimator.predict(X_val)
    y_val_pred = le_y.inverse_transform(y_val_pred_enc)

    print(f"\n--- Validation Results: {model_name} ---")
    print(classification_report(le_y.inverse_transform(y_val_enc), y_val_pred, zero_division=0))

    # Predict test set
    test_preds_enc = best_estimator.predict(X_test)
    test_preds = le_y.inverse_transform(test_preds_enc)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(out_dir) / f"{model_name}.csv"
    pd.DataFrame({
        "participant_id": test_ids,
        "personality_cluster": test_preds
    }).to_csv(fname, index=False)
    print(f"Saved → {fname}")

# ============================================================
# Train & Evaluate All Models
# ============================================================
for model_name, model_info in models_and_grids.items():
    print(f"\n===== Training {model_name} =====")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", model_info["estimator"])
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=model_info["param_grid"],
        scoring="accuracy",
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid.fit(X_train, y_train_enc)

    print(f"Best params for {model_name}: {grid.best_params_}")
    best_model = grid.best_estimator_

    evaluate_and_save(
        best_estimator=best_model,
        model_name=model_name,
        X_val=X_val,
        y_val_enc=y_val_enc,
        X_test=X_test_final,
        test_ids=test_ids
    )
