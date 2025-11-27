# ============================================================
# nn_multiclass_tuner.py — Preprocessing + Samplers + Bayesian-optimized NN
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ML & NN imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import BayesianOptimization

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
# Train/Validation Split
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# One-hot encode labels for NN
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)

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
# Neural Network Builder for KerasTuner
# ============================================================
def build_model(hp, input_dim, num_classes):
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            units=hp.Int('units_1', min_value=32, max_value=512, step=32),
            activation='relu',
            input_shape=(input_dim,)
        )
    )
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(
            keras.layers.Dense(
                units=hp.Int(f'units_{i+2}', min_value=32, max_value=512, step=32),
                activation='relu'
            )
        )
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================================
# Training Loop with Bayesian Optimization + Samplers
# ============================================================
input_dim = X_train.shape[1]
max_trials = 10
epochs = 20
batch_size = 32

for name, sampler in samplers.items():
    print(f"\n--- Training with {name} ---")

    if sampler:
        X_res, y_res = sampler.fit_resample(X_train, y_train_cat)
    else:
        X_res, y_res = X_train, y_train_cat

    # Initialize Bayesian Optimization tuner
    tuner = BayesianOptimization(
        hypermodel=lambda hp: build_model(hp, input_dim, num_classes),
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=2,
        directory='tuner_dir',
        project_name=f'nn_{name}'
    )

    # Search for best hyperparameters
    tuner.search(
        X_res, y_res,
        epochs=epochs,
        validation_data=(X_val, y_val_cat),
        batch_size=batch_size,
        verbose=1
    )

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"\nBest Hyperparameters for {name}:")
    print(f"Number of layers: {best_hps.get('num_layers')}")
    print(f"Learning rate: {best_hps.get('learning_rate')}")
    for i in range(best_hps.get('num_layers') + 1):
        print(f"Layer {i+1} Units: {best_hps.get(f'units_{i+1}')}")

    # Build and train final model with best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    model.fit(
        X_res, y_res,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Validation predictions
    val_pred_probs = model.predict(X_val)
    val_pred = np.argmax(val_pred_probs, axis=1)
    val_acc = accuracy_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val_cat, val_pred_probs, multi_class="ovr")

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation ROC-AUC: {val_auc:.4f}")

    results.append({
        "Sampling Strategy": name,
        "Val_Accuracy": val_acc,
        "Val_ROC_AUC": val_auc
    })

    # Store test predictions
    test_pred_probs = model.predict(X_test_final_processed)
    test_predictions[name] = np.argmax(test_pred_probs, axis=1)

    best_models[name] = model

# ============================================================
# Summary
# ============================================================
summary_df = pd.DataFrame(results).sort_values(by="Val_ROC_AUC", ascending=False)
print("\n===== SUMMARY OF RESULTS =====")
print(summary_df)

best_strategy = summary_df.iloc[0]["Sampling Strategy"]
final_model = best_models[best_strategy]

print(f"\nBest Model Selected → {best_strategy}")

# ============================================================
# Save predictions for ALL samplers
# ============================================================
for name, preds_numeric in test_predictions.items():
    preds_labels = label_encoder.inverse_transform(preds_numeric)
    submission = pd.DataFrame({
        "participant_id": test["participant_id"],
        "personality_cluster": preds_labels
    })
    filename = f"nn_multiclass_submission_{name}.csv"
    submission.to_csv(filename, index=False)
    print(f"Saved → {filename}")
    print(submission.head(), "\n")
