import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

TARGET = "retention_status"
IDCOL = "founder_id"

# Remove duplicates
train = train.drop_duplicates().reset_index(drop=True)

# Log columns
log_cols = ['monthly_revenue_generated', 'funding_rounds_led', 'num_dependents', 'years_with_startup']

def feature_engineer(df):
    df = df.copy()
    if 'years_with_startup' in df and 'years_since_founding' in df:
        df['experience_ratio'] = df['years_with_startup'] / (df['years_since_founding'] + 1e-9)
    if 'founder_age' in df and 'years_with_startup' in df:
        df['founder_join_age'] = df['founder_age'] - df['years_with_startup']
    if 'monthly_revenue_generated' in df and 'funding_rounds_led' in df:
        df['revenue_per_round'] = df['monthly_revenue_generated'] / (df['funding_rounds_led'] + 1)
    for c in log_cols:
        if c in df:
            df[f"log_{c}"] = np.log1p(df[c])
            df.drop(c, axis=1, inplace=True)
    return df

train_fe = feature_engineer(train)
test_fe = feature_engineer(test)

if IDCOL not in test_fe.columns and IDCOL in test.columns:
    test_fe[IDCOL] = test[IDCOL]

X = train_fe.drop(columns=[TARGET, IDCOL])
y = train_fe[TARGET].map({"Stayed": 1, "Left": 0}).astype(int)
X_test = test_fe.drop(columns=[IDCOL], errors="ignore")

# Feature columns
numerical_cols = [c for c in [
    'years_since_founding', 'founder_age', 'distance_from_investor_hub',
    'experience_ratio', 'founder_join_age', 'revenue_per_round',
    'log_monthly_revenue_generated', 'log_funding_rounds_led',
    'log_num_dependents', 'log_years_with_startup'
] if c in X.columns]

binary_cols = [c for c in [
    'working_overtime', 'remote_operations',
    'leadership_scope', 'innovation_support'
] if c in X.columns]

ordinal_cols = {
    'work_life_balance_rating': ['Poor', 'Fair', 'Good', 'Excellent'],
    'venture_satisfaction': ['Low', 'Medium', 'High', 'Very High'],
    'startup_performance_rating': ['Low', 'Below Average', 'Average', 'High'],
    'startup_reputation': ['Poor', 'Fair', 'Good', 'Excellent'],
    'founder_visibility': ['Low', 'Medium', 'High', 'Very High'],
    'startup_stage': ['Entry', 'Mid', 'Senior'],
    'team_size_category': ['Small', 'Medium', 'Large']
}
ordinal_feature_names = [c for c in ordinal_cols.keys() if c in X.columns]
ordinal_categories = [ordinal_cols[c] for c in ordinal_feature_names]

nominal_cols = [c for c in ['founder_gender', 'founder_role', 'education_background', 'personal_status'] if c in X.columns]

# Pipelines
transformers = []

if numerical_cols:
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    transformers.append(('num', num_pipe, numerical_cols))

if binary_cols:
    bin_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[['No','Yes']]*len(binary_cols)))
    ])
    transformers.append(('bin', bin_pipe, binary_cols))

if ordinal_feature_names:
    ord_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ordinal_categories))
    ])
    transformers.append(('ord', ord_pipe, ordinal_feature_names))

if nominal_cols:
    nom_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    transformers.append(('nom', nom_pipe, nominal_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

X = pd.DataFrame(preprocessor.fit_transform(X), columns=preprocessor.get_feature_names_out())
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())

print("Processed X shape:", X.shape, "X_test shape:", X_test.shape)

# Add clustering
K = 12
kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, random_state=42)
kmeans.fit(X)
X["cluster_label"] = kmeans.labels_
X_test["cluster_label"] = kmeans.predict(X_test)

print("Cluster labels added!")

# -------------------------------
# 20% Split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# -------------------------------
# Neural Network
# -------------------------------
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_s.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[keras.metrics.AUC(name="AUC")]
)

history = model.fit(
    X_train_s,
    y_train,
    validation_data=(X_val_s, y_val),
    epochs=40,
    batch_size=64,
    verbose=1
)

# -------------------------------
# Validation AUC
# -------------------------------
val_prob = model.predict(X_val_s).flatten()
auc = roc_auc_score(y_val, val_prob)
print("Validation AUC:", auc)

# -------------------------------
# Predict on test and create submission
# -------------------------------
test_prob = model.predict(X_test_s).flatten()
test_pred = np.where(test_prob >= 0.5, "Stayed", "Left")

submission = pd.DataFrame({
    IDCOL: test[IDCOL] if IDCOL in test.columns else np.arange(len(test)),
    TARGET: test_pred
})

submission.to_csv("submission_nn_20.csv", index=False)
print("Submission saved!")
