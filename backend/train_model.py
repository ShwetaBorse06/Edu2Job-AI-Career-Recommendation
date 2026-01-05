# ==========================================
# EDU2JOB – FINAL DUAL MODEL PIPELINE
# Random Forest + XGBoost
# Metric: Top-5 Accuracy
# ==========================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.sparse import hstack

# ----------------------------------
# DIRECTORIES
# ----------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("encoders", exist_ok=True)

# ----------------------------------
# LOAD & CLEAN DATA
# ----------------------------------
df = pd.read_csv("datasets/career_dataset_large.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")
df = df.rename(columns={"cgpa/percentage": "cgpa"})

df = df[
    ["education_level", "specialization", "skills",
     "certifications", "cgpa", "recommended_job"]
].dropna()

# ----------------------------------
# FEATURE ENGINEERING
# ----------------------------------
df["cert_count"] = df["certifications"].astype(str).apply(lambda x: len(x.split(",")))
df["skills"] = df["skills"].astype(str)

# ----------------------------------
# ENCODING
# ----------------------------------
le_edu = LabelEncoder()
le_spec = LabelEncoder()
le_job = LabelEncoder()

df["education_level"] = le_edu.fit_transform(df["education_level"])
df["specialization"] = le_spec.fit_transform(df["specialization"])
y = le_job.fit_transform(df["recommended_job"])

# ----------------------------------
# TF-IDF
# ----------------------------------
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2))
skills_tfidf = tfidf.fit_transform(df["skills"])

# ----------------------------------
# NUMERIC FEATURES
# ----------------------------------
num_features = df[["cgpa", "cert_count", "education_level", "specialization"]]
scaler = MinMaxScaler()
num_scaled = scaler.fit_transform(num_features)

X = hstack([skills_tfidf, num_scaled])

# ----------------------------------
# SPLIT
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ==================================
# RANDOM FOREST
# ==================================
rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ==================================
# XGBOOST
# ==================================
xgb = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",
    random_state=42
)
xgb.fit(X_train, y_train)

# ----------------------------------
# TOP-5 ACCURACY FUNCTION
# ----------------------------------
def top5_accuracy(model, X, y):
    probs = model.predict_proba(X)
    top5 = np.argsort(probs, axis=1)[:, -5:]
    return np.mean([y[i] in top5[i] for i in range(len(y))])

# ----------------------------------
# RESULTS
# ----------------------------------
print("\n🌲 RANDOM FOREST")
print("Top-1 Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
print("Top-5 Accuracy:", top5_accuracy(rf, X_test, y_test))

print("\n🚀 XGBOOST")
print("Top-1 Accuracy:", accuracy_score(y_test, xgb.predict(X_test)))
print("Top-5 Accuracy:", top5_accuracy(xgb, X_test, y_test))

# ----------------------------------
# SAVE
# ----------------------------------
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(xgb, "models/xgboost.pkl")
joblib.dump(tfidf, "encoders/tfidf.pkl")
joblib.dump(scaler, "encoders/scaler.pkl")
joblib.dump(le_edu, "encoders/education_encoder.pkl")
joblib.dump(le_spec, "encoders/specialization_encoder.pkl")
joblib.dump(le_job, "encoders/job_encoder.pkl")

print("\n✅ BOTH MODELS TRAINED & SAVED")
