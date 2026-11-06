"""Entrainement Random Forest sur Iris avec DVC et MLflow."""
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "params.yaml").open(encoding="utf-8") as handle:
    train = yaml.safe_load(handle)["train"]
df = pd.read_csv(ROOT / "data" / "raw" / "iris.csv")
X, y = df.drop(columns=["target"]), df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=train["test_size"], random_state=train["random_state"], stratify=y
)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TP_Iris_Classification")
with mlflow.start_run(run_name="RandomForest_Iris"):
    model = RandomForestClassifier(
        n_estimators=train["n_estimators"], max_depth=train["max_depth"],
        random_state=train["random_state"]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }
    mlflow.log_params({"model_type": "RandomForest", **train})
    mlflow.log_metrics(metrics)
    (ROOT / "models").mkdir(exist_ok=True)
    model_file = ROOT / "models" / "model.pkl"
    joblib.dump(model, model_file)
    mlflow.log_artifact(str(model_file), artifact_path="model")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )
print("RandomForest - Metriques :", metrics)
print("Modele RandomForest entraine et enregistre.")
