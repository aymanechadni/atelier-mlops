"""Entrainement KNN sur Iris : DVC orchestre, MLflow trace les experiences."""
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "params.yaml").open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
train = config["train"]
knn = config["knn"]

df = pd.read_csv(ROOT / "data" / "raw" / "iris.csv")
X, y = df.drop(columns=["target"]), df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=train["test_size"], random_state=train["random_state"], stratify=y
)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TP_Iris_Classification")
with mlflow.start_run(run_name="KNN_Iris"):
    model = KNeighborsClassifier(n_neighbors=knn["n_neighbors"], weights=knn["weights"])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }
    mlflow.log_params({"model_type": "KNN", **knn,
                       "test_size": train["test_size"], "random_state": train["random_state"]})
    mlflow.log_metrics(metrics)
    (ROOT / "models").mkdir(exist_ok=True)
    model_file = ROOT / "models" / "model_knn.pkl"
    joblib.dump(model, model_file)
    mlflow.log_artifact(str(model_file), artifact_path="model_knn")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics_knn.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )
print("KNN - Metriques :", metrics)
print("Modele KNN entraine et enregistre.")
