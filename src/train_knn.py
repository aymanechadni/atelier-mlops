import os
import yaml
import json
import joblib
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

# Charger les paramètres
with open("params.yaml", "r") as file:
    config = yaml.safe_load(file)
    train_params = config.get("train", {})
    knn_params = config.get("knn", {"n_neighbors": 5, "weights": "uniform"})

# Charger le dataset
df = pd.read_csv("data/raw/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=train_params.get("test_size", 0.2),
    random_state=train_params.get("random_state", 42),
    stratify=y
)

# Configurer MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TP_Iris_Classification")

# Démarrer une expérimentation pour KNN
with mlflow.start_run(run_name="KNN_Iris"):
    model = KNeighborsClassifier(
        n_neighbors=knn_params.get("n_neighbors", 5),
        weights=knn_params.get("weights", "uniform")
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculer les métriques
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, average="weighted"
        ),
        "recall": recall_score(
            y_test, predictions, average="weighted"
        ),
        "f1_score": f1_score(
            y_test, predictions, average="weighted"
        )
    }

    # Enregistrer paramètres et métriques
    logged_params = {
        "model_type": "KNN",
        "n_neighbors": knn_params.get("n_neighbors", 5),
        "weights": knn_params.get("weights", "uniform"),
        "random_state": train_params.get("random_state", 42),
        "test_size": train_params.get("test_size", 0.2)
    }
    mlflow.log_params(logged_params)
    mlflow.log_metrics(metrics)

    # Sauvegarder le modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_knn.pkl")
    mlflow.log_artifact("models/model_knn.pkl", artifact_path="model_knn")

    # Enregistrer les métriques dans un fichier
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics_knn.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("KNN Métriques :", metrics)
    print("Modèle KNN entraîné et enregistré.")
