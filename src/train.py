
import os
import yaml
import json
import joblib
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

# Charger les paramètres
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)["train"]

# Charger le dataset
df = pd.read_csv("data/raw/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=y
)

# Configurer MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TP_Iris_Classification")

# Démarrer une expérimentation
with mlflow.start_run(run_name="RandomForest_Iris"):

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
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
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # Sauvegarder le modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    mlflow.log_artifact("models/model.pkl", artifact_path="model")

    # Enregistrer les métriques dans un fichier
    os.makedirs("reports", exist_ok=True)

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Métriques :", metrics)
    print("Modèle entraîné et enregistré.")
