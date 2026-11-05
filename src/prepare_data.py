
from sklearn.datasets import load_iris
import pandas as pd
import os

# Charger le dataset Iris
iris = load_iris()

# Transformer les données en DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Créer le dossier de stockage
os.makedirs("data/raw", exist_ok=True)

# Enregistrer les données au format CSV
df.to_csv("data/raw/iris.csv", index=False)

print("Dataset Iris généré avec succès")
print("Nombre de lignes :", len(df))
print("Nombre de colonnes :", len(df.columns))
print(df.head())

# Afficher la distribution des classes
print("Distribution des classes :")
print(df["target"].value_counts().sort_index())
