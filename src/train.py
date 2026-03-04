import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import mlflow
import mlflow.sklearn
import joblib
import json


def get_data_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(project_root, "data", "prepared", "train.csv")
    return data_path


def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл даних не знайдено за шляхом: {data_path}")
    return pd.read_csv(data_path)


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_pipeline(tfidf_params, rf_params):
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", RandomForestClassifier(**rf_params)),
        ]
    )


def log_confusion_matrix(y_true, y_pred, filename="confusion_metrics.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    mlflow.log_artifact(filename)


def log_feature_importance(pipeline, filename="feature_importance.png"):
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    feature_names = tfidf.get_feature_names_out()
    importances = clf.feature_importances_

    # сортую та беру топ20
    indices = np.argsort(importances)[-20:]

    plt.figure(figsize=(10, 8))
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.barh(range(len(indices)), importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    mlflow.log_artifact(filename)
    if os.path.exists(filename):
        os.remove(filename)


def main():
    parser = argparse.ArgumentParser(
        description="Тренування моделі Random Forest для класифікації сентименту твітів."
    )
    parser.add_argument(
        "--n_estimators", type=int, default=50, help="Кількість дерев у лісі"
    )
    parser.add_argument(
        "--max_depth", type=int, default=15, help="Максимальна глибина дерева"
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=5000,
        help="Максимальна кількість слів для TF-IDF",
    )

    args = parser.parse_args()

    print("1. Завантаження даних...")
    data_path = get_data_path()
    df = load_data(data_path)

    print("2. Виділення ознак...")
    X, y = df["tweet"], df["label"]

    print("3. Розділення на тренувальний та тестовий набори...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("4. Ініціалізація MLflow...")
    experiment_name = "Tweet_Sentiment_Classification"
    mlflow.set_experiment(experiment_name)

    rf_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": 12,
        "class_weight": "balanced",
    }

    tfidf_params = {"max_features": args.max_features, "stop_words": "english"}

    print("5. Тренування та логування в MLflow...")
    pipeline = build_pipeline(tfidf_params, rf_params)

    with mlflow.start_run():
        mlflow.set_tag("developer", "Liliana")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset", "twitter_sentiment_train_0.1")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_train_pred = pipeline.predict(X_train)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        accuracy_train = accuracy_score(y_train, y_train_pred)
        f1_train = f1_score(y_train, y_train_pred, average="weighted")

        print(f"Test Metrics  - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(
            f"Train Metrics - Accuracy: {accuracy_train:.4f}, F1-Score: {f1_train:.4f}"
        )
        print(
            "\nClassification Report (Test):\n", classification_report(y_test, y_pred)
        )

        mlflow.log_params(rf_params)
        mlflow.log_params(tfidf_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy_train", accuracy_train)
        mlflow.log_metric("f1_score_train", f1_train)

        metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "accuracy_train": accuracy_train,
            "f1_score_train": f1_train,
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.sklearn.log_model(pipeline, "random_forest_pipeline")

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/model.pkl")

        log_confusion_matrix(y_test, y_pred)
        log_feature_importance(pipeline)

    print(
        f"Навчання завершено успішно. Результати залоговано в експеримент '{experiment_name}'"
    )


if __name__ == "__main__":
    main()
