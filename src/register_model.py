import mlflow
import os
from mlflow.tracking import MlflowClient


def register_latest_model():
    experiment_name = "Tweet_Sentiment_Classification"
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment {experiment_name} not found.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        print("No runs found in experiment.")
        return

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    f1_score = latest_run.data.metrics.get("f1_score", 0)

    print(f"Latest run ID: {run_id}, F1-Score: {f1_score}")

    model_name = "SentimentClassifier"
    model_uri = f"runs:/{run_id}/random_forest_pipeline"

    print(f"Registering model {model_name}...")
    result = mlflow.register_model(model_uri, model_name)

    print("Transitioning model to Staging...")
    client.transition_model_version_stage(
        name=model_name, version=result.version, stage="Staging"
    )

    print(
        f"Model {model_name} version {result.version} registered and transitioned to Staging."
    )


if __name__ == "__main__":
    register_latest_model()
