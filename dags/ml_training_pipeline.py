import json
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.python import (
    BranchPythonOperator,
    PythonOperator,
)
from airflow.providers.standard.operators.empty import EmptyOperator
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def evaluate_model():
    """Reads the metrics file and returns it so that XCom can pick it up."""
    metrics_path = "/opt/airflow/app/metrics.json"

    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}")
        return {"accuracy": 0.0, "f1_score": 0.0}

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    print(f"Metrics extracted: {metrics}")
    return metrics


def check_accuracy(**kwargs):
    ti = kwargs["ti"]
    # If the task evaluate_model is used to push metrics to xcom:
    metrics = ti.xcom_pull(task_ids="evaluate_model")

    # Fallback to file reading if evaluate_model is skipped
    if not metrics:
        metrics_path = "/opt/airflow/app/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        else:
            return "stop_pipeline"

    accuracy = metrics.get("accuracy", 0)
    print(f"Current Accuracy: {accuracy}")

    if accuracy > 0.85:
        print(f"Accuracy {accuracy} > 0.85. Registering model.")
        return "register_model"
    else:
        print(f"Accuracy {accuracy} <= 0.85. Stopping pipeline.")
        return "stop_pipeline"


with DAG(
    "ml_training_pipeline",
    default_args=default_args,
    description="MLOps Training Pipeline with Sensor, DVC, Branching, and MLflow Registry",
    schedule=timedelta(days=1),
    catchup=False,
    tags=["mlops"],
) as dag:

    # 1. sensor
    check_data_update = FileSensor(
        task_id="check_data_update",
        filepath="/opt/airflow/app/data/raw/train.csv",
        poke_interval=60,
        timeout=600,
        mode="poke",
    )

    docker_kwargs = {
        "image": "mlops_project:latest",
        "api_version": "auto",
        "auto_remove": "force",
        "docker_url": "unix://var/run/docker.sock",
        "network_mode": "bridge",
        "mounts": [
            Mount(
                source="/Users/lilianamirchuk/Desktop/lpnu/8_семестр/mlops/mlops_lab_1",
                target="/app",
                type="bind",
            )
        ],
    }

    # 2. data preparation
    prepare_data = DockerOperator(
        task_id="prepare_data", command="dvc repro prepare", **docker_kwargs
    )

    # 3. model training
    train_model = DockerOperator(
        task_id="train_model", command="dvc repro train", **docker_kwargs
    )

    # Read metrics file and push to XCom (so that evaluate_branch can pull it)

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    # 4. Evaluation and branching
    evaluate_branch = BranchPythonOperator(
        task_id="evaluate_branch",
        python_callable=check_accuracy,
    )

    # 5. model registration
    register_model = DockerOperator(
        task_id="register_model",
        command="python src/register_model.py",
        **docker_kwargs,
    )

    stop_pipeline = EmptyOperator(task_id="stop_pipeline")

    (
        check_data_update
        >> prepare_data
        >> train_model
        >> evaluate_model
        >> evaluate_branch
    )
    evaluate_branch >> [register_model, stop_pipeline]
