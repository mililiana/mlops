from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "mlops_pipeline",
    default_args=default_args,
    description="A simple ML pipeline DAG using mounted volumes",
    schedule=timedelta(days=1),
    catchup=False,
    tags=["mlops"],
) as dag:

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

    prepare_data = DockerOperator(
        task_id="prepare_data", command="python src/prepare.py", **docker_kwargs
    )

    train_model = DockerOperator(
        task_id="train_model", command="python src/train.py", **docker_kwargs
    )

    evaluate_model = DockerOperator(
        task_id="evaluate_model", command="python compare_metrics.py", **docker_kwargs
    )

    prepare_data >> train_model >> evaluate_model
