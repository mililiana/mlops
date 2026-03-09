import os
import pytest
from airflow.models import DagBag

DAG_PATH = os.path.join(os.path.dirname(__file__), "..", "dags")


@pytest.fixture
def dagbag():
    return DagBag(dag_folder=DAG_PATH, include_examples=False)


def test_dag_import():
    dag_bag = DagBag(dag_folder="dags/", include_examples=False)
    assert (
        len(dag_bag.import_errors) == 0
    ), f"DAG import errors:\n{dag_bag.import_errors}"


def test_ml_training_pipeline_exists(dagbag):
    dag_id = "ml_training_pipeline"
    assert dag_id in dagbag.dags, f"DAG '{dag_id}' не знайдено серед завантажених."

    dag = dagbag.get_dag(dag_id)
    assert dag is not None
    assert len(dag.tasks) > 0

    task_ids = [task.task_id for task in dag.tasks]
    expected_tasks = [
        "check_data_update",
        "prepare_data",
        "train_model",
        "evaluate_branch",
        "register_model",
        "stop_pipeline",
        "evaluate_model",
    ]
    for expected_task in expected_tasks:
        assert (
            expected_task in task_ids
        ), f"Task '{expected_task}' відсутній у DAG '{dag_id}'."
