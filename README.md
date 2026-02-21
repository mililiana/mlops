# MLOps Lab 1: Tweet Sentiment Classification

This project was created as part of the MLOps course. The main goal is to develop a pipeline for tweet sentiment classification (positive/negative) using classic machine learning algorithms (TF-IDF + Random Forest) and experiment tracking tools (MLflow).

## Project Structure
- `data/raw/` - Contains the original `train.csv` dataset.
- `notebooks/` - Jupyter notebook (`01_eda.ipynb`) for initial exploratory data analysis.
- `src/train.py` - The main script for training the model, evaluating metrics, and logging results to MLflow.
- `mlruns/` - A local directory created by MLflow to store all artifacts (models, plots).
- `requirements.txt` - List of Python dependencies.

## Environment Setup

1. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(or run `pip install pandas numpy scikit-learn matplotlib seaborn mlflow`)*

## How to Run the Code

All model hyperparameters are exposed as Command Line Interface (CLI) arguments.

To run a **single basic training execution**, use:
```bash
python src/train.py --n_estimators 50 --max_depth 15 --max_features 5000
```
After a successful run, the metrics (Accuracy, F1-Score) will appear in the console, and the model will be saved to MLflow.

### Hyperparameter Tuning

To see how a specific parameter (e.g., `max_depth`) affects Overfitting, you can run the script in a loop with different values.
Example command for Mac/Linux:
```bash
for depth in 5 10 20 50 100; do
    echo "Running max_depth=$depth"
    python src/train.py --max_depth $depth
done
```

## Viewing Results in MLflow (UI)

All metrics for both training and testing datasets, Feature Importance plots, Confusion Matrices, and packaged models are automatically logged to the MLflow system.

To analyze and compare your experiments:

1. Start the MLflow server in your terminal (make sure you are in the project's root directory):
   ```bash
   mlflow ui
   ```
2. Open your browser and navigate to: **http://127.0.0.1:5000**
3. Find the `Tweet_Sentiment_Classification` experiment in the left sidebar.
4. To find specific models, use search filters (for example, `tags.model_type = "RandomForest"`).
5. Select multiple runs and click the **Compare** button to view plots showing how the selected metric (e.g., F1) depends on changes in hyperparameters (Parallel Coordinates Plot or Scatter Plot).