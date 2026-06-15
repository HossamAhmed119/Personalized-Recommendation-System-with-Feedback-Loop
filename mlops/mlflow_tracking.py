import mlflow

class MLFlowTracker:
    def __init__(self, experiment_name, db_uri="sqlite:///mlflow.db"):
        """
        Initialize MLflow tracking configuration.
        """
        mlflow.set_tracking_uri(db_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name=None, nested=False):
        """
        Start a new MLflow run.
        """
        return mlflow.start_run(run_name=run_name, nested=nested)

    def log_params(self, params: dict):
        """
        Log hyperparameters to the current run.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        """
        Log evaluation metrics to the current run.
        """
        mlflow.log_metrics(metrics)
        
    def log_model(self, model, artifact_path="model"):
        """
        Save the trained model artifact.
        """
        # Note: Depending on the library (e.g., implicit, cornac), 
        # the logging method might change (e.g., mlflow.sklearn, mlflow.pyfunc)
        mlflow.pyfunc.log_model(artifact_path=artifact_path, python_model=model)