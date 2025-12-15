import os

def get_tracking_uri():
    """
    Returns the MLflow tracking URI.
    Defaults to a local 'mlruns' directory in work_dirs if not specified.
    """
    default_uri = os.path.join(os.getcwd(), "work_dirs", "mlruns")
    return os.getenv("MLFLOW_TRACKING_URI", f"file://{default_uri}")

def setup_tracking():
    """
    Sets up the MLflow tracking URI.
    """
    import mlflow
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri
