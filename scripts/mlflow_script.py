import mlflow


def register_model(model_dir, uri = "https://91ce-125-16-189-236.ngrok-free.app/", experiment="MT5-FineTuned"):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run() as run:
        # Log model directory as artifact
        mlflow.log_artifacts(model_dir, artifact_path="model")

        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "MT5_LoRA_Model")

    

def log_metrics(metrics, uri="https://91ce-125-16-189-236.ngrok-free.app/", experiment="MT5-FineTuned"):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    
    with mlflow.start_run():
        # Log each metric from the results dict
        for key, value in metrics.items():
            if isinstance(value, (int, float)):  # mlflow only supports numeric values
                mlflow.log_metric(key, value)



if __name__=="__main__":
    pass