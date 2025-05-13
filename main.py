if __name__ == "__main__":

    from dataset import load_sample_data, push_to_azure_container
    import yaml
    # Import functions from .py in 'scripts' directory, exposed via __init__.py file in the current directory
    from . import process
    from . import train_model
    from . import log_metrics, register_model

     # Open sampleInfo.yaml file for language and n_samples info
    with open("dataset/sampleInfo.yaml") as f:
        config = yaml.safe_load(f)

    # Mention local dir where data will be downloaded (then will be pushed to azure)
    local_dir = "bronze"

    # Download and load the data
    samples_output_path = load_sample_data(config=config, n_samples=1000000, local_dir=local_dir)

    # Push bronze direc to cloud(azure)
    push_to_azure_container(local_dir)

    # Tokenize the raw data which is present in the 'bronze' layer in azure data lake
    # and then store the processed data in 'silver' layer of the same data lake
    process()

    # Set output directory where the trained model gets saved
    output_dir = "mt5-lora"

    # Train the model and get the results
    results = train_model(output_directory="mt5-lora")

    # Register the model in mlflow
    register_model(model_dir=output_dir)

    # Log the metrics in mlflow
    log_metrics(metrics=results)