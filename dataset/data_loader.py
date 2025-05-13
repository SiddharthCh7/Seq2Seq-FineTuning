from datasets import load_dataset
import yaml
import os
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient

# Load env file
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
env_path = os.path.join(project_root, '.env')

if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    print(f".env loaded from: {env_path}")
else:
    print(f".env file not found at: {env_path}")

# Load hugging face token
hf_token = os.getenv("hf_token")

# Hugging Face Authentication
login(token=hf_token)


# Function to fetch samples
def fetch_samples(lang, n_samples):
    
    # storage
    samples = []

    try:
        # fetch data through streaming
        ds = load_dataset("uonlp/CulturaX", lang, streaming=True, split="train")

        count = 0
        for sample in tqdm(ds, desc=f"Sampling '{lang}'"):
            if count >= n_samples:
                count = 0
                break
            samples.append({
                "text" : sample["text"],
            })
            count += 1
    except Exception as e:
        print(f"Error loading {lang}: {str(e)}")
        
    return samples


# Function to load the data
def load_sample_data(config, n_samples, local_dir="bronze"):
    """Load, sample, and stack data for all languages in config."""
    
    # Initialize directory to store the raw data
    os.makedirs(local_dir, exist_ok=True)

    # Define no of samples per batch
    samples_per_batch = 100000

    for i in range(n_samples // samples_per_batch):
        print("this is i:",i)

        # Storage
        all_samples = []

        try:
            # Sequential processing
            for lang, _ in config["languages"].items():
                try:
                    samples = fetch_samples(lang, samples_per_batch)
                except:
                    print(f"Language '{lang}' not present. Skipping it!!!")
                all_samples.extend(samples)
        except Exception as e:
            print("Exception loading sample data: ",str(e))
            return
        
        print(f"No. of samples : {len(all_samples)}")
        if all_samples:
        
            # Convert to arrow table
            table = pa.Table.from_pylist(all_samples)

            # Save as parquet file
            file_name = f"{local_dir}/part_{i}.parquet"
            pq.write_table(table, file_name, compression="snappy")
            print(f"Data Successfully saved to {file_name}.")
        else:
            print("Not enough data was fetched!!!!")
            return None
    return file_name


# Function to push the data from local storage to azure containers
def push_to_azure_container(local_base_dir):

    # Config
    connect_str = os.getenv("azure_connection_string")
    container_name = os.getenv("azure_container_name")

    # Connect to the Blob service
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)

    # === UPLOAD FILES WITH DIRECTORY STRUCTURE ===
    for root, dirs, files in os.walk(local_base_dir):
        for file in files:
            local_file_path = os.path.join(root, file)

            # Create blob path that preserves top-level folder name
            blob_path = os.path.relpath(local_file_path, os.path.dirname(local_base_dir)).replace("\\", "/")

            print(f"Uploading {local_file_path} as {blob_path}...")
            with open(local_file_path, "rb") as data:
                container_client.upload_blob(name=blob_path, data=data, overwrite=True)

    print("Upload complete.")




if __name__ == "__main__":
    pass