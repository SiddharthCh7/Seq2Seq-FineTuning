from azure.storage.blob import BlobServiceClient
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from peft import get_peft_model, TaskType, LoraConfig
import torch
from datasets import Dataset
import pandas as pd
import pyarrow.parquet as pq
import os, io, yaml
from dotenv import load_dotenv
load_dotenv()



# Function to load data from silver layer from azure
def load_data():
    # Your Azure credentials
    account_name = "storageforregular"
    account_key = os.getenv("account_key_azure")
    container_name = "containerfortextdata"

    # Create service client
    blob_service_client = BlobServiceClient(
        f"https://{account_name}.blob.core.windows.net",
        credential=account_key
    )

    # List blobs in the 'silver/' directory
    container_client = blob_service_client.get_container_client(container_name)

    # Reading language info file
    with open("dataset/sampleInfo.yaml") as f:
        config = yaml.safe_load(f)
    langs = [i for i in config['languages']]

    dfs = []

    for lang in langs:
        # Prefix to search under
        prefix = f"silver/{lang}/part_"

        # List all blobs under 'silver/part_*' that are actual parquet files
        blob_list = container_client.list_blobs(name_starts_with=prefix)
        parquet_blobs = [
            blob.name for blob in blob_list
            if blob.name.endswith(".parquet") and "part-" in blob.name
        ]
        # Download and load each file into a DataFrame
        for blob_name in parquet_blobs:
            try:
                print(f"Reading {blob_name}")
                blob_client = container_client.get_blob_client(blob_name)
                stream = io.BytesIO(blob_client.download_blob().readall())
                table = pq.read_table(stream)
                dfs.append(table.to_pandas())
            except Exception as e:
                print(blob_name, "is empty")
                continue

    # Concatenate all into a single DataFrame
    final_df = pd.concat(dfs, ignore_index=True)

    # return stacked dataframe
    return final_df


# Function that trains the model
def train_model(output_directory="mt5-lora"):

    # Define model
    model_name = "google/mt5-small"

    # Intialize model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        target_modules=["q", "v"],
        modules_to_save=["lm_head"]
    )

    # Final model
    lora_model = get_peft_model(base_model, lora_config)

    df = load_data()

    #Convert pandas dataframe to dataset.Dataset
    dataset = Dataset.from_pandas(df)

    # Split the dataset into train-test(90-10)
    dataset = dataset.train_test_split(test_size=0.1)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_directory,
        run_name="r-0",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_steps = 1,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=500,
        learning_rate=2e-4,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        label_names=["labels"]
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer_results = trainer.train()

    return trainer_results



if __name__ == "__main__":
    pass
