from transformers import AutoTokenizer
import yaml
import gc, os
import time
from dotenv import load_dotenv
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType

spark = SparkSession.builder.appName("MT5-Lora").getOrCreate()
print("Spark Session created!!.............")
load_dotenv()

# Reading language info file
with open("dataset/sampleInfo.yaml") as f:
    config = yaml.safe_load(f)


# Define schema for tokenized output
schema = StructType([
    StructField("input_ids", ArrayType(IntegerType()), True),
    StructField("attention_mask", ArrayType(IntegerType()), True),
    StructField("labels", ArrayType(IntegerType()), True),
])

# storing it in a list
langs = [i for i in config['languages']]

#model name
model_name = "google/mt5-small"
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True) # use_fast will be False when running on databricks

# Replace pad_token_id with -100 so that loss ignores it
def replace_pad_with_neg100(labels, pad_token_id):
    return [token if token != pad_token_id else -100 for token in labels]


# Define tokenizer function using closure
def tokenize_text(text, input_type=None):

  # Ensure input_type is mentioned
  if not input_type:
    print("Input type not mentioned")
    return None

  # Tokenizer
  global tokenizer

  # Process for inputs (input_ids, attention_mask)
  if input_type == "inputs":
    if text is None:
      print("Text is None for input_type == inputs")
      return None, None

    try:
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128
        )
        print("In tokenize_text, 'inputs'")
        print(tokens["input_ids"])
        return tokens["input_ids"], tokens["attention_mask"]
    except Exception as e:
        print(f"Exception in tokenization: {str(e)}, text: {text[:50]}...")
        return None, None

  # Process for labels (only input_ids)
  elif input_type == "labels":
    if text is None:
      print("Text is None for input_type == labels")
      return None

    try:
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128
        )

        labels = replace_pad_with_neg100(tokens["input_ids"], tokenizer.pad_token_id)
        print("In tokenize_text, 'labels'")
        print(labels)
        return labels

    except Exception as e:
        print(f"Exception in tokenization: {str(e)}, text: {text[:50]}...")
        return None


# Split the text (50-50) and tokenize text
def tokenize_partition(partition):
  print("Entered tokenize_partition")
  results = []
  for row in partition:
    try:
      text = row['text'] if isinstance(row, dict) else row.text
      print("Printing text in tokenize_partition:")
      print(text)
      if not text or len(text) < 2:
        continue

      n = len(text) // 2
      input_text, label_text = text[:n], text[n:]

      input_ids, attention_mask = tokenize_text(input_text, input_type="inputs")
      labels = tokenize_text(label_text, input_type="labels")

      results.append(Row(input_ids=input_ids, attention_mask=attention_mask, labels=labels))

    except Exception as e:
      print("Exception in preprocess_partition:", str(e))
      results.append(Row(input_ids=[0], attention_mask=None, labels=None))
  return results


def process():
    # Configure Azure with spark
    spark.conf.set(
        f"fs.azure.account.key.{os.getenv("azure_storage_account")}.dfs.core.windows.net",
        os.getenv("account_key_azure")
    )
    # Process each languages
    for lang in langs:
        # Process all partitions (10 partitions 0-9, each of 100k rows)
        for i in range(10):
            try:
                df = spark.read.parquet(f"abfss://{os.getenv("azure_container_name")}@{os.getenv("azure_storage_account")}.dfs.core.windows.net/bronze/{lang}/part_{i}.parquet")
                # Ensure 'text' column exists
                if 'text' not in df.columns:
                    print(f"Error: 'text' column not found in dataset for language {lang}. Available columns: {df.columns}")

                # Apply tokenizer to each partition
                tokenized_rdd = df.rdd.mapPartitions(tokenize_partition)
                print("Data tokenized")

                # Convert RDD back to DataFrame
                tokenized_df = spark.createDataFrame(tokenized_rdd, schema=schema)
                print("dataframe created")

                # Repartition DataFrame into 10 partitions
                num_partitions = 10
                tokenized_df = tokenized_df.repartition(num_partitions)

                # Create output directory (in azure) if it doesn't exist
                tokenized_df.write.mode("overwrite").parquet(f"abfss://{os.getenv("azure_container_name")}@{os.getenv("azure_storage_account")}.dfs.core.windows.net/silver/{lang}/part_{i}.parquet")
                print("Data stored in azure")

                print(f"Successfully wrote tokenized data for language: {lang} for {i} partitions")

                # Force garbage collection
                gc.collect()
                time.sleep(1)  # Give time for memory to be freed
    
            except Exception as e:
                print(f"Error processing language {lang}: {str(e)}")
                import traceback
                traceback.print_exc()    

    # Force garbage collection (nothing wrong in trying to be more efficient:))
    gc.collect()
    time.sleep(1)

    spark.stop()


if __name__ == "__main__":
   pass