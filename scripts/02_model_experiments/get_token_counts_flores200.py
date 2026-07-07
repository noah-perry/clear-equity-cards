
# %% Setup
# Python standard library
import os

# Third-party libraries
from dotenv import load_dotenv
import pandas as pd
from transformers import AutoTokenizer


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Models and data to use
model_list = ["Qwen/Qwen3.6-35B-A3B", 
              "microsoft/phi-4", 
              "google/gemma-4-31B-it", 
              "ibm-granite/granite-4.1-30b", 
              "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
              "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"]

# Data: FLORES-200 dev
flores200_dev = pd.read_table("01_data_processed/flores200_dev.tsv")

# %% Loop through each model and get token counts
results = []
for model in model_list:
    print(f"Working on model: {model}") 
    tokenizer = AutoTokenizer.from_pretrained(model)

    for row_index, row_data in flores200_dev.iterrows():
        file = row_data["file"]
        text = row_data["text"]
                
        # Tokenize data
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        
        # Append results for this iteration in results list
        result_i = {"model": model, "file": file, "token_count": token_count}
        results.append(result_i)
        
results_df = pd.DataFrame(data = results)

# Write results to CSV file
results_df.to_csv("02_output_model_experiments/token_counts_flores200.csv", index = False)


# TODO: try roundtrip tokenization to make sure the text is tokenized correctly (for deepseek llama issue)
# TODO: Add additional models

