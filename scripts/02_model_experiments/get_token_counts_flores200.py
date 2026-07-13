
# %% Setup
# Python standard library
import os

# Third-party libraries
import anthropic
from dotenv import load_dotenv
import google.genai
import pandas as pd
import tiktoken
from transformers import AutoTokenizer


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Models and data to use
hf_model_list = ["Qwen/Qwen3.6-35B-A3B", 
                 "microsoft/phi-4", 
                 "google/gemma-4-31B-it", 
                 "ibm-granite/granite-4.1-30b", 
                 "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
                 "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                 "meta-llama/Llama-3.3-70B-Instruct",
                 "CohereLabs/tiny-aya-base",
                 "openai/gpt-oss-20b",
                 "mistralai/Ministral-3-14B-Instruct-2512",
                 "facebook/nllb-200-3.3B"]

openai_model_list = ["gpt-5"]

anthropic_model_list = ["claude-opus-4-8", 
                        "claude-fable-5"]

google_model_list = ["gemini-2.5-pro",
                     "gemini-3.1-pro-preview",
                     "gemini-3.5-flash"]

# Data: FLORES-200 dev
flores200_dev = pd.read_table("01_data_processed/flores200_dev.tsv")

# %% Loop through each model and get token counts
results = []

# Hugging Face open-source models
for model in hf_model_list:
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

# OpenAI closed-source models
for model in openai_model_list:
    print(f"Working on model: {model}") 
    tokenizer = tiktoken.encoding_for_model(model)

    for row_index, row_data in flores200_dev.iterrows():
        file = row_data["file"]
        text = row_data["text"]
                
        # Tokenize data
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        
        # Append results for this iteration in results list
        result_i = {"model": model, "file": file, "token_count": token_count}
        results.append(result_i)

# Anthropic closed-source models
client = anthropic.Anthropic()
for model in anthropic_model_list:
    print(f"Working on model: {model}") 

    for row_index, row_data in flores200_dev.iterrows():
        file = row_data["file"]
        text = row_data["text"]
                
        # Tokenize data
        response = client.messages.count_tokens(model = model, messages = [{"role": "user", "content": text}])
        token_count = response.input_tokens
        
        # Append results for this iteration in results list
        result_i = {"model": model, "file": file, "token_count": token_count}
        results.append(result_i)

# Google closed-source models
client = google.genai.Client()
for model in google_model_list:
    print(f"Working on model: {model}") 

    for row_index, row_data in flores200_dev.iterrows():
        file = row_data["file"]
        text = row_data["text"]
                
        # Tokenize data
        response = client.models.count_tokens(model = model, contents = text)
        token_count = response.total_tokens
        
        # Append results for this iteration in results list
        result_i = {"model": model, "file": file, "token_count": token_count}
        results.append(result_i)

        
results_df = pd.DataFrame(data = results)

# Write results to CSV file
results_df.to_csv("02_output_model_experiments/token_counts_flores200.csv", index = False)


# TODO: try roundtrip tokenization to make sure the text is tokenized correctly (for deepseek llama issue)
