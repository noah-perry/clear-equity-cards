
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


# %% Lists of tokenizers to use
hf_models = pd.read_table("01_data_processed/hf_models_by_tokenizer.tsv")
hf_model_list = hf_models["tokenizer_repo_id"].tolist()

hf_remote_code_list = ["moonshotai/Kimi-Linear-48B-A3B-Instruct"]
    # requires `trust_remote_code = True` to run

openai_model_list = ["gpt-5"]

anthropic_model_list = ["claude-fable-5"]
    # "claude-opus-4-8" appears to have same tokenizer

google_model_list = ["gemini-3.5-flash"]
    # gemini-2.5-pro and gemini-3.1-pro-preview appear to have same tokenizer


# %% Read FLORES-200 dev data
flores200_dev = pd.read_table("01_data_processed/flores200_dev.tsv")

# %% Loop through each model and get token counts
results = []

# Hugging Face open-source models
for model in hf_model_list:
    print(f"Working on model: {model}")
    if model in hf_remote_code_list:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code = True)
    else:
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
