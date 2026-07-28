
# %% Setup
# Python standard library
import os
import hashlib

# Third-party libraries
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import pandas as pd


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Read chosen models table
chosen_models = pd.read_table("00_data_raw/clear_benchmarking_team/LLM CLEAR Sheet - Chosen Models.tsv")

# Rename columns and drop unneeded columns
chosen_models = chosen_models.rename(columns = {"Link": "link", 
                                                "Release Date / Last Update": "release_date_str"})

# Extract model from Hugging Face link
chosen_models["model"] = chosen_models["link"].str.replace("https://huggingface.co/", "")
chosen_models[["organization", "model_name"]] = chosen_models["model"].str.split("/" , expand = True)

# Create sortable release_date column
chosen_models["release_date"] = pd.to_datetime(chosen_models["release_date_str"], format = "mixed")

# Drop unneeded columns
chosen_models = chosen_models[["model", "organization", "model_name", "release_date"]]

# Make tokenizer_file column
no_tokenizer_list = ["apple/OpenELM-1_1B-Instruct", "apple/OpenELM-3B-Instruct"]
tiktoken_list = ["moonshotai/Kimi-Linear-48B-A3B-Instruct", "moonshotai/Moonlight-16B-A3B-Instruct"]

chosen_models["tokenizer_file"] = "tokenizer.json"
chosen_models.loc[chosen_models["model"].isin(no_tokenizer_list), "tokenizer_file"] = None
chosen_models.loc[chosen_models["model"].isin(tiktoken_list), "tokenizer_file"] = "tiktoken.model"
    # based on Hugging Face model repos


# %% Hash each model's tokenizer file
results = []
for row_index, row_data in chosen_models.iterrows():
    model = row_data["model"]
    tokenizer_file = row_data["tokenizer_file"]

    if pd.isna(tokenizer_file):
        result_i = {"model": model, "tokenizer_hash": None}
    else:
        path = hf_hub_download(model, filename = tokenizer_file)
        tokenizer_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()
        result_i = {"model": model, "tokenizer_hash": tokenizer_hash}
    
    results.append(result_i)

    tokenizer_hashes = pd.DataFrame(results)


# %% Merge to add tokenizer_hashes to chosen_models DataFrame
assert tokenizer_hashes["model"].is_unique
assert chosen_models["model"].is_unique

chosen_models = pd.merge(left = chosen_models, right = tokenizer_hashes,
                         how = "outer", on = "model", indicator = True)

assert all(chosen_models["_merge"] == "both")
chosen_models = chosen_models.drop(columns = "_merge")


# %% Summary checks
tokenizer_summary = chosen_models.groupby("tokenizer_hash").agg(n_orgs = ("organization", "nunique"),
                                                                n_models = ("model_name", "count"),
                                                                n_dates = ("release_date", "nunique"),
                                                                models = ("model_name", list))
tokenizer_summary = tokenizer_summary.reset_index(drop = False)
assert all(tokenizer_summary["n_orgs"] == 1)
    # tokenizers are never shared by models from different organizations
assert any(tokenizer_summary["n_dates"] > 1)
    # there exist groups of models that have the same tokenizer but different release dates


# %% Only keep one model when multiple models use the same tokenizer
chosen_models = chosen_models.sort_values(by = ["tokenizer_hash", "release_date", "model"], ascending = True, ignore_index = True)
chosen_models["dup"] = chosen_models["tokenizer_hash"].duplicated(keep = "last")
    # For each tokenizer, keep model with latest release date. Break ties using model (keep last in sort order)

include_models = chosen_models.loc[chosen_models["dup"] == False,]
    # Models to include in token cost experiments

exclude_models = chosen_models.loc[chosen_models["dup"] == True,].groupby("tokenizer_hash").agg(models_sharing_tokenizer = ("model_name", list))
exclude_models = exclude_models.reset_index(drop = False)
    # Models in chosen list that will be excluded from token cost experiment since they have the same tokenizer as another model

model_groups = pd.merge(left = include_models, right = exclude_models, how = "outer", on = "tokenizer_hash")
    # each row per group of models with the same tokenizer

model_groups = model_groups[["model", "models_sharing_tokenizer"]]
model_groups = model_groups.sort_values(by = "model", ignore_index = True)


# %% Write to TSV file
assert model_groups["model"].str.contains("\t").sum() == 0
assert model_groups["models_sharing_tokenizer"].str.contains("\t").sum() == 0

model_groups.to_csv("01_data_processed/benchmarking_team_models_by_tokenizer.tsv", sep = "\t", index = False)
