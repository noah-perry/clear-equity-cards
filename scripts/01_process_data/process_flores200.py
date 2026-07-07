
# %% Setup
# Python standard library
import os

# Third-party libraries
from dotenv import load_dotenv
import pandas as pd


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Combine all FLORES dev files into a DataFrame
os.chdir(os.path.join(PROJECT_ROOT, "00_data_raw/flores200_dataset/dev"))
file_list = os.listdir() 
    # each file contains data for a different language-script combination

langdata = []
for file in file_list:
    with open(file, "r", encoding = "utf-8") as f:
        text = f.read()
        langdata_i = {"file": file, "text": text}
        langdata.append(langdata_i)

flores200_dev = pd.DataFrame(data = langdata)

assert all(flores200_dev["text"].str.contains("\t") == False)
    # confirming no tab characters in flores200

os.chdir(PROJECT_ROOT)
flores200_dev.to_csv("01_data_processed/flores200_dev.tsv", sep = "\t", index = False)
