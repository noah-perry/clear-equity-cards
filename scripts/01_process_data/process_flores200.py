
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

flores200_dev = flores200_dev.loc[flores200_dev["file"] != "ajp_Arab.dev",]
    # exclude ISO 639-3 "ajp" from analysis
    # Reasoning:
    #   North Levantine Arabic (formerly "apc") and South Levantine Arabic (formerly "ajp")
    #   were merged into Levantine Arabic ("apc") in 2023. ISO 639-3 code "ajp" is now depracated.
    # Sources:
    #   https://en.wikipedia.org/wiki/South_Levantine_Arabic
    #   https://en.wikipedia.org/wiki/North_Levantine_Arabic
    #   https://glottolog.org/resource/languoid/id/sout3123

assert all(flores200_dev["text"].str.contains("\t") == False)
    # confirming no tab characters, safe to write data as TSV file

os.chdir(PROJECT_ROOT)
flores200_dev.to_csv("01_data_processed/flores200_dev.tsv", sep = "\t", index = False)
