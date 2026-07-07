"""
Notice: 
A small amount of data is taken from Ethnologue for nonprofit research purposes in compliance with SIL's fair use guidelines
See https://www.ethnologue.com/general-terms-use/
"""


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


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Read linguameta data
linguameta = pd.read_table("00_data_raw/linguameta/linguameta.tsv")

# %% Clean linguameta data
# Add missing languages
assert all(linguameta["iso_639_3_code"] != "san")
    # Sanskrit is not in linguameta but is included in FLORES-200

new_row = pd.DataFrame({"iso_639_3_code": ["san"], 
                        "english_name": ["Sanskrit"], 
                        "estimated_number_of_speakers": [31420]})
    # speaker figure based on https://ethnologue.com/language/san

linguameta = pd.concat([linguameta, new_row], ignore_index=True)

# Fill in estimated number of speakers when missing for FLORES-200 languages 
iso_speakers = {"aeb":  12500000, # https://en.wikipedia.org/wiki/Tunisian_Arabic
                "ajp":      None, # iso "ajp" was merged with "apc" in 2023 according to https://en.wikipedia.org/wiki/South_Levantine_Arabic, https://en.wikipedia.org/wiki/North_Levantine_Arabic
                "apc":  60000000, # https://en.wikipedia.org/wiki/Levantine_Arabic
                "arb": 335000000, # https://en.wikipedia.org/wiki/Modern_Standard_Arabic
                "crh":    580000, # https://en.wikipedia.org/wiki/Crimean_Tatar_language
                "gaz":  26372150, # https://ethnologue.com/language/gaz
                "kmr":  17000000, # https://en.wikipedia.org/wiki/Kurmanji
                "plt":   7549210, # https://ethnologue.com/language/plt
                "swh":  97300000, # https://en.wikipedia.org/wiki/Swahili
                "tgl":  87000000, # https://en.wikipedia.org/wiki/Tagalog_language
                "zsm":  34000000} # based on population of Malaysia, https://en.wikipedia.org/wiki/Malaysia 

fill_list = list(iso_speakers.keys())
assert all(linguameta.loc[linguameta["iso_639_3_code"].isin(fill_list), "estimated_number_of_speakers"].isna())
    # confirming that all languages where the speaker figure will be replaced currently have missing values in "estimated_number_of_speakers"

for iso, speakers in iso_speakers.items():
    linguameta.loc[linguameta["iso_639_3_code"] == iso, "estimated_number_of_speakers"] = speakers


# %% Write cleaned linguameta data to TSV
linguameta.to_csv('01_data_processed/linguameta_clean.tsv', sep = "\t", index = False)
