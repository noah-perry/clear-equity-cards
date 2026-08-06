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


# %% Read linguameta data
linguameta = pd.read_table("00_data_raw/linguameta/linguameta.tsv", keep_default_na = False, na_values = [""])
    # "nan" is a valid ISO 639-3 code

# %% Clean linguameta data
# Add missing languages
assert all( linguameta["iso_639_3_code"] != "san" )
    # Sanskrit is not in linguameta but is included in FLORES-200

new_row = pd.DataFrame({"iso_639_3_code": ["san"], 
                        "english_name": ["Sanskrit"], 
                        "estimated_number_of_speakers": [31_420]})
    # speaker figure based on https://ethnologue.com/language/san

linguameta = pd.concat([linguameta, new_row], ignore_index=True)

# Fill in estimated number of speakers when missing for FLORES-200 languages 
iso_speakers = {"aeb":  12_500_000, # https://en.wikipedia.org/wiki/Tunisian_Arabic
                "apc":  60_000_000, # https://en.wikipedia.org/wiki/Levantine_Arabic
                "arb": 335_000_000, # https://en.wikipedia.org/wiki/Modern_Standard_Arabic
                "crh":     580_000, # https://en.wikipedia.org/wiki/Crimean_Tatar_language
                "gaz":  26_372_150, # https://ethnologue.com/language/gaz
                "kmr":  17_000_000, # https://en.wikipedia.org/wiki/Kurmanji
                "plt":   7_549_210, # https://ethnologue.com/language/plt
                "swh":  97_300_000, # https://en.wikipedia.org/wiki/Swahili
                "tgl":  87_000_000, # https://en.wikipedia.org/wiki/Tagalog_language
                "zsm":  34_000_000, # based on population of Malaysia, https://en.wikipedia.org/wiki/Malaysia 
               }
fill_list = list(iso_speakers.keys())
assert all( linguameta.loc[linguameta["iso_639_3_code"].isin(fill_list), "estimated_number_of_speakers"].isna() )
    # confirming that all languages where the speaker figure will be replaced currently have missing values in "estimated_number_of_speakers"

for iso, speakers in iso_speakers.items():
    linguameta.loc[linguameta["iso_639_3_code"] == iso, "estimated_number_of_speakers"] = speakers
    # filling missing speaker counts

# Correct estimated number of speakers where incorrect
assert all( linguameta.loc[linguameta["iso_639_3_code"] == "lvs", "estimated_number_of_speakers"] == 1 )
    # population of speakers for Standard Latvian [lvs] is 1
linguameta.loc[linguameta["iso_639_3_code"] == "lvs", "estimated_number_of_speakers"] = 2_041_430
    # corrected speaker figure based on https://ethnologue.com/language/lvs

assert all( linguameta.loc[linguameta["iso_639_3_code"].isin(["azj", "azb"]), "estimated_number_of_speakers"] == 24_000_000 )
    # population of speakers for North Azerbaijani [azj] and South Azerbaijani [azb] is the same
linguameta.loc[linguameta["iso_639_3_code"] == "azj", "estimated_number_of_speakers"] = 10_339_420
    # corrected speaker figure based on https://ethnologue.com/language/azj
linguameta.loc[linguameta["iso_639_3_code"] == "azb", "estimated_number_of_speakers"] = 13_319_270
    # corrected speaker figure based on https://ethnologue.com/language/azb



linguameta = linguameta.rename(columns = {"iso_639_3_code": "iso_639_3", 
                                          "english_name": "name", 
                                          "estimated_number_of_speakers": "speakers"})
    # shorten column names

linguameta = linguameta[["iso_639_3", "name", "speakers"]]
    # keep subset of columns


# %% Write cleaned linguameta data to TSV
linguameta.to_csv('01_data_processed/linguameta_clean.tsv', sep = "\t", index = False)
