

# %% Setup
# Python standard library
import os

# Third-party libraries
from dotenv import load_dotenv
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% Project folder
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise EnvironmentError("PROJECT_ROOT is not set. Check your .env file.")
os.chdir(PROJECT_ROOT)


# %% Read FLORES-200 data
flores200_dev = pd.read_table("01_data_processed/flores200_dev.tsv")
flores200_dev[["iso_script", "filetype"]] = flores200_dev["file"].str.split("." , expand = True)
flores200_dev[["iso_639_3", "script"]] = flores200_dev["iso_script"].str.split("_", expand = True)

flores200_langs = flores200_dev.drop(columns = ["filetype", "text"])   
assert flores200_langs["iso_script"].is_unique

flores200_langs = flores200_langs.loc[flores200_langs["iso_639_3"] != "ajp",]
    # exclude ajp from analysis (see process_metalingua.py for rationale)
    # TODO: move this exclusion to "script/01_process_data/process_flores200.py"


# %% Read macrolanguage mappings
macro = pd.read_csv("01_data_processed/macrolanguage_mappings.csv")
assert macro["iso_639_3_indiv"].is_unique


# %% Read Common Crawl data
cc = pd.read_csv("01_data_processed/commoncrawl_clean.csv")
assert cc["iso_639_3"].is_unique


# %% Read LinguaMeta data
lm = pd.read_table("01_data_processed/linguameta_clean.tsv")
assert lm["iso_639_3"].is_unique


# %% Create DataFrame of various language rankings for FLORES-200 languages
# Identify individual languages in FLORES-200 that are part of a macrolanguage and add column with macrolanguage ISO 639-3 codes
macro = macro.rename(columns = {"iso_639_3_indiv": "iso_639_3"})
flores200_ranks = pd.merge(left = flores200_langs, right = macro, how = "left", on = "iso_639_3")
    # merge adds "iso_639_3_macro"

# Merge FLORES-200 and Common Crawl using FLORES-200's original set of ISO 639-3 codes
merge_map = {"left_only": 0, "both": 1}
flores200_ranks = pd.merge(left = flores200_ranks, right = cc, how = "left", on = "iso_639_3", indicator = True)
    # merge adds "cc_pages" and "cc_rank" (to be renamed "*_o" below)
flores200_ranks["flag_orig"] = flores200_ranks["_merge"].map(merge_map)
    # create column to track whether match was found in this merge
flores200_ranks = flores200_ranks.drop(columns = ["_merge"])
flores200_ranks = flores200_ranks.rename(columns = {"cc_pages": "cc_pages_o", 
                                                    "cc_rank": "cc_rank_o"})

# Merge FLORES-200 and Common Crawl using added macrolanguage ISO 639-3 codes
cc = cc.rename(columns = {"iso_639_3": "iso_639_3_macro"})
flores200_ranks = pd.merge(left = flores200_ranks, right = cc, how = "left", on = "iso_639_3_macro", indicator = True)
    # merge adds "cc_pages" and "cc_rank" (to be renamed "*_m" below)
flores200_ranks["flag_macro"] = flores200_ranks["_merge"].map(merge_map)
    # create column to track whether match was found in this merge
flores200_ranks = flores200_ranks.drop(columns = ["_merge"])
flores200_ranks = flores200_ranks.rename(columns = {"cc_pages": "cc_pages_m",
                                                    "cc_rank": "cc_rank_m"})

# Create Common Crawl ranking, giving precendence to original ISO 639-3 codes over macrolanguage ISO 639-3 codes
flores200_ranks = flores200_ranks.rename(columns = {"cc_pages_o": "cc_pages"})
flores200_ranks = flores200_ranks.rename(columns = {"cc_rank_o": "cc_rank"})
    # use original code-based values wherever possible
flores200_ranks.loc[flores200_ranks["cc_pages"].isna(), "cc_pages"] = flores200_ranks["cc_pages_m"]
flores200_ranks.loc[flores200_ranks["cc_rank"].isna(), "cc_rank"] = flores200_ranks["cc_rank_m"]
    # add macrolanguage code-based values only where original code-based rank not available
flores200_ranks = flores200_ranks.drop(columns = ["cc_pages_m", "cc_rank_m"])

# Add speaker counts from LinguaMeta
flores200_ranks = pd.merge(left = flores200_ranks, right = lm, how = "left", on = "iso_639_3", indicator = True)
    # merge adds "speakers"
assert all(flores200_ranks["_merge"] == "both")
flores200_ranks = flores200_ranks.drop(columns = ["_merge"])

# Create ranking based on speaker count only
flores200_ranks = flores200_ranks.sort_values(by = "speakers", ascending = False, ignore_index = True)
flores200_ranks["spk_rank"] = flores200_ranks.index + 1

# Create ranking based on Common Crawl rank (primarily) and speaker count (secondarily)
flores200_ranks = flores200_ranks.sort_values(by = ["cc_pages", "speakers"], ascending = [False, False], ignore_index = True)
flores200_ranks["cc_spk_rank"] = flores200_ranks.index + 1


"""
# %% Checks
flores200_ranks["flag_orig"].value_counts(dropna = False)

flores200_ranks["flag_om"] = flores200_ranks["flag_orig"] + flores200_ranks["flag_macro"]
flores200_ranks["flag_om"].value_counts(dropna = False)

N_matched = flores200_ranks.loc[flores200_ranks["flag_om"] != 0,].shape[0]

flores200_ranks[["cc_spk_rank", "spk_rank"]].corr()
flores200_ranks.loc[flores200_ranks["cc_spk_rank"] <= N_matched, ["cc_spk_rank", "spk_rank"]].corr()
flores200_ranks.loc[flores200_ranks["cc_spk_rank"] >  N_matched, ["cc_spk_rank", "spk_rank"]].corr()

plt.scatter(flores200_ranks["cc_spk_rank"], flores200_ranks["spk_rank"])
"""


# %% Keep columns to be used in analysis scripts
flores200_ranks = flores200_ranks[["file", "iso_script", "iso_639_3", "script", "name", "cc_pages", "speakers", "cc_spk_rank", "spk_rank"]]


# %% Write data to CSV
flores200_ranks.to_csv("01_data_processed/flores200_langinfo.tsv", sep = "\t", index = False)

