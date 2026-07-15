
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


# %% Read data
macro_df = pd.read_table("00_data_raw/iso-639-3/iso-639-3-macrolanguages.tab", na_filter = False)


# %% Clean data
macro_df = macro_df.rename(columns = {"M_Id": "iso_639_3_macro", "I_Id": "iso_639_3_indiv"})
macro_df = macro_df[["iso_639_3_macro", "iso_639_3_indiv"]]


# %% Write data
macro_df.to_csv('01_data_processed/macrolanguage_mappings.csv', index = False)
