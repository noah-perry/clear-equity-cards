
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
cc = pd.read_csv("00_data_raw/commoncrawl/languages.csv", na_filter = True)


# %% Clean data
crawl_list = cc["crawl"].unique().tolist()
last_crawl = max(crawl_list)
    # find identifier of most recent crawl

assert last_crawl == "CC-MAIN-2026-25"
    # assertion fails if data for a more recent crawl has been added

cc_last = cc.loc[cc["crawl"] == last_crawl,]
    # filter data to most recent crawl

cc_last = cc_last.loc[cc_last["primary_language"] != "<unknown>",]
    # exclude data where language is <unknown>

cc_last = cc_last.sort_values(by = "pages", ascending = False, ignore_index = True)
cc_last["cc_rank"] = cc_last.index + 1
    # rank languages by number of pages crawled, ranking starts at 1

cc_last = cc_last.rename(columns = {"primary_language": "iso_639_3"})
cc_last = cc_last[["iso_639_3", "cc_rank"]]


# %% Write data
cc_last.to_csv("01_data_processed/commoncrawl_language_ranking.csv", index = False)
