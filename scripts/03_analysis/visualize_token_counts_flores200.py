
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


# %% Clean results DataFrame for plotting and analysis
results_df = pd.read_csv("02_output_model_experiments/token_counts_flores200.csv")

# Get list of models
model_list = results_df["model"].unique().tolist()

# Split filename to extract ISO 639-3 and script
results_df[["iso_script", "filetype"]] = results_df["file"].str.split("." , expand = True)
results_df[["iso_639_3", "script"]] = results_df["iso_script"].str.split("_", expand = True)
results_df = results_df.drop(columns = ["file"])

# Merge to add info from linguameta
linguameta = pd.read_table("01_data_processed/linguameta_clean.tsv")

results_df = pd.merge(left = results_df, right = linguameta, how = "left", on = "iso_639_3", indicator = True)

assert all(results_df["_merge"] == "both")
    # confirming no merge issues
results_df = results_df.drop(columns = ["_merge"])

assert all(results_df.loc[results_df["speakers"].isna(), "iso_639_3"] == "ajp")
    # confirming that speaker figure is only missing for ajp (see process_metalingua.py for rationale)
results_df = results_df.loc[results_df["iso_639_3"] != "ajp",]
    # exclude ajp from analysis

assert results_df["speakers"].isna().sum() == 0
    # no more missing values in "speakers" column

# Show token counts in thousands for ease of reading chart axis labels
results_df["token_count"] = results_df["token_count"] / 1000


# %% Prepare for plotting
# Find languages with multiple scripts
N_model = len(model_list)
script_cnt = results_df.groupby("iso_639_3").size() / N_model
script_cnt = script_cnt.reset_index(drop = False)
script_cnt.columns = ["iso_639_3", "count"]
iso_multiscript_list = script_cnt.loc[script_cnt["count"] > 1, "iso_639_3"].tolist()

# Build dictionary for mapping colors to ISO 639-3 codes
color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:gray", "tab:pink"]
assert len(iso_multiscript_list) == len(color_list) 
N_iso = len(iso_multiscript_list)
iso_color_map = {iso_multiscript_list[i]: color_list[i] for i in range(N_iso)}   


# %% Generate plots for each model
with PdfPages("03_output_analysis/Token Count Charts - FLORES200.pdf") as pdf:
    for model in model_list:
        print(f"Working on report: {model}")
        
        chart_data = results_df.loc[results_df["model"] == model,]
        chart_data = chart_data.sort_values(by = "token_count", ascending = True, ignore_index = True)

        tok_max = chart_data["token_count"].max() + 10
            # +10 for visual padding in chart
            
        axs0_max = 2
        axs1_max = 2
   
        fig, axs = plt.subplots(axs0_max, axs1_max, figsize = (14,14))
        fig.suptitle("Differences in Token Cost Across Languages\nModel: " + model + "\nData: FLORES-200 dev")
        
        # Bar Chart
        axs[0,0].bar(x = chart_data["iso_script"], height = chart_data["token_count"])
        axs[0,0].set_title("Token Count by Language")
        axs[0,0].set_xlabel("Language-Script Pair (in Ascending Order by Token Count)")
        axs[0,0].set_ylabel("Token Count (in Thousands)")
        axs[0,0].set_ylim(0, tok_max)
        axs[0,0].tick_params(axis = "x", labelbottom = False)

        # Histogram         
        axs[0,1].hist(x = chart_data["token_count"], bins = 40, range = (0, tok_max))
        axs[0,1].set_title("Histogram of Token Counts")
        axs[0,1].set_xlabel("Token Count (in Thousands)")
        axs[0,1].set_ylabel("Number of Languages")

        # Scatter Plot by Script
        axs[1,0].scatter(x = chart_data["script"], y = chart_data["token_count"], s = 20, alpha = 0.5)
        axs[1,0].set_title("Token Counts by Script")
        axs[1,0].set_xlabel("Script (ISO 15924)")
        axs[1,0].set_ylabel("Token Count (in Thousands)")
        axs[1,0].set_ylim(0, tok_max)
        axs[1,0].tick_params(axis = "x", labelrotation = 90, labelsize = 8)

        # Scatter Plot for Multiscript Languages
        chart_data = chart_data.loc[chart_data["iso_639_3"].isin(iso_multiscript_list),]
            # Filter chart data to only languages with multiple scripts
        chart_data["color"] = chart_data["iso_639_3"].map(iso_color_map)
            # Assign color to each ISO 639-3 code for plotting
        chart_data = chart_data.sort_values(by = "iso_script", ascending = True, ignore_index = True)
                
        axs[1,1].scatter(x = chart_data["iso_script"], y = chart_data["token_count"], s = 20, alpha = 1,
                         c = chart_data["color"])
        axs[1,1].set_title("Languages with Multiple Scripts")
        axs[1,1].set_xlabel("Language-Script Pair (ISO 639-3, ISO 15924)")
        axs[1,1].set_ylabel("Token Count (in Thousands)")
        axs[1,1].set_ylim(0, tok_max)
        axs[1,1].tick_params(axis = "x", labelrotation = 90, labelsize = 8)

        plt.tight_layout()
        pdf.savefig()
        plt.close()


# %% Gini coefficient
def gini(x):
    """
    Compute the Gini coefficient of a dataset using the sorting shortcut.
 
    G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
 
    where x_i are sorted ascending and i in {1, 2, ... , n}
 
    Parameters
    ----------
    x : array-like
        Non-negative values (e.g., incomes, wealth). Must contain at
        least one positive value.
 
    Returns
    -------
    float
        Gini coefficient, ranging from 0 (perfect equality) to
        just under 1 (maximal inequality).
    """
    x = np.asarray(x, dtype=np.float64).flatten()
 
    if np.any(x < 0):
        raise ValueError("Gini coefficient requires non-negative values.")
    if x.sum() == 0:
        raise ValueError("Gini coefficient is undefined when all values are zero.")
 
    n = len(x)
    x_sorted = np.sort(x)
    ranks = np.arange(1, n + 1)
 
    return (2 * np.sum(ranks * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n


gini_N_list = np.arange(5, 200 + 1, step = 5)
    # +1 to include 200
gini_results = []
ratio_results = []

for model in model_list:
    print(f"Working on {model}")
        
    calc_data = results_df.loc[results_df["model"] == model,]
    calc_data = calc_data.sort_values(by = "speakers", ascending = False, ignore_index = True)

    if model == model_list[0]:
        calc_data[["iso_script", "name", "speakers"]].to_csv("03_output_analysis/gini_languages.csv", index = True)
            # exporting list of languages that Gini coefficients are based on

    for N in gini_N_list:
        token_count_vector = calc_data.loc[calc_data.index < N, "token_count"]
        
        g = gini(token_count_vector)
        gini_result_i = {"model": model, "N": N, "gini": g}
        gini_results.append(gini_result_i)

        max_min_ratio = token_count_vector.max() / token_count_vector.min()
        ratio_result_i = {"model": model, "N": N, "max_min_ratio": max_min_ratio}
        ratio_results.append(ratio_result_i)

gini_df = pd.DataFrame(data = gini_results)
ratio_df = pd.DataFrame(data = ratio_results)


# Gini Charts
with PdfPages("03_output_analysis/Token Gini Charts - FLORES200.pdf") as pdf:
    axs0_max = 2
    axs1_max = 1

    for i in range(len(model_list)):
        print(f"Working on charts for {model}")
        model = model_list[i]

        chart_data_T = results_df.loc[results_df["model"] == model,]
        chart_data_T = chart_data_T.sort_values(by = "speakers", ascending = False, ignore_index = True)

        tok_max = chart_data_T["token_count"].max() + 10

        chart_data_B = gini_df.loc[gini_df["model"] == model,]
        chart_data_B = chart_data_B.sort_values(by = "N", ascending = True, ignore_index = True)

        gini_max = max(0.5, chart_data_B["gini"].max() + 0.05)
        corr_B = round(chart_data_B[["N", "gini"]].corr().iloc[0,1], 2)
            # Pearson correlation between N and gini
     
        fig, axs = plt.subplots(axs0_max, axs1_max, figsize = (14,14))
        fig.suptitle("Gini Coefficient as Function of Language Count\nModel:" + model + "\nData: FLORES-200 dev\n")

        # Bar Chart
        axs[0].bar(x = chart_data_T["iso_script"], height = chart_data_T["token_count"])
        axs[0].set_title("Token Count by Language")
        axs[0].set_xlabel("Language-Script Pair (in Descending Order by Speaker Population)")
        axs[0].set_ylabel("Token Count (in Thousands)")
        axs[0].set_ylim(0, tok_max)
        axs[0].tick_params(axis = "x", labelbottom = False)

        # Line Chart
        axs[1].plot(chart_data_B["N"], chart_data_B["gini"])
        axs[1].set_title("Gini Coefficient by Top N Languages (By Speaker Population)")
        axs[1].set_xlabel("N (Number of Languages)")
        axs[1].set_ylabel("Gini Coefficient")
        axs[1].set_ylim(0, gini_max)
        axs[1].text(0.02, 0.98, f"Correlation(N, Gini) = {corr_B}",
                    transform = axs[1].transAxes,
                    fontsize = 14,
                    verticalalignment = "top",
                    horizontalalignment = "left",
                    bbox=dict(boxstyle = "round", facecolor = "white", alpha = 0.8, edgecolor = "gray"))
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()


out_N_list = [5, 10, 20, 50, 100, 200]
gini_df_long = gini_df.loc[gini_df["N"].isin(out_N_list),]
gini_df_long["gini"] = gini_df_long["gini"].round(2)

ratio_df_long = ratio_df.loc[ratio_df["N"].isin(out_N_list),]
ratio_df_long["max_min_ratio"] = ratio_df_long["max_min_ratio"].round(1)

gini_df_wide = gini_df_long.pivot(index="model", columns="N", values="gini")
gini_df_wide.columns = ["gini_" + str(N) for N in out_N_list]
gini_df_wide = gini_df_wide.reset_index(drop = False)
gini_df_wide = gini_df_wide.sort_values(by = "gini_50", ascending = True, ignore_index = True)

gini_50_sort = gini_df_wide[["model"]].reset_index(drop = False)
gini_50_sort.columns = ["sort_order", "model"]

ratio_df_wide = ratio_df_long.pivot(index="model", columns="N", values="max_min_ratio")
ratio_df_wide.columns = ["max_min_ratio_" + str(N) for N in out_N_list]
ratio_df_wide = ratio_df_wide.reset_index(drop = False)
ratio_df_wide = pd.merge(left = ratio_df_wide, right = gini_50_sort, how = "outer", on = "model", indicator = True)
assert all(ratio_df_wide["_merge"] == "both")
ratio_df_wide = ratio_df_wide.drop(columns = "_merge")
ratio_df_wide = ratio_df_wide.sort_values(by = "sort_order", ascending = True, ignore_index = True)


gini_df_wide.to_csv("03_output_analysis/gini_flores200.csv", index = False)
ratio_df_wide.to_csv("03_output_analysis/ratio_flores200.csv", index = False)

