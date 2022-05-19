# create a pareto plot of outcomes

import matplotlib.pyplot as plt
import pandas as pd
from experiment import DELIMITER
import ast
import numpy as np

# ---------
# read data
# ---------

# dict of outcome files
outcome_file_dict = {
    "Istanbul": "/research/blood-matching_github/output/blood_experiment_istanbul.csv",
    "Jakarta": "/research/blood-matching_github/output/blood_experiment_jakarta.csv",
    "San Francisco": "/research/blood-matching_github/output/blood_experiment_sanfrancisco.csv",
    "S\\~ao Paulo": "/research/blood-matching_github/output/blood_experiment_saopaulo.csv",
}


def get_city_data(file):
    df = pd.read_csv(file, sep=DELIMITER, skiprows=1)

    # get the "fair" outcomes
    df["recipient_weight_dict"] = df["recipient_matched_weight"].apply(
        lambda x: ast.literal_eval(x)
    )
    fair_weight_dict = df[df["method"] == "aggregated_random"][
        "recipient_weight_dict"
    ].values[0]

    df["recipient_outcome_dict"] = df["recipient_weight_dict"].apply(
        lambda d: {
            key: d[key] / fair_weight_dict[key] for key in fair_weight_dict.keys()
        }
    )

    # get mean matched weight for each policy
    df_agg = df.groupby(by="method")["total_matched_weight"].mean().reset_index()

    # get all recipient scaled utility for all methods
    scaled_u_dict = {}
    for method in df["method"].unique():
        # get average scaled utility for each recipient over all realizations, for this method
        scaled_u_dict[method] = {
            id: list(
                df[df["method"] == method].apply(
                    lambda row: row["recipient_outcome_dict"][id], axis=1
                )
            )
            for id in fair_weight_dict.keys()
        }

    # get mean (avg) scaled utility over all donors (this is the metric we aim for)
    mean_scaled_u_dict = {}
    for method in df["method"].unique():
        # get average scaled utility for each recipient over all realizations, for this method
        mean_scaled_u_dict[method] = np.median(
            [
                df[df["method"] == method]
                .apply(lambda row: row["recipient_outcome_dict"][id], axis=1)
                .mean()
                for id in fair_weight_dict.keys()
            ]
        )

    # get min (avg) scaled utility over all donors (this is the metric we aim for)
    min_scaled_u_dict = {}
    for method in df["method"].unique():
        # get average scaled utility for each recipient over all realizations, for this method
        min_scaled_u_dict[method] = np.min(
            [
                df[df["method"] == method]
                .apply(lambda row: row["recipient_outcome_dict"][id], axis=1)
                .mean()
                for id in fair_weight_dict.keys()
            ]
        )

    # get max (avg) scaled utility over all donors (this is the metric we aim for)
    max_scaled_u_dict = {}
    for method in df["method"].unique():
        # get average scaled utility for each recipient over all realizations, for this method
        max_scaled_u_dict[method] = np.max(
            [
                df[df["method"] == method]
                .apply(lambda row: row["recipient_outcome_dict"][id], axis=1)
                .mean()
                for id in fair_weight_dict.keys()
            ]
        )

    # get alpha
    # (this is the ratio of max_scaled_u_dict / min_scaled_u_dict)
    alpha_dict = {}
    for method in df["method"].unique():
        # get average scaled utility for each recipient over all realizations, for this method
        alpha_dict[method] = min_scaled_u_dict[method] / max_scaled_u_dict[method]

    # get mean matched weight for all methods
    mean_wt = {}
    for method in df["method"].unique():
        # get average scaled utility for each recipient over all realizations, for this method
        mean_wt[method] = np.mean(
            [
                df[df["method"] == method]["total_matched_weight"].mean()
                for id in fair_weight_dict.keys()
            ]
        )

    # get "cr" (ratio of wt to max_wt)
    cr_dict = {}
    for method in df["method"].unique():
        # get average scaled utility for each recipient over all realizations, for this method
        cr_dict[method] = mean_wt[method] / mean_wt["greedy"]

    # ----------------------------------------
    # aggregate outcome data across recipients
    # ----------------------------------------

    df["median_recip_outcome"] = df["recipient_outcome_dict"].apply(
        lambda x: np.median(list(x.values()))
    )
    df["mean_recip_outcome"] = df["recipient_outcome_dict"].apply(
        lambda x: np.mean(list(x.values()))
    )
    df["min_recip_outcome"] = df["recipient_outcome_dict"].apply(
        lambda x: min(list(x.values()))
    )
    df["max_recip_outcome"] = df["recipient_outcome_dict"].apply(
        lambda x: max(list(x.values()))
    )

    data = {
        "mean_scaled_u_dict": mean_scaled_u_dict,
        "mean_wt": mean_wt,
        "min_scaled_u_dict": min_scaled_u_dict,
        "max_scaled_u_dict": max_scaled_u_dict,
        "cr_dict": cr_dict,
        "alpha_dict": alpha_dict,
    }
    return data


# ------------------------
# get all city data
# ------------------------

data_dict = {key: get_city_data(val) for key, val in outcome_file_dict.items()}
city_names = list(data_dict.keys())
method_names = list(data_dict[city_names[0]]["cr_dict"].keys())

# ---------------------
# bookkeeping for plots
# ---------------------

# set fonts
plt.rcParams["font.family"] = "Times New Roman"
plt.rc("text", usetex=True)

plt.rc("figure", titlesize=10)  # fontsize of the figure title
plt.rc("font", size=9)


# overall marker size
markersize = 8

# plot style for each method
style_dict = {
    "aggregated_random": {
        "color": "b",
        "linestyle": "-",
        "marker": "x",
        "markersize": markersize,
    },
    "greedy": {"color": "r", "marker": "o", "markersize": markersize},
    "lp_non_adapt": {
        "color": "purple",
        "marker": "^",
        "linestyle": ":",
        "markersize": markersize,
    },
    "lp_appx_adapt": {
        "color": "green",
        "marker": "+",
        "linestyle": ":",
        "markersize": markersize,
    },
}


# --------------------------------------------------------------------------------
#  plot alpha vs. CR
# --------------------------------------------------------------------------------

# top row is expected matched weight (y) by scaled utility (x)
# bottom row is CR (y) by alpha (x)

# each col is a different city

# figure size
fig_height = 4
fig_width = 7


num_cities = len(data_dict)

# create plot
num_rows = 2
num_cols = num_cities
fig, axs = plt.subplots(
    num_rows, num_cols, figsize=(fig_width, fig_height), sharex="row",
)

axs[0, 0].set_ylabel("Matched Weight")
axs[0, 0].set_xlabel("Normalized Recip. Weight")

axs[1, 0].set_ylabel("$\\texttt{CR}$ (Competitive Ratio)")
axs[1, 0].set_xlabel("$\\texttt{Gamma}$ (Expected Fairness)")

for i_col, city_name in enumerate(city_names):
    axs[0, i_col].set_title(city_name)

    # ------------------------------------------------
    # top plot : matched weight and normalized outcome

    # -- random --
    method = "aggregated_random"

    x = data_dict[city_name]["mean_scaled_u_dict"][method]
    y = data_dict[city_name]["mean_wt"][method]
    x_min = [
        data_dict[city_name]["mean_scaled_u_dict"][method]
        - data_dict[city_name]["min_scaled_u_dict"][method]
    ]
    x_max = [
        data_dict[city_name]["max_scaled_u_dict"][method]
        - data_dict[city_name]["mean_scaled_u_dict"][method]
    ]
    axs[0, i_col].errorbar(x, y, xerr=[x_min, x_max], **style_dict[method])

    # -- greedy --
    method = "greedy"

    x = data_dict[city_name]["mean_scaled_u_dict"][method]
    y = data_dict[city_name]["mean_wt"][method]
    x_min = [
        data_dict[city_name]["mean_scaled_u_dict"][method]
        - data_dict[city_name]["min_scaled_u_dict"][method]
    ]
    x_max = [
        data_dict[city_name]["max_scaled_u_dict"][method]
        - data_dict[city_name]["mean_scaled_u_dict"][method]
    ]
    axs[0, i_col].errorbar(x, y, xerr=[x_min, x_max], **style_dict[method])

    # -- appx adaptive --
    method = "lp_appx_adapt"

    # get all lp-non-adaptive policies
    policies = [m for m in method_names if method in m]

    # get the gamma value for each
    gamma_list = [float(m.split("_")[-1]) for m in policies]

    x_list = [data_dict[city_name]["mean_scaled_u_dict"][name] for name in policies]
    y_list = [data_dict[city_name]["mean_wt"][name] for name in policies]

    x_min = [
        data_dict[city_name]["mean_scaled_u_dict"][name]
        - data_dict[city_name]["min_scaled_u_dict"][name]
        for name in policies
    ]
    x_max = [
        data_dict[city_name]["max_scaled_u_dict"][name]
        - data_dict[city_name]["mean_scaled_u_dict"][name]
        for name in policies
    ]

    # ax.plot(x_list, y_list, **style_dict[method])
    axs[0, i_col].errorbar(x_list, y_list, xerr=[x_min, x_max], **style_dict[method])

    annotate_x = [x_list[i] + x_max[i] for i in range(len(x_max))]
    # annotate each point
    offset_y = -20
    offset_x = 2
    for i, (x, y, gamma), in enumerate(zip(annotate_x, y_list, gamma_list)):
        if i % 10 == 0:
            axs[0, i_col].annotate(
                f"{gamma}",
                (x, y),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                arrowprops=dict(
                    arrowstyle="->",
                    color="0.5",
                    shrinkA=5,
                    shrinkB=5,
                    patchA=None,
                    patchB=None,
                ),
            )

    # ------------------------------------------------
    # bottom plot : CR and EF

    # -- random --
    method = "aggregated_random"

    y_dict = data_dict[city_name]["cr_dict"]
    x_dict = data_dict[city_name]["alpha_dict"]

    x = x_dict[method]
    y = y_dict[method]

    axs[1, i_col].plot(x, y, **style_dict[method])

    # -- greedy --
    method = "greedy"

    x = x_dict[method]
    y = y_dict[method]

    axs[1, i_col].plot(x, y, **style_dict[method])

    # -- appx adaptive --
    method = "lp_appx_adapt"

    # get all lp-non-adaptive policies
    lp_non_adap_policies = [m for m in method_names if method in m]

    # get the gamma value for each
    lp_non_adap_gamma = [float(m.split("_")[-1]) for m in lp_non_adap_policies]

    # get all lp-non-adaptive policies
    lp_non_adap_policies = [m for m in method_names if method in m]

    # get the gamma value for each
    gamma_list = [float(m.split("_")[-1]) for m in lp_non_adap_policies]

    x_list = [x_dict[name] for name in lp_non_adap_policies]
    y_list = [y_dict[name] for name in lp_non_adap_policies]

    # annotate each point
    offset_y = -20
    offset_x = -10
    for i, (x, y, gamma), in enumerate(zip(x_list, y_list, gamma_list)):
        if i % 5 == 0:
            axs[1, i_col].annotate(
                f"{gamma}",
                (x, y),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                arrowprops=dict(
                    arrowstyle="->",
                    color="0.5",
                    shrinkA=5,
                    shrinkB=5,
                    patchA=None,
                    patchB=None,
                ),
            )

    axs[1, i_col].plot(x_list, y_list, **style_dict[method])

plt.subplots_adjust(
    wspace=0.1, hspace=0.1
)  # left=None, bottom=None, right=None, top=None,
plt.tight_layout()

plt.savefig("/plots/cities.pdf")
