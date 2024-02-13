#!/usr/bin/env python3
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.patches import Patch

from common import RENAME_FLAGS, DEFAULT_SEA_FLUX, FINAL_PROFILE_DATABASE, PARSER_DIRECTORY, CROSS_SECTION_KEPLER_2021
from common import calc_err_due, calc_err_sdc, load_profile_database


def parse_header(r):
    pattern = r".*nvcc_version:MAJOR_(\d+)_MINOR_(\d+) nvcc_optimization_flags:(.*)"
    m = re.match(pattern, r["header"])
    r["nvcc"] = f"{m.group(1)}.{m.group(2)}" if m else None
    r["flag"] = RENAME_FLAGS[m.group(3)] if m else None
    # Split the benchmark column in two
    r["app"], r["ECC"] = re.match(r"(\S+)_ECC_(\S+)", r["benchmark"]).groups()
    return r


def get_fit(cross_section_database_path):
    df = pd.read_csv(cross_section_database_path)
    # Parse header
    df = df.apply(parse_header, axis="columns")
    df = df[
        # --> The only ones that have ACC time
        # Using this approach is the most correct one, the runs that are too small CANNOT be considered,
        # they have a problem of aborting too much. On the JSC paper we used only the ECC ON configs,
        # which do not change because of the low-ACC TIME, so filtering the values by 100s we got only the good runs
        (df["acc_time"] > 100) &
        # MAIS DE 1/3
        (df["Time Beam Off"] <= 1200) &
        # SDC > 0 or DUE > 0, if at least one is bigger than zero
        ((df["#SDC"] > 0) | (df["#DUE"] > 0)) &
        # Remove lava
        (~df["benchmark"].str.contains("lava")) &
        # Remove broken runs
        (df["header"] != "unknown")
        ]
    # Remove unnecessary info
    k40_cs = df[["app", "ECC", "nvcc", "flag", "start_dt", "end_dt", "#SDC", "#DUE", "acc_time", "Flux 1h",
                 "Time Beam Off"]].copy()
    # Fluency hand calc
    k40_cs["Fluency"] = k40_cs["acc_time"] * k40_cs["Flux 1h"]
    k40_grouped = k40_cs.groupby(["ECC", "nvcc", "flag"]).sum()
    k40_grouped = k40_grouped[(k40_grouped["#SDC"] > 0) & (k40_grouped["#DUE"] > 0)]
    # Too much beam off accumulated from all the runs is also harmful
    k40_grouped = k40_grouped[(k40_grouped["Time Beam Off"] / k40_grouped["acc_time"]) <= 2.0]

    """ Calc cross section """
    k40_grouped["SDC"] = (k40_grouped["#SDC"] / k40_grouped["Fluency"])
    k40_grouped["DUE"] = (k40_grouped["#DUE"] / k40_grouped["Fluency"])
    k40_grouped["Err SDC"] = k40_grouped.apply(calc_err_sdc, axis="columns")
    k40_grouped["Err DUE"] = k40_grouped.apply(calc_err_due, axis="columns")
    return k40_grouped


def mean_workload_between_failures(cross_section_df, profile_df):
    """
    Based on: Impact of GPUs Parallelism Management on Safety-Critical and HPC Applications Reliability (Rech et al.)
    """

    workload = 4096 ** 2
    final_df = dict()
    for harden in ["OFF", "ON"]:
        reliability_metrics = cross_section_df.loc[f"{harden}"].copy()

        """ SUM MWBF """
        # MTBF - Mean Time Between Failure = 1 / (cross_section * sea_flux)
        # It is already multiplied by DEFAULT_SEA_FLUX
        full_cross_section = reliability_metrics["SDC"] + reliability_metrics["DUE"]
        reliability_metrics["MTBF"] = 1.0 / full_cross_section
        # MEBF - Mean Execution Between Failure = MTBF / execution_time
        execution_time = profile_df.loc[reliability_metrics.index, "execution_time"]
        reliability_metrics["MEBF"] = reliability_metrics["MTBF"] / execution_time
        # MWBF - Mean Work Between Failure (MWBF) = MEBF * workload
        reliability_metrics["MWBF"] = reliability_metrics["MEBF"] * workload
        final_df[harden] = reliability_metrics
    final_df = pd.concat(final_df)
    final_df["Err MWBF"] = final_df[["Err SDC", "Err DUE"]].sum(axis="columns")
    # The errors are relative to the FIT error
    final_df["Err MWBF"] *= (final_df["MWBF"] / final_df[["SDC", "DUE"]].sum(axis="columns"))

    mwbf = final_df["MWBF"]
    err = final_df["Err MWBF"]
    """PLOT GRAPH"""
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    width = 0.35  # the width of the bars
    color_113 = "chocolate"
    color_102 = "dimgrey"
    for axi, ecc in enumerate(["OFF", "ON"]):
        df_ecc = mwbf.loc[ecc].reset_index()
        x = np.arange(df_ecc["flag"].shape[0])
        df_err = err.loc[ecc]
        rects_mwbf = ax[axi].bar(x - width / 2, df_ecc["MWBF"].values, width, label='MWBF', edgecolor="black",
                                 linewidth=1, align='edge',
                                 yerr=df_err.values, capsize=3, color=color_113)
        ax[axi].set_xticks(x)
        ax[axi].set_xticklabels(df_ecc["flag"])
        ax[axi].set_ylabel("Mean Work Between Failures", fontdict=dict(fontsize=12))
        ax[axi].set_title(f"ECC {ecc}")
        ax[axi].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))
        # Setting the hatch
        for i in range(len(df_ecc[df_ecc.nvcc == "10.2"].index)):
            # rects_mwbf[i].set_hatch("////")
            rects_mwbf[i].set_color(color_102)
            rects_mwbf[i].set_edgecolor("black")

    custom_lines = [
        Patch(facecolor=color_102, edgecolor='black', label='NVCC 10.2'),
        Patch(facecolor=color_113, edgecolor='black', label='NVCC 11.3')
    ]
    plt.legend(handles=custom_lines, loc="best", edgecolor=None, frameon=False)

    plt.tight_layout()
    plt.savefig(f"{PARSER_DIRECTORY}/fig/mwbf.pdf")
    plt.show()
    return final_df


def fit_graph(cross_section_df):
    """Plot the graph"""
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    width = 0.35  # the width of the bars
    for axi, ecc in enumerate(["OFF", "ON"]):
        df_ecc = cross_section_df.loc[ecc].reset_index()
        x = np.arange(df_ecc["flag"].shape[0])
        sdc, errs_dc = df_ecc["SDC"].values, df_ecc["Err SDC"].values
        due, errs_due = df_ecc["DUE"].values, df_ecc["Err DUE"].values

        rects_sdc = ax[axi].bar(x - width / 2, sdc, width, label='SDC', edgecolor="black", linewidth=1,
                                yerr=errs_dc, capsize=3, color="RED")
        rects_due = ax[axi].bar(x + width / 2, due, width, label='DUE', edgecolor="black", linewidth=1,
                                yerr=errs_due, capsize=3, color="BLUE")
        ax[axi].set_xticks(x)
        ax[axi].set_xticklabels(df_ecc["flag"])
        ax[axi].set_ylabel("Failure In Time FIT [a.u.]", fontdict=dict(fontsize=12))
        ax[axi].set_title(f"ECC {ecc}")

        # Setting the hatch
        for i in range(len(df_ecc[df_ecc.nvcc == "10.2"].index)):
            rects_sdc[i].set_hatch("////")
            rects_due[i].set_hatch("////")
            rects_sdc[i].set_color("lightcoral")
            rects_due[i].set_color("deepskyblue")
            rects_sdc[i].set_edgecolor("black")
            rects_due[i].set_edgecolor("black")
            rects_sdc[i].set_linewidth(1)
            rects_due[i].set_linewidth(1)

    # Only for ecc off
    max_y = cross_section_df[["SDC", "DUE"]].max().max() + cross_section_df[["Err SDC", "Err DUE"]].max().max()
    ax[0].set_ylim(0, max_y)

    custom_lines = [Patch(facecolor='lightcoral', edgecolor='black', label='SDC 10.2', hatch="////"),
                    Patch(facecolor='deepskyblue', edgecolor='black', label='DUE 10.2', hatch="////"),
                    Patch(facecolor='RED', edgecolor='black', label='SDC 11.3'),
                    Patch(facecolor='BLUE', edgecolor='black', label='DUE 11.3')]
    plt.legend(handles=custom_lines, loc="best", edgecolor=None, frameon=False)

    plt.tight_layout()
    plt.savefig(f"{PARSER_DIRECTORY}/fig/fit_rate.pdf")
    plt.show()


def main():
    """ Cross-section extract """
    cross_section_df = get_fit(CROSS_SECTION_KEPLER_2021)
    cross_section_df["SDC"] *= DEFAULT_SEA_FLUX
    cross_section_df["DUE"] *= DEFAULT_SEA_FLUX
    cross_section_df["Err SDC"] *= DEFAULT_SEA_FLUX
    cross_section_df["Err DUE"] *= DEFAULT_SEA_FLUX
    cross_section_df["Beam off ratio"] = cross_section_df["Time Beam Off"] / cross_section_df["acc_time"]

    norm_value = cross_section_df[["SDC", "DUE"]].min().min()
    k40_grouped = cross_section_df[["SDC", "DUE", "Err SDC", "Err DUE"]]
    gb = k40_grouped.reset_index().groupby(["ECC", "nvcc"])
    relative_std = (gb.agg(np.std, ddof=0) * 100) / gb.mean()
    print(k40_grouped)
    print(relative_std)
    print(relative_std.loc["ON"].describe())

    k40_grouped /= norm_value
    fit_graph(cross_section_df=k40_grouped)

    # first load profile data
    profile_df = load_profile_database(FINAL_PROFILE_DATABASE).sort_index()

    mwbf_df = mean_workload_between_failures(cross_section_df=k40_grouped,
                                             profile_df=profile_df.loc[pd.IndexSlice["kepler", "FMXM"]])
    with pd.ExcelWriter(f"{PARSER_DIRECTORY}/sheets/cross_section_ChipIr.xlsx") as writer:
        mwbf_df.to_excel(writer, sheet_name="CS+MWBF")
        cross_section_df.to_excel(writer, sheet_name="RawCS")


if "__main__" == __name__:
    main()
