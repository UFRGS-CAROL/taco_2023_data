#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.io as pio
from matplotlib.patches import Patch
from plotly.subplots import make_subplots

from common import CROSS_SECTION_KEPLER, CROSS_SECTION_VOLTA, CROSS_SECTION_KEPLER_2021
from common import FINAL_PROFILE_DATABASE, KEPLER_PROFILE_DATABASE, VOLTA_PROFILE_DATABASE
from common import LD_INSTRUCTIONS, GP_INSTRUCTIONS, REDUCED_APP_NAMES, PARSER_DIRECTORY, FINAL_NVBITFI_DATABASE
from common import PARSED_FLAGS, INSTRUCTION_GROUP_ID, METRIC_NAMES, MARKERS_PLOTLY, FP32_INSTRUCTIONS
from common import calc_err_sdc, calc_err_due, load_profile_database
from draw_ChipIR_cross_section import get_fit


def load_injection_database(nvbit_database_path):
    nvbitfi_df = pd.read_csv(nvbit_database_path, sep=";")
    print(nvbitfi_df["#faults"].unique(), nvbitfi_df["fault_num"].unique())
    nvbitfi_df["flag"] = nvbitfi_df["flag"].apply(lambda r: PARSED_FLAGS[r])
    nvbitfi_df["igid"] = nvbitfi_df["igid"].apply(lambda r: INSTRUCTION_GROUP_ID[r])
    nvbitfi_df = nvbitfi_df[["board", "benchmark", "nvcc", "flag", 'igid', '#faults', 'SDC', 'DUE', 'FRAMEWORK_DUE']]
    nvbitfi_df["nvcc"] = nvbitfi_df["nvcc"].astype(str)
    nvbitfi_df = nvbitfi_df.rename(columns={"benchmark": "app"})
    avf_df = nvbitfi_df.groupby(["board", "app", "nvcc", "flag", "igid"]).agg({"#faults": np.average,
                                                                               "SDC": np.sum, "DUE": np.sum,
                                                                               "FRAMEWORK_DUE": np.sum})
    avf_df["SDC AVF"] = avf_df["SDC"] / avf_df["#faults"]
    avf_df["DUE AVF"] = (avf_df["DUE"] + avf_df["FRAMEWORK_DUE"]) / avf_df["#faults"]
    avf_df = avf_df[["SDC AVF", "DUE AVF"]]
    avf_unstacked = avf_df.unstack().fillna(0)
    return avf_unstacked


def load_old_micro_metrics():
    micros = ['FFMA', 'FADD', 'FMUL', 'IMUL', 'IADD', 'IMAD']
    micros_kepler = micros + ['LDST']
    kepler = pd.read_csv(KEPLER_PROFILE_DATABASE)
    volta = pd.read_csv(VOLTA_PROFILE_DATABASE)
    kepler["metric"] = kepler["metric"].apply(lambda x: METRIC_NAMES[x] if x in METRIC_NAMES else x)
    volta["metric"] = volta["metric"].apply(lambda x: METRIC_NAMES[x] if x in METRIC_NAMES else x)
    kepler = kepler.set_index("metric").transpose()
    volta = volta.set_index("metric").transpose()
    kepler["board"] = "kepler"
    volta["board"] = "volta"
    df = pd.concat([kepler, volta])
    df.index.name = "instruction"
    return df.loc[micros_kepler, :]


def draw_plotly_pvf_graph(df, y_name, output_image_file):
    boards = df["board"].unique()
    nvcc_colors = {"10.2": 'LightSkyBlue', "11.3": 'MediumPurple'}
    text_font_size = 30
    df["markers"] = df["flag"].apply(lambda x: MARKERS_PLOTLY[x])

    fig = make_subplots(
        rows=1, cols=len(boards), shared_yaxes=True, horizontal_spacing=0.01,
        subplot_titles=[i.capitalize() for i in boards]
    )
    for i, board in enumerate(boards):
        board_df = df[df["board"] == board]
        for marker_category, symbol in MARKERS_PLOTLY.items():
            maker_df = board_df[board_df["markers"] == symbol]
            for nvcc in board_df["nvcc"].unique():
                filtered_data = maker_df[maker_df["nvcc"] == nvcc]
                fig.add_trace(
                    px.strip(filtered_data, x="app", y="P(SDC)", color="flag", facet_row="board")
                    .update_traces(marker=dict(
                        color=nvcc_colors[nvcc], symbol=symbol, size=25, line=dict(width=2, color='DarkSlateGrey')
                    )).data[0],
                    row=1, col=i + 1,
                )
    fig.update_traces(showlegend=False)  # Hide the legend for all traces
    for nvcc, color in nvcc_colors.items():
        second_legend_trace = plotly.graph_objs.Scatter(
            x=[None], y=[None], mode='markers', marker=dict(size=30),
            showlegend=True, legendgroup=nvcc, name=f'NVCC {nvcc}'
        )
        fig.add_trace(second_legend_trace)
        fig.update_traces(marker=dict(symbol='square', line=dict(width=2), color=color),
                          selector=dict(legendgroup=nvcc))
    for flag, maker in MARKERS_PLOTLY.items():
        fig.add_trace(plotly.graph_objs.Scatter(x=[None], y=[None], mode='markers',
                                                marker=dict(size=25, color="black", symbol=maker), name=flag))
    # Create a new layout with custom properties
    new_layout = plotly.graph_objs.Layout(
        yaxis_title=y_name,
        legend=dict(orientation="h", x=0.0, y=1.1, font=dict(size=text_font_size, color="black")),
        margin=dict(l=5, r=5, t=5, b=5),  # Adjust the margin values as needed
    )
    # Add the new layout to the existing figure
    fig.update_layout(new_layout)

    for i in range(1, len(boards) + 1):
        fig.update_xaxes(tickfont=dict(size=text_font_size, color="black"), ticks="outside", col=i)
        fig.layout.annotations[i - 1].update(font=dict(size=text_font_size, color="black"))
    fig.update_yaxes(title_font=dict(size=text_font_size, color="black"))
    fig.update_yaxes(tickfont=dict(size=text_font_size, color="black"), ticks="outside")
    # fig.show()
    pio.write_image(fig, output_image_file, width=1980, height=980)


def sdc_probability_approach_graph_full(avf_df, profile_df, boards_list):
    # SDC PROBABILITY
    # Profile data process
    profile_df["FP32"] = profile_df[FP32_INSTRUCTIONS].sum(axis="columns")
    profile_df["GP"] = profile_df[GP_INSTRUCTIONS].sum(axis="columns")
    profile_df["LD"] = profile_df[LD_INSTRUCTIONS].sum(axis="columns")
    nvbit_profile_df = profile_df[["FP32", "GP", "LD"]]
    nvbit_profile_df = nvbit_profile_df.div(nvbit_profile_df.sum(axis="columns"), axis="rows")
    # SDC probability
    # Filter everything before continuing
    sdc_probability = (avf_df.loc[:, "SDC AVF"] * nvbit_profile_df).dropna(how="all").sum(axis="columns")
    sdc_probability = sdc_probability.rename("P(SDC)").reset_index()
    sdc_probability = sdc_probability[sdc_probability["board"].isin(boards_list)].copy()
    sdc_probability["app"] = sdc_probability["app"].apply(lambda r: REDUCED_APP_NAMES[r])
    sdc_probability["app"] = sdc_probability["app"].apply(lambda r: "CCL" if "ACCL" == r else r)
    gb = sdc_probability.drop("flag", axis="columns").groupby(["board", "app", "nvcc"])
    relative_std = (gb.agg(np.std, ddof=0) * 100) / gb.mean()
    print(relative_std.mean(), "mean")

    draw_plotly_pvf_graph(df=sdc_probability, y_name='Silent Data Corruption Probability',
                          output_image_file=f"{PARSER_DIRECTORY}/fig/sdc_probability_plotly.pdf")


def compare_with_ipdps_2021(ecc_on, cross_section):
    mxm_beam_cs = cross_section[(cross_section["app"] == "FMXM") & (cross_section["board"] == "kepler")]
    mxm_beam_cs = mxm_beam_cs.set_index(["ECC", "app", "nvcc", "flag"])
    pred = ecc_on.loc[pd.IndexSlice["kepler", "FMXM", ::]].droplevel(level=0).rename("Predicted")
    beam = mxm_beam_cs.loc["ON", "SDC"].rename("Beam")
    # Equalize both
    pred = pred[beam.index]
    pred.to_excel(f"{PARSER_DIRECTORY}/sheets/cross_section_predicted.xlsx", sheet_name="Predicted")


def ipdps_approach_graph(avf_df, profile_df, cross_section_df, micro_df, boards_list):
    # Metrics
    count_metrics_list = FP32_INSTRUCTIONS + GP_INSTRUCTIONS + LD_INSTRUCTIONS
    count_metrics = profile_df[count_metrics_list]
    count_metrics = count_metrics.div(count_metrics.sum(axis="columns"), axis="rows")
    # Calc SDC probability
    sdc_avf = avf_df.loc[:, "SDC AVF"].copy()
    for new_col in FP32_INSTRUCTIONS:
        sdc_avf[new_col] = sdc_avf["FP32"]
    for new_col in GP_INSTRUCTIONS + ["RF"]:
        sdc_avf[new_col] = sdc_avf["GP"]
    for new_col in LD_INSTRUCTIONS:
        sdc_avf[new_col] = sdc_avf["LD"]
    sdc_avf = sdc_avf.drop(["FP32", "GP", "LD"], axis="columns")

    # Calc error rate approximation
    micro_cross_section = cross_section_df[
        (cross_section_df["benchmark type"] == "ECC MICRO") |
        (cross_section_df["benchmark type"] == "Unhardened MICRO")
        ]
    sdc_micro_cross_section = micro_cross_section[["board", "app", "#SDC", "Fluency"]]
    sdc_micro_cross_section = sdc_micro_cross_section.groupby(["board", "app"]).sum()
    sdc_micro_cross_section["SDC"] = sdc_micro_cross_section["#SDC"] / sdc_micro_cross_section["Fluency"]
    sdc_micro_cross_section = sdc_micro_cross_section[["SDC"]]
    sdc_micro_cross_section = sdc_micro_cross_section.unstack().droplevel(0, axis="columns")
    sdc_micro_cross_section.columns.name = None
    sdc_micro_cross_section = sdc_micro_cross_section.fillna(0)
    sdc_micro_cross_section["INT"] = sdc_micro_cross_section[["IADD", "IMAD", "IMUL"]].max(axis="columns")

    microbenchmark_full_list = [f"{p}{m}" for p in "DFI" for m in ["ADD", "FMA", "MUL", "MAD"]]
    arithmetic_micro = sdc_micro_cross_section[sdc_micro_cross_section.columns.intersection(microbenchmark_full_list)]
    # Set micros that does not have cross-section
    sdc_micro_cross_section["MISC"] = arithmetic_micro.loc["volta"].max()
    sdc_micro_cross_section.loc["volta", "LDST"] = sdc_micro_cross_section.loc["kepler", "LDST"]

    # CrossSectionMicro * AVF * %inst
    error_prediction = sdc_micro_cross_section * sdc_avf * count_metrics
    error_prediction = error_prediction.dropna(how="all", axis="index")
    error_prediction = error_prediction.dropna(how="all", axis="columns")
    error_prediction = error_prediction.sum(axis="columns")
    # IPDPS 2021 PHI FACTOR
    micro_df = micro_df.set_index("board", append=True)
    # Volta is 0 to 100
    micro_df.loc[pd.IndexSlice[:, "volta"], "achieved_occupancy"] /= 100.0
    # All FP32 micros have the same occupancy and IPC
    performance_ffma = micro_df.loc["FFMA", ["ipc", "achieved_occupancy"]]
    performance_ffma.columns.name = None
    phi_factor = (profile_df[["achieved_occupancy", "ipc"]] / performance_ffma).dropna(how='all').prod(axis="columns")
    ecc_on_cross_sec = error_prediction * phi_factor

    tt = ecc_on_cross_sec.loc[pd.IndexSlice[:, :, '10.2', :]] / ecc_on_cross_sec.loc[pd.IndexSlice[:, :, '11.3', :]]
    tt = tt.dropna().sort_values()
    print("MEASURING SDC PRED " + "=" * 30)
    print(tt.describe())
    print(tt.mean())
    print("=" * 30)

    # Compare to ipdps 2021
    compare_with_ipdps_2021(ecc_on=ecc_on_cross_sec, cross_section=cross_section_df)

    # Normalization
    for board in ["volta", "kepler"]:
        ecc_on_cross_sec.loc[board, :] /= ecc_on_cross_sec.loc[board].max()
    ecc_on_cross_sec = ecc_on_cross_sec.rename("P(SDC)").reset_index()
    ecc_on_cross_sec["app"] = ecc_on_cross_sec["app"].apply(lambda r: REDUCED_APP_NAMES[r])
    ecc_on_cross_sec["app"] = ecc_on_cross_sec["app"].apply(lambda r: "CCL" if "ACCL" == r else r)
    draw_plotly_pvf_graph(df=ecc_on_cross_sec, y_name=r"Silent Data Corruption rate estimation",
                          output_image_file=f"{PARSER_DIRECTORY}/fig/sdc_estimation_plotly.pdf")


def avf_graph(avf_df, boards_list):
    default_opt = (
        avf_df.
        groupby(level=0, axis=1).
        sum().
        loc[pd.IndexSlice[:, :, :, "O3"]].
        sort_index(level=["board", "app"])
    )
    default_opt = default_opt.reset_index()
    default_opt["app"] = default_opt["app"].apply(lambda r: REDUCED_APP_NAMES[r])
    default_opt = default_opt.set_index(["board", "app", "nvcc"])
    kepler, volta = default_opt.loc["kepler"], default_opt.loc["volta"]
    diff = (kepler["DUE AVF"] / volta["DUE AVF"]).replace([np.inf, -np.inf], np.nan).dropna()
    print("DUE DIFF:", diff.mean())

    default_opt = default_opt.rename(columns={"DUE AVF": "DUE", "SDC AVF": "SDC"}).drop("flag", axis="columns")
    default_opt["Masked"] = 1.0 - default_opt.sum(axis="columns")
    """Plot the graph"""
    # Kepler, Volta, Ampere
    fig, ax = plt.subplots(1, len(boards_list), figsize=(13, 3), sharey="all", sharex='all', constrained_layout=False)
    width = 0.3  # the width of the bars
    half_width = width / 2
    little_space = width * 0.4
    for board_i, board in enumerate(boards_list):
        x, labels = None, None
        for (nvcc, walk, hatch, colors) in [("10.2", -half_width, None, ("RED", "BLUE", "forestgreen")),
                                            ("11.3", half_width + little_space, '////',
                                             ("lightcoral", "deepskyblue", "lightgreen"))]:
            if "ampere" == board and nvcc == "10.2":
                continue
            df_nvcc = default_opt.loc[pd.IndexSlice[board, :, nvcc]]
            df_nvcc = df_nvcc.reset_index()
            labels = df_nvcc["app"]
            x = np.arange(labels.shape[0])
            position = x if "ampere" == board else x + walk

            # print(df_nvcc)
            sdc, due, msk = df_nvcc["SDC"].values, df_nvcc["DUE"].values, df_nvcc["Masked"].values
            sdc_plus_due = sdc + due
            ax[board_i].bar(position, sdc, width, label='SDC', edgecolor="black", linewidth=1, capsize=3,
                            color=colors[0], hatch=hatch)
            ax[board_i].bar(position, due, width, label='DUE', edgecolor="black", linewidth=1, capsize=3,
                            color=colors[1], bottom=sdc, hatch=hatch)
            ax[board_i].bar(position, msk, width, label='Masked', edgecolor="black", linewidth=1, capsize=3,
                            color=colors[2], bottom=sdc_plus_due, hatch=hatch)

        ax[board_i].set_ylim((0.0, 1.0))
        ax[board_i].set_xticks(x, fontsize=14)
        ax[board_i].set_xticklabels(labels, fontsize=14)  # , rotation=90)
        ax[board_i].set_title(f"{board.capitalize()}", fontsize=14)

    ax[0].set_ylabel("Program Vulnerability Factor", fontdict={"fontsize": 14})
    plt.yticks(fontsize=14)
    custom_lines = [Patch(facecolor='BLACK', edgecolor='black', label='10.2'),
                    Patch(facecolor='LIGHTGRAY', edgecolor='black', label='11.3', hatch="////"),
                    Patch(facecolor='RED', edgecolor='black', label='SDC'),
                    Patch(facecolor='BLUE', edgecolor='black', label='DUE'),
                    Patch(facecolor='forestgreen', edgecolor='black', label='Masked')]
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)  # Need to play with this number.
    plt.legend(handles=custom_lines, bbox_to_anchor=(0.6, 1.27), edgecolor=None, frameon=False,
               ncol=len(custom_lines), fontsize=14)
    plt.savefig(f"{PARSER_DIRECTORY}/fig/avf_o3.pdf")


def main():
    avf_df = load_injection_database(FINAL_NVBITFI_DATABASE).dropna(how="all")
    profile_df = load_profile_database(FINAL_PROFILE_DATABASE).dropna(how="all").sort_index()

    assert profile_df[profile_df["achieved_occupancy"] == 0].shape[0] == 1
    # This occupancy is from: make PRECISION=float SIZE=12 boxes with STREAMS=1
    # This magic number is because the counter overflow on the profiling
    profile_df.loc[pd.IndexSlice["kepler", "FLAVA", "11.3", "O0"], "achieved_occupancy"] = 0.093750
    micro_profile_df = load_old_micro_metrics()
    boards_list = ["kepler", "volta"]  # , "ampere"]

    # Why we use SDC probability:
    # - Modeling Input-Dependent Error Propagation in Programs
    # NVBITFI inject faults selecting random faults, however, they are uniformly chosen in the instruction list
    # ONLY SDC PROBABILITY
    # - Quantifying the Accuracy of High-Level Fault Injection Techniques for Hardware Faults
    # Write SDC probability approach
    profile_df = profile_df[
        ~profile_df.index.get_level_values('app').str.contains("NW") & ~profile_df.index.get_level_values(
            'app').str.contains("ACCL")]
    avf_df = avf_df[~avf_df.index.get_level_values('app').str.contains("NW") & ~avf_df.index.get_level_values(
        'app').str.contains("ACCL")]

    sdc_probability_approach_graph_full(profile_df=profile_df, avf_df=avf_df, boards_list=boards_list)

    avf_graph(avf_df=avf_df, boards_list=boards_list)

    # IPDPS approach
    # first load the cross-sections
    kepler_df = pd.read_csv(CROSS_SECTION_KEPLER).drop("Flux", axis="columns")
    kepler_2021_df = get_fit(CROSS_SECTION_KEPLER_2021).reset_index()
    volta_df = pd.read_csv(CROSS_SECTION_VOLTA).drop("Flux", axis="columns")
    kepler_df["board"] = "kepler"
    kepler_2021_df["board"] = "kepler"
    kepler_2021_df["app"] = "FMXM"
    kepler_2021_df["benchmark type"] = "benchmark"

    volta_df["board"] = "volta"
    cross_section_df = pd.concat([kepler_df, volta_df]).dropna(how="all")

    cross_section_df["SDC"] = (cross_section_df["#SDC"] / cross_section_df["Fluence"])
    cross_section_df["DUE"] = (cross_section_df["#DUE"] / cross_section_df["Fluence"])
    cross_section_df["Err SDC"] = cross_section_df.apply(calc_err_sdc, axis="columns")
    cross_section_df["Err DUE"] = cross_section_df.apply(calc_err_due, axis="columns")
    cross_section_df["ECC"] = cross_section_df["benchmark type"].apply(lambda x: "ON" if "ECC" in x else "OFF")
    cross_section_df["nvcc"] = "10.1"
    cross_section_df["flag"] = "O3"
    cross_section_df = cross_section_df.rename(columns={"benchmark": "app", "Fluence": "Fluency"})
    cross_section_df = pd.concat([kepler_2021_df, cross_section_df]).drop(
        ["acc_time", "Flux 1h", "Time Beam Off"], axis="columns")
    ipdps_approach_graph(avf_df=avf_df, profile_df=profile_df, cross_section_df=cross_section_df,
                         micro_df=micro_profile_df, boards_list=boards_list)


if __name__ == '__main__':
    main()
