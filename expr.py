from zigzag.api import get_hardware_performance_zigzag
import math, re
import pickle
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import os
from zigzag.classes.hardware.architecture.ImcArray import ImcArray
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.accelerator import Accelerator
import json

from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.get_cacti_cost import get_w_cost_per_weight_from_cacti
from zigzag.classes.hardware.architecture.get_cacti_cost import get_cacti_cost
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
import random
import time
import matplotlib.cm as cm

def drawPieMarker(xs, ys, ratios, sizes, colors, ax=None):
    """
    This function is to plot scatter pie chart.
    The source is: https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
    @para xs: position of x axis (list or array)
    @para ys: position of y axis (list or array)
    @para ratios: pie ratio (list)
    @para sizes: pie size of each pie (list)
    @para colors: color of each section
    @para ax: plot handle
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    assert sum(ratios) <= 1, 'sum of ratios needs to be < 1'

    markers = []
    previous = 0
    # calculate the points of the pie pieces
    # generate all points shaping the pie pieces
    for color, ratio in zip(colors, ratios):
        this = 2 * np.pi * ratio + previous
        # circle shape
        x = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
        y = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
        # square shape
        # x, y = generate_array_square_shape(previous, this, 10)
        xy = np.column_stack([x, y])
        previous = this
        markers.append({'marker':xy, 's':np.abs(xy).max()**2*np.array(sizes), 'facecolor':color, 'edgecolor': 'black'})

    # scatter each of the pie pieces to create pies
    for marker in markers:
        ax.scatter(xs, ys, **marker)

def plot_m_factor_across_techs():
    """
    This function is to interpolate m factor for different technology nodes, based on data provided by ACT paper.
    m is in equation: CF/op = k1/topsw + m/(yield*lifetime*topsmm2) + constant
    """
    font_size = 15
    techs = [28, 20, 14, 10, 7, 5, 3]
    EPA = [0.9, 1.2, 1.2, 1.475, 1.52, 2.75, 2.75]
    GPA = [100, 110, 125, 150, 200, 225, 275]

    known_m = []
    for i in range(len(techs)):
        k2 = (301 * EPA[i] + GPA[i] + 500)/100  # g, CO2/mm2
        known_m.append(k2)

    # plot the curve
    plt.plot(techs, known_m, color=u'#ff7f0e', marker="o", markeredgecolor="black",
                       linestyle="--")
    for x,y in zip(techs, known_m):
        plt.text(x, y, f"({x}, {round(y,2)})", ha="left", fontsize=font_size-2)
    plt.xlabel("Technology nodes (nm)",  fontsize=font_size)
    plt.ylabel(f"Factor m (g, CO$_2$/mm$^2$)", fontsize=font_size)
    plt.grid(True)

    # linear interpolation for other techs
    x_interps = np.array([12, 16, 22])
    techs = np.array(techs)
    known_m = np.array(known_m)
    # convert techs, known_m to ascending order
    indices = np.argsort(techs)
    techs = techs[indices]
    known_m = known_m[indices]

    m_interp = np.interp(x_interps, techs, known_m)
    print(f"Tech: {x_interps}, m_interp: {m_interp}")

    plt.scatter(x_interps, m_interp, marker="s", edgecolors="black")  # interpolated values
    for x, y in zip(x_interps, m_interp):
        plt.text(x, y-0.5, f"({x}, {round(y, 2)})", ha="right", fontsize=font_size - 2)

    plt.tight_layout()
    plt.show()

def plot_performance_in_literature(data, period=4e+3, op_per_task=1):
    """
    This function is to plot carbon footprint in literature across different years, for sram-based in-memory computing accelerators.
    x axis: topsmm2
    there are in total 1 subplots
    fig1: topsw vs. topsmm2
    Marker: o: AIMC, square: DIMC.
    @para data: paper data collection
    @para period: the average activation period per task (unit: ns)
    @para op_per_task: the number of ops per inference (used to calculate the carbon per inference)
    order in data (dataframe):
    doi, year, IMC type, tech(nm), input precision, topsw (peak-macro), tops (peak-macro), topsmm2 (peak-macro), imc_size
    all metrics are normalized to 8b precision
    """

    colors = [u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22',
              u'#17becf']
    markers = ["s", "o", "^", "p"]
    font_size = 15

    # filter out points on 55, 65nm, since carbon cost cannot be calculated on these techs, due to data limitation
    new_data = []
    techs = data[:,3].astype(int).tolist()
    for i in range(len(techs)):
        if techs[i] in [55, 65]:
            continue
        else:
            point = data[i,:].tolist()
            new_data.append(point)
    data = np.array(new_data)
    # data=data[(data[:,3] != 55) & (data[:,3] != 65)]  # this line suddenly does not work for numpy, not sure why.

    years = np.unique(data[:,1]).astype(int).tolist()
    imcs = ["AIMC", "DIMC"]
    markersize=[55, 40]  # AIMC marker size and DIMC marker size
    for i in range(len(imcs)):
        for year in range(len(years)):
            # filter result
            res = []
            year_options = data[:,1].astype(int).tolist()
            imc_options = data[:,2].astype(str).tolist()
            for opi in range(len(year_options)):
                if year_options[opi] == years[year] and imc_options[opi] == imcs[i]:
                    res.append(data[opi,:])
            if len(res) == 0:
                continue
            res = np.array(res)

            # res = data[(data[:,1] == years[y]) & (data[:,2] == imcs[i])]  # this line suddenly does not work for numpy, not sure why.

            # plot topsw
            plt.scatter(res[:, 7].astype(float), res[:, 5].astype(float), marker=markers[i],
                              color=colors[i], edgecolors="black", s=markersize[i])
            # topsw_cur.set_yscale("log")
            # for x, y, tech in zip(cur_year, res[:, 5].astype(float).tolist(), res[:, 3].astype(int).tolist()):
            #     topsw_cur.text(x-0.1, y, f"{tech}", ha="right", fontsize=font_size - 7)

    # configuration
    plt.ylabel(f"TOP/s/W", fontsize=font_size)
    plt.xlabel(f"TOP/s/mm$^2$", fontsize=font_size)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid("both")

    plt.tight_layout()
    plt.show()

def plot_carbon_footprint_in_literature(data, period=4e+3, op_per_task=1):
    """
    This function is to plot carbon footprint in literature.
    fixed time: continuous-active (CA) scenario. fixed work: periodic-active (PA) scenario.
    Two figures will be plotted, where the 1st figure will show pie breakdown of carbon (fixed-time) and the 2nd
    figure will show pie breakdown of carbon (fixed-work).
    @para data: paper data collection
    @para period: the average activation period per task (unit: ns)
    @para op_per_task: the number of ops per inference (used to calculate the carbon per inference)
    """
    colors = [u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22',
              u'#17becf']
    markers = ["s", "o", "^", "p"]
    font_size = 12

    fig_rows_nbs = 1
    fig_cols_nbs = 2
    fig, axs = plt.subplots(nrows=fig_rows_nbs, ncols=fig_cols_nbs, figsize=(10, 6))
    pie_cf_ft_cur = axs[0]  # CF/op pie (fixed-time)
    pie_cf_fw_cur = axs[1]  # CF/op pie (fixed-time)

    # filter out points on 55, 65nm, since carbon cost cannot be calculated on these techs, due to data limitation
    new_data = []
    techs = data[:, 3].astype(int).tolist()
    for i in range(len(techs)):
        if techs[i] in [55, 65]:
            continue
        else:
            point = data[i, :].tolist()
            new_data.append(point)
    data = np.array(new_data)

    years = np.unique(data[:, 1]).astype(int).tolist()
    paper_dios = data[:, 0].astype(str).tolist()
    imcs = ["AIMC", "DIMC"]
    for i in range(len(imcs)):
        # for year_idx in range(len(years)):
        for paper_dio in range(len(paper_dios)):
            # filter result
            res = []
            year_options = data[:, 1].astype(int).tolist()
            imc_options = data[:, 2].astype(str).tolist()
            res.append(data[paper_dio, :])
            if res[0][2] != imcs[i]:
                continue
            # for opi in range(len(year_options)):
                # if year_options[opi] == years[year_idx] and imc_options[opi] == imcs[i]:
                #     res.append(data[opi, :])
            # if len(res) == 0:
            #     continue
            res = np.array(res)
            paper_cit_idx = res[:, 12].astype(int).tolist()[0]
            # calc carbon cost
            topsws = res[:, 5].astype(float).tolist()
            topss = res[:, 6].astype(float).tolist()
            topsmm2s = res[:, 7].astype(float).tolist()
            paralls = res[:, 8].astype(float).tolist()  # ops/cycle
            techs = res[:, 3].astype(int).tolist()
            paper_idx = res[:, 0].astype(str).tolist()[0]
            print(f"paper doi: {paper_idx}")
            ci_fabs = {  # tech | CI_fab [g, CO2/mm2]
                28: 8.709,
                22: 9.446,
                20: 9.712,
                16: 9.812,
                14: 9.862,
                12: 10.408,
                10: 10.955,
                7: 11.575,
                5: 15.528,
                3: 16.028,
            }
            # calc cf (fixed-time)
            cf_fts = []
            cf_fws = []
            pie_cf_fts = []
            pie_cf_fws = []
            chip_yield = 0.95  # yield
            lifetime = 3  # year
            # period = 4e+6  # 4us
            k1 = 301 / (3.6E+18)  # g, CO2/pJ
            for idx in range(len(topsws)):
                topsw = topsws[idx]
                tops = topss[idx]
                topsmm2 = topsmm2s[idx]
                tech = techs[idx]
                operational_carbon = k1 / topsw  # g, CO2/op
                # Convert unit to g, CO2/task
                operational_carbon *= op_per_task  # g, CO2/task

                ci_fab = ci_fabs[tech]
                k2 = ci_fab / chip_yield / (lifetime * 365 * 24 * 60 * 60E+12)  # g, CO2/mm2/ps
                parallel = paralls[idx]
                embodied_carbon_ft_per_op = k2 / topsmm2  # g, CO2/op
                embodied_carbon_fw_per_op = k2 * (period * 1000) * tops / topsmm2 / parallel  # period: unit: ns -> ps
                # Convert unit to g, CO2/inference
                embodied_carbon_ft = embodied_carbon_ft_per_op * op_per_task
                embodied_carbon_fw = embodied_carbon_fw_per_op * parallel

                cf_ft = operational_carbon + embodied_carbon_ft  # unit: g, carbon/task
                cf_fw = operational_carbon + embodied_carbon_fw  # unit: g, carbon/task
                cf_fts.append(cf_ft)
                cf_fws.append(cf_fw)
                pie_cf_fts.append([operational_carbon / cf_ft, 1 - operational_carbon / cf_ft])
                pie_cf_fws.append([operational_carbon / cf_fw, 1 - operational_carbon / cf_fw])
            cf_fts = np.array(cf_fts)
            cf_fws = np.array(cf_fws)

            # plot pie cf (fixed-time)
            pie_size = 200
            pie_colors = colors[1:3]
            for x, y, ratio in zip(topsws, cf_fts, pie_cf_fts):
                drawPieMarker(xs=[x],
                              ys=[y],
                              ratios=ratio,
                              sizes=[pie_size],
                              colors=pie_colors,
                              ax=pie_cf_ft_cur)
            pie_cf_ft_cur.set_xscale("log")
            pie_cf_ft_cur.set_yscale("log")
            for x, y, tech in zip(topsws, cf_fts, res[:, 3].astype(int).tolist()):
                pie_cf_ft_cur.text(x, 1.1*y, f"[{paper_cit_idx}] {tech} nm", ha="center", fontsize=font_size)

            # plot pie cf (fixed-work)
            pie_size = 200
            for x, y, ratio in zip(topsws, cf_fws, pie_cf_fws):
                drawPieMarker(xs=[x],
                              ys=[y],
                              ratios=ratio,
                              sizes=[pie_size],
                              colors=pie_colors,
                              ax=pie_cf_fw_cur)
            pie_cf_fw_cur.set_xscale("log")
            pie_cf_fw_cur.set_yscale("log")
            for x, y, tech in zip(topsws, cf_fws, res[:, 3].astype(int).tolist()):
                pie_cf_fw_cur.text(x, 1.1*y, f"[{paper_cit_idx}] {tech} nm", ha="center", fontsize=font_size)

    # configuration
    for i in range(0, fig_rows_nbs):
        for j in range(0, fig_cols_nbs):
            # axs[j].set_xlabel(f"g, CO$_2$/inference (fixed-work)", fontsize=font_size+5)
            axs[j].set_xlabel(f"TOP/s/W", fontsize=font_size + 5)
            axs[j].set_ylim([10**-10, 10**-6])
            # axs[j].set_xlim([10 ** -6, 2*10 ** -3])
            axs[j].grid("both")
            axs[j].set_axisbelow(True)
            # axs[i][j].legend(loc="upper left")
        axs[0].set_ylabel(f"g, CO$_2$$_e$$_q$/inference (CA)", fontsize=font_size+5)
        axs[1].set_ylabel(f"g, CO$_2$$_e$$_q$", fontsize=font_size + 5)

    # legend description
    # green: operational carbon footprint
    # orange: embodied carbon footprint

    plt.tight_layout()
    plt.show()

def plot_curve(i_df, acc_types, workload, sram_size=256*1024, d1_equal_d2=True):
    # This function is to plot topsw, tops, topsmm2, carbon footprint in curve, for both AIMC and DIMC, under
    # a fixed workload and sram size.
    colors = [u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22',
              u'#17becf']
    markers = ["s", "o", "^", "p"]
    width = 0.35
    font_size = 15
    df = i_df[(i_df.workload == workload) & (i_df.sram_size == sram_size)]
    if d1_equal_d2:
        dim_d1 = df.dim
    else:
        dim_d1 = df.dim // 8
    df["imc"] = dim_d1.astype(str) + f"$\\times$" + df.dim.astype(str)

    fig_rows_nbs = 2
    fig_cols_nbs = 3
    fig, axs = plt.subplots(nrows=fig_rows_nbs, ncols=fig_cols_nbs, figsize=(10, 8))
    topsw_cur = axs[0, 0]
    tops_cur = axs[0, 1]
    topsmm2_cur = axs[0, 2]
    cf_ft_cur = axs[1, 0]  # CF/inference
    cf_fw_cur = axs[1, 1]  # CF/inference
    for ii_a, a in enumerate(acc_types):
        dff = df[df.acc_type == a]
        labels = dff.imc.to_numpy()  # x label
        # Create positions for the bars on the x-axis
        x_pos = np.arange(len(labels))

        # Plot the curve
        # Plot topsw
        topsw = dff.topsw.to_numpy()
        topsw_cur.plot(x_pos, topsw, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black",
                       linestyle="--")
        # Plot tops
        tops = dff.tops.to_numpy()
        tops_cur.plot(x_pos, tops, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black",
                      linestyle="--")
        # Plot topsmm2
        topsmm2 = dff.topsmm2.to_numpy()
        topsmm2_cur.plot(x_pos, topsmm2, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black",
                         linestyle="--")
        # Plot cf (fixed-time)
        cf_ft = dff.t_cf_ft_ex_pkg.to_numpy()
        cf_ft_cur.plot(x_pos, cf_ft, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black",
                    linestyle="--")

        # Plot cf (fixed-work)
        cf_fw = dff.t_cf_fw_ex_pkg.to_numpy()
        cf_fw_cur.plot(x_pos, cf_fw, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black",
                       linestyle="--")

    # configuration
    for i in range(0, fig_rows_nbs):
        for j in range(0, fig_cols_nbs):
            axs[i][j].set_xticks(x_pos, labels, rotation=45)
            axs[i][j].legend(loc="upper left")
    topsw_cur.set_ylabel(f"TOP/s/W ({workload})", fontsize=font_size)
    tops_cur.set_ylabel(f"TOP/s ({workload})", fontsize=font_size)
    topsmm2_cur.set_ylabel(f"TOP/s/mm$^2$ ({workload})", fontsize=font_size)
    cf_ft_cur.set_ylabel(f"g, CO$_2$/Inference (fixed-time)", fontsize=font_size)
    cf_fw_cur.set_ylabel(f"g, CO$_2$/Inference (fixed-work)", fontsize=font_size)

    # if workload == "geo":
    #     cf_ft_cur.set_ylim(0, 4e-9)  # manually set the range
    #     cf_fw_cur.set_ylim(0, 4e-9)  # manually set the range

    # plt.text(0.5, -0.1, f"IMC size (#rows $\\times$ #cols)", ha="center", va="center", fontsize=20)  # does not work well
    plt.tight_layout()
    plt.show()

def calc_carbon_footprint_for_complex_task(raw_data, acc_type, workload, sram_size, complexity):
    # function to calculate carbon for complex tasks under the fixed-work scenario
    def calc_carbon_footprint_for_complex_task_per_workload(raw_data, acc_type, workload, sram_size, complexity):
        assert workload not in ["peak", "geo"], f"Workload {workload} is not supported."
        assert complexity >= 1, f"Complexity [{complexity}] < 1 is not permitted."
        # (1) clip the raw data
        entire_df = raw_data["data"]
        periods = raw_data["periods"]
        period = periods[workload]
        try:
            dim = raw_data["dim"]
        except KeyError:
            dim = None
        if isinstance(sram_size, list) and len(sram_size) > 0:
            use_df = entire_df[(entire_df.workload == workload) & (entire_df.dim == dim) & (
                    entire_df.acc_type == acc_type)]
        else:
            use_df = entire_df[(entire_df.workload == workload) & (entire_df.sram_size == sram_size) & (
                    entire_df.acc_type == acc_type)]
        # (2) collect original carbon breakdown for the workload
        cf_ft_breakdown = use_df.cf_ft.to_numpy()
        cf_fw_breakdown = use_df.cf_fw.to_numpy()
        operational_cf = np.array([case["opcf"] for case in cf_fw_breakdown])
        inference_time = np.array([case["runtime"] for case in cf_ft_breakdown])
        soc_epa = np.array([case["soc_epa"] for case in cf_fw_breakdown])
        soc_gpa = np.array([case["soc_gpa"] for case in cf_fw_breakdown])
        soc_mpa = np.array([case["soc_mpa"] for case in cf_fw_breakdown])
        dram_cost = np.array([case["dram"] for case in cf_fw_breakdown])
        embodied_cf = soc_epa + soc_gpa + soc_mpa + dram_cost
        # (3) check if the complexity exceeds the limit
        max_complexity_scaling = period / max(inference_time)
        assert complexity <= max_complexity_scaling, \
            f"Current complexity level {complexity} exceed the maximum allowed level {max_complexity_scaling} for workload {workload}"
        # (4) convert to the new complex case
        complex_operational_cf = operational_cf * complexity
        complex_embodied_cf = embodied_cf
        cf_fw_complex = complex_operational_cf + complex_embodied_cf
        return cf_fw_complex

    if workload != "geo":
        cf_fw_complex = calc_carbon_footprint_for_complex_task_per_workload(raw_data, acc_type, workload, sram_size, complexity)
    else:
        cf_fw_complex_list = []
        workload_counts = 0
        for workload in raw_data["workloads"]:
            if workload in ["peak", "geo"]:
                continue
            workload_counts += 1
            cf_fw_complex_list.append(calc_carbon_footprint_for_complex_task_per_workload(raw_data, acc_type, workload, sram_size, complexity))
        for i in range(len(cf_fw_complex_list)):
            if i == 0:
                geo_cf_fw_complex = cf_fw_complex_list[i]
            else:
                geo_cf_fw_complex = geo_cf_fw_complex * cf_fw_complex_list[i]
        cf_fw_complex = geo_cf_fw_complex ** (1/workload_counts)
    return cf_fw_complex

def plot_total_carbon_curve_four_cases(i_df, acc_types, workload, sram_size, complexity=50, raw_data={}, plot_breakdown=False, d1_equal_d2=True):
    # This function is to plot total carbon cost (in curve) for pdigital, aimc and dimc, under 4 cases:
    # (1) fixed-time, simple task
    # (2) fixed-work, simple task
    # (3) fixed-time, complex task with 10x (default) more complexity than case (1)
    # (4) fixed-work, complex task with 10x (default) complexity than case (2)
    # The output figure consists of 4 subplots (x-axis: area, y-axis: carbon/inference), corresponding to the cases above.
    # @para sram_size: value [x axis is dim size]; list [x axis is sram size].

    # colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
    #           u'#bcbd22', u'#17becf']
    colors = [u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
              u'#bcbd22', u'#17becf']
    markers = ["s", "o", "^", "p"]
    font_size = 18
    marker_size = 6
    line_width = 1
    df = i_df[(i_df.workload == workload)]
    if d1_equal_d2:
        dim_d1 = df.dim
    else:
        dim_d1 = df.dim // 8
    df["imc"] = dim_d1.astype(str) + f"$\\times$" + df.dim.astype(str)  # D1 x D2

    fig_rows_nbs = 2
    fig_cols_nbs = 2
    fig, axs = plt.subplots(nrows=fig_rows_nbs, ncols=fig_cols_nbs, figsize=(10, 8))
    cf_ft_simple_cur = axs[0][0]  # CF/inference
    cf_fw_simple_cur = axs[0][1]  # CF/inference
    cf_ft_complex_cur = axs[1][0]  # CF/inference
    cf_fw_complex_cur = axs[1][1]  # CF/inference

    for ii_a, a in enumerate(acc_types):
        dff = df[df.acc_type == a]
        # Create positions for the bars on the x-axis
        x_pos = dff.t_area.to_numpy()
        # Due to cacti, the peripheral mem area for 64x64 imc array is smaller than 32x32 imc array, this looks weird
        # Use following code to fix it.
        if x_pos[1] < x_pos[0]:
            mem_area = dff.iloc[0].area["mem_area"]
            updt_area = dff.iloc[1].area["imc_area"] + mem_area
            x_pos = np.array([x_pos[0], updt_area] + x_pos[2:].tolist())

        if not plot_breakdown:
            # Plot cf, simple (fixed-time)
            cf_ft = dff.t_cf_ft_ex_pkg.to_numpy()
            cf_ft_simple_cur.plot(x_pos, cf_ft, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black",
                           linestyle="--", markersize=marker_size, linewidth=line_width)
            # cf_ft_simple_cur.scatter(x_pos, cf_ft, label=a, color=colors[ii_a], marker=markers[ii_a], edgecolors="black",
            #                          s=15)

            ## Text for dimension sizes on the plot
            # if ii_a in [0, 1, 2]:
            #     if isinstance(sram_size, list) and len(sram_size) > 0:
            #         attached_text = dff.sram_size.to_numpy()
            #         attached_text = [f"{unit_text//1024}KB" for unit_text in attached_text]
            #     else:
            #         attached_text = dff.imc
            #     for x, y, txt in zip(x_pos, cf_ft, attached_text):
            #         cf_ft_simple_cur.text(x, y, f"({txt})", ha="center", fontsize=8)

            # Plot cf, simple (fixed-work)
            cf_fw = dff.t_cf_fw_ex_pkg.to_numpy()
            cf_fw_simple_cur.plot(x_pos, cf_fw, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black",
                           linestyle="--", markersize=marker_size, linewidth=line_width)

            ## Text for dimension sizes on the plot
            # if ii_a in [2]:
            #     if isinstance(sram_size, list) and len(sram_size) > 0:
            #         attached_text = dff.sram_size.to_numpy()
            #         attached_text = [f"{unit_text // 1024}KB" for unit_text in attached_text]
            #     else:
            #         attached_text = dff.imc
            #     for x, y, txt in zip(x_pos, cf_fw, attached_text):
            #         cf_fw_simple_cur.text(x, y, f"({txt})", ha="center", fontsize=8)

            # Plot cf, complex (fixed-time)
            cf_ft_complex = cf_ft * complexity
            cf_ft_complex_cur.plot(x_pos, cf_ft_complex, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black",
                           linestyle="--", markersize=marker_size, linewidth=line_width)

            # Plot cf, complext (fixed-work)
            cf_fw_complex = calc_carbon_footprint_for_complex_task(raw_data=raw_data,
                                                                  acc_type=a,
                                                                  workload=workload,
                                                                  sram_size=sram_size,
                                                                  complexity=complexity)
            cf_fw_complex_cur.plot(x_pos, cf_fw_complex, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black",
                                   linestyle="--", markersize=marker_size, linewidth=line_width)
        else:
            assert workload != "geo", f"geo workload does not have carbon breakdown."
            # Plot cf, simple (fixed-time)
            cf_ft_bd = dff.cf_ft.to_numpy()
            cf_ft_operational = []
            cf_ft_embodied = []
            for i in range(len(cf_ft_bd)):
                operational_carbon = cf_ft_bd[i]["opcf"]
                embodied_carbon = cf_ft_bd[i]["soc_epa"] + cf_ft_bd[i]["soc_gpa"] + cf_ft_bd[i]["soc_mpa"]
                cf_ft_operational.append(operational_carbon)
                cf_ft_embodied.append(embodied_carbon)
            cf_ft_operational = np.array(cf_ft_operational)
            cf_ft_embodied = np.array(cf_ft_embodied)
            cf_ft_simple_cur.plot(x_pos, cf_ft_operational, label=f"{a}-opcf", color=colors[ii_a], marker=markers[ii_a],
                                markeredgecolor="black",
                                markerfacecolor=colors[ii_a],
                                linestyle="--", markersize=marker_size, linewidth=line_width)
            cf_ft_simple_cur.plot(x_pos, cf_ft_embodied, label=f"{a}-edcf", color=colors[ii_a], marker=markers[ii_a],
                                markeredgecolor="black",
                                markerfacecolor=colors[ii_a],
                                linestyle=":", markersize=marker_size, linewidth=line_width)
            # Plot cf, simple (fixed-work)
            cf_fw_bd = dff.cf_fw.to_numpy()
            cf_fw_operational = []
            cf_fw_embodied = []
            for i in range(len(cf_fw_bd)):
                operational_carbon = cf_fw_bd[i]["opcf"]
                embodied_carbon = cf_fw_bd[i]["soc_epa"] + cf_fw_bd[i]["soc_gpa"] + cf_fw_bd[i]["soc_mpa"]
                cf_fw_operational.append(operational_carbon)
                cf_fw_embodied.append(embodied_carbon)
            cf_fw_operational = np.array(cf_fw_operational)
            cf_fw_embodied = np.array(cf_fw_embodied)
            cf_fw_simple_cur.plot(x_pos, cf_fw_operational, label=f"{a}-opcf", color=colors[ii_a], marker=markers[ii_a],
                                markeredgecolor="black",
                                markerfacecolor=colors[ii_a],
                                linestyle="--", markersize=marker_size, linewidth=line_width)
            cf_fw_simple_cur.plot(x_pos, cf_fw_embodied, label=f"{a}-edcf", color=colors[ii_a], marker=markers[ii_a],
                                markeredgecolor="black",
                                markerfacecolor=colors[ii_a],
                                linestyle=":", markersize=marker_size, linewidth=line_width)
            # Plot cf, complex (fixed-time)
            cf_ft_complex_cur.plot(x_pos, complexity*cf_ft_operational, label=f"{a}-opcf", color=colors[ii_a], marker=markers[ii_a],
                                  markeredgecolor="black",
                                  markerfacecolor=colors[ii_a],
                                  linestyle="--", markersize=marker_size, linewidth=line_width)
            cf_ft_complex_cur.plot(x_pos, complexity*cf_ft_embodied, label=f"{a}-edcf", color=colors[ii_a], marker=markers[ii_a],
                                  markeredgecolor="black",
                                  markerfacecolor=colors[ii_a],
                                  linestyle=":", markersize=marker_size, linewidth=line_width)
            # Plot cf, complext (fixed-work)
            cf_fw_complex_cur.plot(x_pos, complexity*cf_fw_operational, label=f"{a}-opcf", color=colors[ii_a], marker=markers[ii_a],
                                  markeredgecolor="black",
                                  markerfacecolor=colors[ii_a],
                                  linestyle="--", markersize=marker_size, linewidth=line_width)
            cf_fw_complex_cur.plot(x_pos, cf_fw_embodied, label=f"{a}-edcf", color=colors[ii_a], marker=markers[ii_a],
                                  markeredgecolor="black",
                                  markerfacecolor=colors[ii_a],
                                  linestyle=":", markersize=marker_size, linewidth=line_width)
    # configuration
    for i in range(0, fig_rows_nbs):
        for j in range(0, fig_cols_nbs):
            axs[i][j].set_xscale("log")
            axs[i][j].set_yscale("log")
            axs[i][j].grid(which="both")
            axs[i][j].set_axisbelow(True)
            axs[i][j].legend(loc="upper left")
    axs[fig_rows_nbs - 1][0].set_xlabel(f"Area [mm$^2$]")
    axs[fig_rows_nbs - 1][1].set_xlabel(f"Area [mm$^2$]")
    axs[1][0].set_ylabel(f"\t\t\t\t\t\t\tg, CO$_2$/Inference (fixed-time)", fontsize=font_size)
    axs[1][1].set_ylabel(f"\t\t\t\t\t\t\tg, CO$_2$/Inference (fixed-work)", fontsize=font_size)
    axs[0][0].set_title(f"Simple task [{workload}]")
    axs[0][1].set_title(f"Simple task [{workload}]")
    axs[1][0].set_title(f"Complex task [{complexity}x {workload}]")
    axs[1][1].set_title(f"Complex task [{complexity}x {workload}]")

    plt.tight_layout()
    plt.show()

def plot_performance_bar(i_df, acc_types: list, workload: str, sram_size=256*1024, d1_equal_d2=True, breakdown=True):
    # This function is to plot performance for pdigital_os, pdigital_ws, aimc and dimc, under fixed sram size.
    # @para sram_size: value [x axis is D1xD2]; list [x axis is sram size]
    # @para d1_equal_d2: True [D1=D2=dim]; False [D1=dim//8, D2=dim]
    # @para breakdown: True [show cost breakdown for PE (colored bar) and memories (white bar)]; False [only show total cost]

    assert workload != "peak", "Plotting for peak is not supported so far."

    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
              u'#bcbd22', u'#17becf']
    markers = ["s", "o", "^", "p"]
    width = 0.15
    font_size = 15
    df = i_df[(i_df.workload == workload)]
    if d1_equal_d2:
        dim_d1 = df.dim
    else:
        dim_d1 = df.dim // 8
    df["imc"] = dim_d1.astype(str) + f"$\\times$" + df.dim.astype(str)

    fig_rows_nbs = 2
    fig_cols_nbs = 3
    fig, axs = plt.subplots(nrows=fig_rows_nbs, ncols=fig_cols_nbs, figsize=(10, 8))
    energy_bar = axs[0][0]  # energy (pJ)/inference
    latency_bar = axs[0][1]  # latency (cycles)/inference
    tclk_bar = axs[1][0]  # tclk (ns)
    area_bar = axs[1][1]  # area (mm2)
    time_bar = axs[0][2]  # latency (ns)/inference
    log_area_bar = axs[1][2]  # log(area)



    for ii_a, acc_type in enumerate(acc_types):
        if ii_a == 0:
            ii_pos = -3
        elif ii_a == 1:
            ii_pos = -1
        elif ii_a == 2:
            ii_pos = 1
        else:
            ii_pos = 3

        dff = df[df.acc_type == acc_type]

        # identify x axis
        if isinstance(sram_size, list) and len(sram_size) > 1:
            labels = dff.sram_size.to_numpy()
            # convert unit to KB
            labels = labels//1024
            labels = [f"{size}KB" for size in labels]
        else:
            labels = dff.imc.to_numpy()
        # Create positions for the bars on the x-axis
        x_pos = np.arange(len(labels))

        # Plot the bars
        if breakdown:
            assert workload != "geo", "geo does not have cost breakdown."
            # Plot energy
            base = [0 for x in range(0, len(labels))]
            val = np.array([x[0][0].MAC_energy for x in dff.cme])
            energy_bar.bar(x_pos + width / 2 * ii_pos, val, width, label=acc_type, bottom=base, color=colors[ii_a],
                      edgecolor="black")
            base = base + val
            val = np.array([x[0][0].mem_energy for x in dff.cme])
            energy_bar.bar(x_pos + width / 2 * ii_pos, val, width, label=None, bottom=base, color="white",
                           edgecolor="black")

            # Plot tclk
            t_tclk = dff.t_tclk  # ns
            tclk_bar.bar(x_pos + width / 2 * ii_pos, t_tclk, width, label=acc_type, color=colors[ii_a],
                         edgecolor="black")

            # Plot latency
            base = [0 for x in range(0, len(labels))]
            val = np.array([x[0][0].ideal_temporal_cycle for x in dff.cme])
            latency_bar.bar(x_pos + width / 2 * ii_pos, val, width, label=acc_type, bottom=base, color=colors[ii_a],
                           edgecolor="black")
            base = base + val
            latency_total0 = np.array([x[0][0].latency_total0 for x in dff.cme])
            val = latency_total0 - val  # stall cycles
            latency_bar.bar(x_pos + width / 2 * ii_pos, val, width, label=None, bottom=base, color="white",
                            edgecolor="black")
            # Plot area
            t_area = dff.t_area.to_numpy()  # series: mm2
            mem_area = np.array([x["mem_area"] for x in dff.area])
            pe_area = t_area - mem_area
            base = [0 for x in range(0, len(labels))]
            val = pe_area
            area_bar.bar(x_pos + width / 2 * ii_pos, val, width, label=acc_type, bottom=base, color=colors[ii_a],
                            edgecolor="black")
            log_area_bar.bar(x_pos + width / 2 * ii_pos, val, width, label=acc_type, bottom=base, color=colors[ii_a],
                         edgecolor="black")
            base = base + val
            val = mem_area
            area_bar.bar(x_pos + width / 2 * ii_pos, val, width, label=None, bottom=base, color="white",
                            edgecolor="black")

            log_area_bar.bar(x_pos + width / 2 * ii_pos, val, width, label=None, bottom=base, color="white",
                            edgecolor="black")
        else:
            # Plot energy
            t_en = dff.t_en  # series: pJ
            energy_bar.bar(x_pos + width / 2 * ii_pos, t_en, width, label=acc_type, color=colors[ii_a],
                      edgecolor="black")

            # Plot tclk
            t_tclk = dff.t_tclk  # ns
            tclk_bar.bar(x_pos + width / 2 * ii_pos, t_tclk, width, label=acc_type, color=colors[ii_a],
                           edgecolor="black")

            # Plot latency
            t_lat = dff.t_lat  # total latency (unit: cycle counts)
            # t_lat *= t_tclk  # unit: ns/inference
            latency_bar.bar(x_pos + width / 2 * ii_pos, t_lat, width, label=acc_type, color=colors[ii_a],
                         edgecolor="black")
            latency_bar.set_yscale("log")

            # Plot area
            t_area = dff.t_area  # series: mm2
            area_bar.bar(x_pos + width / 2 * ii_pos, t_area, width, label=acc_type, color=colors[ii_a],
                           edgecolor="black")
            log_area_bar.bar(x_pos + width / 2 * ii_pos, t_area, width, label=acc_type, color=colors[ii_a],
                         edgecolor="black")

        # Plot inference time
        t_lat = dff.t_lat  # total latency (unit: cycle counts)
        t_lat *= t_tclk  # unit: ns/inference
        time_bar.bar(x_pos + width / 2 * ii_pos, t_lat, width, label=acc_type, color=colors[ii_a],
                        edgecolor="black")
        time_bar.set_yscale("log")

    # configuration
    for i in range(0, fig_rows_nbs):
        for j in range(0, fig_cols_nbs):
            axs[i][j].set_xticks(x_pos, labels, rotation=45)
            axs[0][j].legend(loc="upper right")
            axs[1][j].legend(loc="upper left")
            axs[i][j].grid(which="both")
            axs[i][j].set_axisbelow(True)
    energy_bar.set_ylabel(f"Energy [pJ/inference] ({workload})", fontsize=font_size)
    latency_bar.set_ylabel(f"Latency [cycles/inference] ({workload})", fontsize=font_size)
    tclk_bar.set_ylabel(f"Tclk [ns] ({workload})", fontsize=font_size)
    area_bar.set_ylabel(f"Area [mm$^2$] ({workload})", fontsize=font_size)
    time_bar.set_ylabel(f"Inference time [ns] ({workload})", fontsize=font_size)
    log_area_bar.set_ylabel(f"Area [log(mm$^2$)] ({workload})", fontsize=font_size)
    log_area_bar.set_yscale("log")

    plt.tight_layout()
    plt.show()

def plot_carbon_breakdown_in_curve(i_df, acc_types, workload, sram_size=256*1024, d1_equal_d2=True):
    # This function is to plot carbon cost breakdown (in curve) for pdigital, aimc and dimc, under fixed workload and fixed sram size.
    # The output figure consists of 2 subplots (x-axis: area):
    # 1th (left): carbon footprint breakdown (fixed-time)
    # 2th (right): carbon footprint breakdown (fixed-work)

    assert workload != "geo", "geo does not have carbon breakdown, so the breakdown plots cannot be generated."

    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
              u'#bcbd22', u'#17becf']
    markers = ["s", "o", "^", "p"]
    font_size = 15
    df = i_df[(i_df.workload == workload) & (i_df.sram_size == sram_size)]
    if d1_equal_d2:
        dim_d1 = df.dim
    else:
        dim_d1 = df.dim // 8
    df["imc"] = dim_d1.astype(str) + f"$\\times$" + df.dim.astype(str)  # D1 x D2

    fig_rows_nbs = 2
    fig_cols_nbs = 2
    fig, axs = plt.subplots(nrows=fig_rows_nbs, ncols=fig_cols_nbs, figsize=(10, 8))
    cf_ft_cur = axs[0][0]  # CF/inference
    cf_fw_cur = axs[0][1]  # CF/inference
    opcf_ft_bd_cur = axs[1][0]
    emcf_ft_bd_cur = axs[1][0]
    opcf_fw_bd_cur = axs[1][1]
    emcf_fw_bd_cur = axs[1][1]

    for ii_a, a in enumerate(acc_types):
        dff = df[df.acc_type == a]
        # Create positions for the bars on the x-axis
        x_pos = dff.t_area.to_numpy()
        # Due to cacti, the peripheral mem area for 64x64 imc array is smaller than 32x32 imc array, this looks weird
        # Use following code to fix it.
        if x_pos[1] < x_pos[0]:
            mem_area = dff.iloc[0].area["mem_area"]
            updt_area = dff.iloc[1].area["imc_area"] + mem_area
            x_pos = np.array([x_pos[0], updt_area] + x_pos[2:].tolist())

        # Plot cf (fixed-time)
        cf_ft = dff.t_cf_ft_ex_pkg.to_numpy()
        cf_ft_cur.plot(x_pos, cf_ft, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black",
                       linestyle="--")

        if ii_a in [0, 1, 2]:
            for x, y, txt in zip(x_pos, cf_ft, dff.imc):
                cf_ft_cur.text(x, y, f"({txt})", ha="center", fontsize=8)

        # Plot cf (fixed-work)
        cf_fw = dff.t_cf_fw_ex_pkg.to_numpy()
        cf_fw_cur.plot(x_pos, cf_fw, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black",
                       linestyle="--")

        # Plot cf breakdown (fixed-time)
        cf_ft_bd = dff.cf_ft.to_numpy()
        cf_ft_operational = []
        cf_ft_embodied = []
        cf_ft_dram = []
        for i in range(len(cf_ft_bd)):
            operational_carbon = cf_ft_bd[i]["opcf"]
            embodied_carbon = cf_ft_bd[i]["soc_epa"] + cf_ft_bd[i]["soc_gpa"] + cf_ft_bd[i]["soc_mpa"]
            dram_carbon = cf_ft_bd[i]["dram"]
            cf_ft_operational.append(operational_carbon)
            cf_ft_embodied.append(embodied_carbon)
            cf_ft_dram.append(dram_carbon)
        cf_ft_operational = np.array(cf_ft_operational)
        cf_ft_embodied = np.array(cf_ft_embodied)
        cf_ft_dram = np.array(cf_ft_dram)
        opcf_ft_bd_cur.plot(x_pos, cf_ft_operational, label=f"{a}-opcf", color=colors[ii_a], marker=markers[ii_a],
                       markeredgecolor="black",
                       markerfacecolor=colors[ii_a],
                       linestyle="--")
        emcf_ft_bd_cur.plot(x_pos, cf_ft_embodied, label=f"{a}-edcf", color=colors[ii_a], marker=markers[ii_a],
                       markeredgecolor="black",
                       markerfacecolor=colors[ii_a],
                       linestyle="--")
        # cf_ft_cur.plot(x_pos, cf_ft_dram, label="dramcf", color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black",
        #                linestyle="--")

        # Plot cf breakdown (fixed-work)
        cf_fw_bd = dff.cf_fw.to_numpy()
        cf_fw_operational = []
        cf_fw_embodied = []
        cf_fw_dram = []
        for i in range(len(cf_fw_bd)):
            operational_carbon = cf_fw_bd[i]["opcf"]
            embodied_carbon = cf_fw_bd[i]["soc_epa"] + cf_fw_bd[i]["soc_gpa"] + cf_fw_bd[i]["soc_mpa"]
            dram_carbon = cf_fw_bd[i]["dram"]
            cf_fw_operational.append(operational_carbon)
            cf_fw_embodied.append(embodied_carbon)
            cf_fw_dram.append(dram_carbon)
        cf_fw_operational = np.array(cf_fw_operational)
        cf_fw_embodied = np.array(cf_fw_embodied)
        cf_fw_dram = np.array(cf_fw_dram)
        opcf_fw_bd_cur.plot(x_pos, cf_fw_operational, label=f"{a}-opcf", color=colors[ii_a], marker=markers[ii_a],
                       markeredgecolor="black",
                       markerfacecolor=colors[ii_a],
                       linestyle="--")
        emcf_fw_bd_cur.plot(x_pos, cf_fw_embodied, label=f"{a}-edcf", color=colors[ii_a], marker=markers[ii_a],
                       markeredgecolor="black",
                       markerfacecolor=colors[ii_a],
                       linestyle="--")
        # cf_fw_cur.plot(x_pos, cf_fw_dram, label="dramcf", color=colors[ii_a], marker=markers[ii_a],
        #                markeredgecolor="black",
        #                linestyle="--")

        # Plot latency
        # lat = dff.t_lat.to_numpy() * dff.t_tclk.to_numpy()
        # axs[2][0].plot(x_pos, lat, label=a, color=colors[ii_a], marker=markers[ii_a],
        #                markeredgecolor="black",
        #                linestyle="--")
        # axs[2][1].plot(x_pos, [period]*len(x_pos), label=a, color=colors[ii_a], marker=markers[ii_a],
        #                markeredgecolor="black",
        #                linestyle="--")

    # configuration
    for i in range(0, fig_rows_nbs):
        for j in range(0, fig_cols_nbs):
            axs[i][j].set_xscale("log")
            axs[i][j].set_yscale("log")
            axs[i][j].grid(which="both")
            axs[i][j].set_axisbelow(True)
            axs[i][j].legend(loc="upper right")
    axs[fig_rows_nbs-1][0].set_xlabel(f"Area (mm$^2$)")
    axs[fig_rows_nbs-1][1].set_xlabel(f"Area (mm$^2$)")
    axs[1][0].set_ylabel(f"g, CO$_2$/Inference (fixed-time)", fontsize=font_size)
    axs[1][1].set_ylabel(f"g, CO$_2$/Inference (fixed-work)", fontsize=font_size)
    # axs[2][0].set_ylabel(f"ns/Inference", fontsize=font_size)
    # axs[2][1].set_ylabel(f"ns/Inference", fontsize=font_size)
    axs[0][0].set_title(f"Total [{workload}]")
    axs[1][0].set_title("Carbon breakdown")
    # axs[2][0].set_title("Inference speed")
    # axs[2][0].set_title("Carbon breakdown")

    # if workload == "geo":
    #     cf_ft_cur.set_ylim(0, 4e-9)  # manually set the range
    #     cf_fw_cur.set_ylim(0, 4e-9)  # manually set the range

    plt.tight_layout()
    plt.show()

def scatter_performance(acc_types, sram_sizes, workload,
                                   raw_data={}, d1_equal_d2=True, active_plot=True):
    # This function is to plot carbon cost in entire exploration space in a scatter plot.
    # The output figure consists of 4 subplots (x-axis: area):
    # 1th (left): carbon footprint breakdown (fixed-time)
    # 2th (right): carbon footprint breakdown (fixed-work)
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
              u'#bcbd22', u'#17becf']
    markers = ["s", "o", "^", "p", "D", "P"]
    font_size = 15
    df = raw_data["data"]
    df = df[(df.workload == workload)]
    periods = raw_data["periods"]
    if d1_equal_d2:
        dim_d1 = df.dim
    else:
        dim_d1 = df.dim // 8
    df["imc"] = dim_d1.astype(str) + f"$\\times$" + df.dim.astype(str)  # D1 x D2
    ## For active plotting using plotly.express
    if active_plot:
        df["dim_str"] = df["dim"].astype(str)
        df["sram_size_KB"] = (df["sram_size"]/1024).astype(str)  # unit: B -> KB
        if workload != "geo":
            df["t_cf_ft_op"] = [x["opcf"] for x in df.cf_ft]
            df["t_cf_ft_em"] = [x["soc_epa"]+x["soc_gpa"]+x["soc_mpa"]+x["dram"] for x in df.cf_ft]
        fig = px.scatter(df, x="t_area", y="t_cf_ft_ex_pkg", color="sram_size_KB", symbol="acc_type",
                         hover_data={"acc_type", "sram_size_KB", "dim"}, log_y=True, log_x=True)
        ## add marker style
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(title="Scenario: continuously active",
                          yaxis=dict(title="Carbon/task (exclude packaging) [g, CO2/task]"),
                          xaxis=dict(title="Area (mm2)"))
        fig.show()
    else:
        fig_rows_nbs = 1
        fig_cols_nbs = 1
        fig, axs = plt.subplots(nrows=fig_rows_nbs, ncols=fig_cols_nbs, figsize=(5, 4))
        topsw_scatter = axs
        marker_size = 50
        for ii_a, a in enumerate(acc_types):
            for ii_b, sram_size in enumerate(sram_sizes):
                dims = sorted(list(set(df["dim"].to_list())))
                for ii_c, dim in enumerate(dims):
                    dff = df[(df.acc_type == a) & (df.sram_size == sram_size) & (df.dim == dim)]
                    # Create positions for the bars on the x-axis
                    area = dff.t_area.to_numpy()


                    # Plot cf, simple (fixed-time)
                    topsw = dff.topsw.to_numpy()
                    cfft = dff.t_cf_ft_ex_pkg.to_numpy()
                    cffw = dff.t_cf_fw_ex_pkg.to_numpy()
                    tmp = raw_data["data"]
                    geo_cfft_op = 1
                    geo_cfft_em = 1
                    geo_cffw_op = 1
                    geo_cffw_em = 1
                    # for workload in ["ae", "ds_cnn", "resnet8", "mobilenet"]:
                    for workload in ["deeplabv3", "mobilebert", "mobilenet_edgetpu", "mobilenet_v2"]:
                        tmp_wk = tmp[(tmp.workload == workload)]
                        tmp_wk = tmp_wk[(tmp_wk.acc_type == a) & (tmp_wk.sram_size == sram_size) & (tmp_wk.dim == dim)]
                        cfft_op = np.array([x["opcf"] for x in tmp_wk.cf_ft])
                        cfft_em = np.array([x["soc_epa"]+x["soc_gpa"]+x["soc_mpa"]+x["dram"] for x in tmp_wk.cf_ft])
                        cffw_op = np.array([x["opcf"] for x in tmp_wk.cf_fw])
                        cffw_em = np.array([x["soc_epa"]+x["soc_gpa"]+x["soc_mpa"]+x["dram"] for x in tmp_wk.cf_fw])
                        geo_cfft_op *= cfft_op
                        geo_cfft_em *= cfft_em
                        geo_cffw_op *= cffw_op
                        geo_cffw_em *= cffw_em
                    geo_cfft_op = geo_cfft_op ** 0.25
                    geo_cfft_em = geo_cfft_em ** 0.25
                    geo_cffw_op = geo_cffw_op ** 0.25
                    geo_cffw_em = geo_cffw_em ** 0.25

                    # topsw_scatter.scatter(area, topsw,
                    # topsw_scatter.scatter(topsw, cfft,
                    # topsw_scatter.scatter(topsw, geo_cffw_op,
                    #                       color="white", marker=markers[ii_a], edgecolors=colors[ii_b], s=marker_size)
                    topsw_scatter.scatter(cfft, cffw,
                                          color=colors[ii_b], marker=markers[ii_a], edgecolors="black", s=marker_size)
        # configuration
        for i in range(0, fig_rows_nbs):
            axs.set_xscale("log")
            axs.set_yscale("log")
            axs.grid(which="both")
            axs.set_axisbelow(True)
        # axs.set_xlabel(f"Area (mm$^2$)")
        # axs.set_ylabel(f"TOP/s/W")
        # axs.set_xlabel(f"TOP/s/W")
        axs.set_xlabel(f"g, CO2/inf (CA scenario)")
        axs.set_ylabel(f"g, CO2/inf (PA scenario)")
        # axs[1][0].set_ylabel(f"\t\t\t\t\t\t\tg, CO$_2$/Inference (fixed-time)", fontsize=font_size * 1.2)
        # axs[1][1].set_ylabel(f"\t\t\t\t\t\t\tg, CO$_2$/Inference (fixed-work)", fontsize=font_size * 1.2)
        # axs[0][0].set_title(f"Simple task [{workload}]")
        # axs[0][1].set_title(f"Simple task [{workload}]")
        # axs[1][0].set_title(f"Complex task [{complexity}x {workload}]")
        # axs[1][1].set_title(f"Complex task [{complexity}x {workload}]")

        plt.tight_layout()
        plt.show()

def scatter_carbon_cost_four_cases(acc_types, sram_sizes, workload, complexity=50,
                                   raw_data={}, d1_equal_d2=True, active_plot=True):
    # This function is to plot carbon cost in entire exploration space in a scatter plot.
    # The output figure consists of 4 subplots (x-axis: area):
    # 1th (left): carbon footprint breakdown (fixed-time)
    # 2th (right): carbon footprint breakdown (fixed-work)
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
              u'#bcbd22', u'#17becf']
    markers = ["s", "o", "^", "p"]
    font_size = 15
    df = raw_data["data"]
    df = df[(df.workload == workload)]
    periods = raw_data["periods"]
    if d1_equal_d2:
        dim_d1 = df.dim
    else:
        dim_d1 = df.dim // 8
    df["imc"] = dim_d1.astype(str) + f"$\\times$" + df.dim.astype(str)  # D1 x D2
    ## For active plotting using plotly.express
    if active_plot:
        df["dim_str"] = df["dim"].astype(str)
        df["sram_size_KB"] = (df["sram_size"]/1024).astype(str)  # unit: B -> KB
        if workload != "geo":
            df["t_cf_ft_op"] = [x["opcf"] for x in df.cf_ft]
            df["t_cf_ft_em"] = [x["soc_epa"]+x["soc_gpa"]+x["soc_mpa"]+x["dram"] for x in df.cf_ft]
        # ans=df[(df.sram_size==8*1024) | (df.dim==32)]  # for breakdown
        # if workload != "geo":
        #     ans["opcf_ft"] = [x["opcf"] for x in ans.cf_ft]
        #     ans["emcf_ft"] = [x["soc_epa"]+x["soc_gpa"]+x["soc_mpa"] for x in ans.cf_ft]
        #     ans["dramcf_ft"] = [x["dram"] for x in ans.cf_ft]
        #     ans["opcf_fw"] = [x["opcf"] for x in ans.cf_fw]
        #     ans["emcf_fw"] = [x["soc_epa"] + x["soc_gpa"] + x["soc_mpa"] for x in ans.cf_fw]
        #     ans["dramcf_fw"] = [x["dram"] for x in ans.cf_fw]
        ## Scenario: continuously active
        # fig = px.scatter(ans, x="t_area", y="t_cf_ft_ex_pkg", color="sram_size_KB", symbol="dim_str", hover_data={"acc_type", "sram_size_KB", "dim"}, log_y=True, log_x=True)
        fig = px.scatter(df, x="t_area", y="t_cf_ft_ex_pkg", color="sram_size_KB", symbol="acc_type", hover_data={"acc_type", "sram_size_KB", "dim"}, log_y=True, log_x=True)
        ## add marker style
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
        fig.update_layout(title="Scenario: continuously active", yaxis=dict(title="Carbon/task (exclude packaging) [g, CO2/task]"), xaxis=dict(title="Area (mm2)"))
        fig.show()
        ## Scenario: periodically active
        fig = px.scatter(df, x="t_area", y="t_cf_fw_ex_pkg", color="sram_size_KB", symbol="acc_type",
                         hover_data={"acc_type", "sram_size_KB", "dim"})
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(title="Scenario: periodically active",
                          yaxis=dict(title="Carbon/task (exclude packaging) [g, CO2/task]"),
                          xaxis=dict(title="Area (mm2)"))
        fig.show()
        ## topsw
        fig = px.scatter(df, x="t_area", y="topsw", color="sram_size_KB", symbol="acc_type",
                         hover_data={"acc_type", "sram_size_KB", "dim"}, log_x=True, log_y=True)
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(title="TOP/s/W",
                          yaxis=dict(title="TOP/s/W"),
                          xaxis=dict(title="Area (mm2)"))
        fig.show()
        ## tops
        fig = px.scatter(df, x="t_area", y="tops", color="sram_size_KB", symbol="acc_type",
                         hover_data={"acc_type", "sram_size_KB", "dim"}, log_x=True, log_y=True)
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(title="TOP/s",
                          yaxis=dict(title="TOP/s"),
                          xaxis=dict(title="Area (mm2)"))
        fig.show()
        ## topsmm2
        fig = px.scatter(df, x="t_area", y="topsmm2", color="sram_size_KB", symbol="acc_type",
                         hover_data={"acc_type", "sram_size_KB", "dim"}, log_x=True, log_y=True)
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(title="TOP/s/mm2",
                          yaxis=dict(title="TOP/s/mm2"),
                          xaxis=dict(title="Area (mm2)"))
        fig.show()
        ## cf_fw (y) vs. cf_ft (x)
        fig = px.scatter(df, x="t_cf_ft", y="t_cf_fw", color="sram_size_KB", symbol="acc_type",
                         hover_data={"acc_type", "sram_size_KB", "dim", "t_area"}, log_x=True, log_y=True)
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(title="Carbon: periodic active vs. continuous active",
                          yaxis=dict(title="Carbon/task (PA) [g, CO2/task]"),
                          xaxis=dict(title="Carbon/task (CA) [g, CO2/task]"))
        fig.show()
        ## cf_ft (y) vs. topsw (x)
        fig = px.scatter(df, x="topsw", y="t_cf_ft", color="sram_size_KB", symbol="acc_type",
                         hover_data={"acc_type", "sram_size_KB", "dim", "t_area"}, log_x=True, log_y=True)
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(title="Carbon [continuous active] vs. TOP/s/W",
                          yaxis=dict(title="Carbon/task (CA) [g, CO2/task]"),
                          xaxis=dict(title="TOP/s/W"))
        fig.show()
        breakpoint()
    else:
        fig_rows_nbs = 2
        fig_cols_nbs = 2
        fig, axs = plt.subplots(nrows=fig_rows_nbs, ncols=fig_cols_nbs, figsize=(10, 8))
        cf_ft_simple_cur = axs[0][0]  # CF/inference
        cf_fw_simple_cur = axs[0][1]  # CF/inference
        cf_ft_complex_cur = axs[1][0]  # CF/inference
        cf_fw_complex_cur = axs[1][1]  # CF/inference

        for ii_a, a in enumerate(acc_types):
            for ii_b, sram_size in enumerate(sram_sizes):
                dims = sorted(list(set(df["dim"].to_list())))
                for ii_c, dim in enumerate(dims):
                    dff = df[(df.acc_type == a) & (df.sram_size == sram_size) & (df.dim == dim)]
                    # Create positions for the bars on the x-axis
                    x_pos = dff.t_area.to_numpy()

                    # Plot cf, simple (fixed-time)
                    cf_ft = dff.t_cf_ft_ex_pkg.to_numpy()
                    if ii_b == 0:
                        label = f"{a}, {dim}"
                    else:
                        label = None
                    cf_ft_simple_cur.scatter(x_pos, cf_ft, label=label, color=colors[ii_c], marker=markers[ii_a])

                    # Plot cf, simple (fixed-work)
                    cf_fw = dff.t_cf_fw_ex_pkg.to_numpy()
                    cf_fw_simple_cur.plot(x_pos, cf_fw, label=label, color=colors[ii_c], marker=markers[ii_a])

                    # Plot cf, complex (fixed-time)
                    cf_ft_complex = cf_ft * complexity
                    cf_ft_complex_cur.plot(x_pos, cf_ft_complex, label=label, color=colors[ii_c], marker=markers[ii_a])

                    # Plot cf, complext (fixed-work)
                    cf_fw_complex = calc_carbon_footprint_for_complex_task(raw_data=raw_data,
                                                                           acc_type=a,
                                                                           workload=workload,
                                                                           sram_size=sram_size,
                                                                           complexity=complexity)
                    cf_fw_complex_cur.plot(x_pos, cf_fw_complex[ii_c], label=label, color=colors[ii_c], marker=markers[ii_a])
    # configuration
        for i in range(0, fig_rows_nbs):
            for j in range(0, fig_cols_nbs):
                axs[i][j].set_xscale("log")
                axs[i][j].set_yscale("log")
                axs[i][j].grid(which="both")
                axs[i][j].set_axisbelow(True)
                # axs[i][j].legend(loc="upper left")
        axs[fig_rows_nbs - 1][0].set_xlabel(f"Area (mm$^2$)")
        axs[fig_rows_nbs - 1][1].set_xlabel(f"Area (mm$^2$)")
        axs[1][0].set_ylabel(f"\t\t\t\t\t\t\tg, CO$_2$/Inference (fixed-time)", fontsize=font_size*1.2)
        axs[1][1].set_ylabel(f"\t\t\t\t\t\t\tg, CO$_2$/Inference (fixed-work)", fontsize=font_size*1.2)
        axs[0][0].set_title(f"Simple task [{workload}]")
        axs[0][1].set_title(f"Simple task [{workload}]")
        axs[1][0].set_title(f"Complex task [{complexity}x {workload}]")
        axs[1][1].set_title(f"Complex task [{complexity}x {workload}]")

        plt.tight_layout()
        plt.show()

def memory_hierarchy_dut_for_pdigital_os(parray, visualize=False, sram_size=256*1024, dram_size=1 * 1024 * 1024, dram_ac_cost_per_bit=3.7):
    # This function defines the memory hierarchy in the hardware template.
    # @para parray: digital pe array object
    # @para visualize: whether illustrate teh memory hierarchy
    # @para sram_size: define the on-chip sram size, unit: byte
    # @para dram_size: define the off-chip dram size, unit: byte
    # @para dram_ac_cost_per_bit: dram access cost per bit, unit: pJ
    """ [OPTIONAL] Get w_cost of imc cell group from CACTI if required """
    cacti_path = "zigzag/classes/cacti/cacti_master"

    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """
    reg_W1 = MemoryInstance(
        name="reg_W1",
        size=8,
        r_bw=8,
        w_bw=8,
        r_cost=0,
        w_cost=0.7 * 3 / 1e3 * (0.9**2) * 8,  # unit: pJ/access
        area=0.614 * 6 / 1e6 * 8,  # mm^2
        r_port=1,  # no standalone read port
        w_port=1,  # no standalone write port
        rw_port=0,  # 1 port for both reading and writing
        latency=1,  # no extra clock cycle required
    )
    reg_I1 = MemoryInstance(
        name="rf_I1",
        size=8,
        r_bw=8,
        w_bw=8,
        r_cost=0,
        w_cost=0.7 * 3 / 1e3 * (0.9**2) * 8,  # pJ/access
        area=0.614 * 6 / 1e6 * 8,  # mm^2
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
    )

    reg_O1 = MemoryInstance(
        name="rf_O1",
        size=16,
        r_bw=16,
        w_bw=16,
        r_cost=0,
        w_cost=0.7 * 3 / 1e3 * (0.9**2) * 16, # pJ/access
        area=0.614 * 6 / 1e6 * 16, # mm^2
        r_port=2,
        w_port=2,
        rw_port=0,
        latency=1,
    )

    ##################################### on-chip memory hierarchy building blocks #####################################

    # sram_size = 256 * 1024 # unit: byte
    sram_bw = max(parray.dimension_sizes[1] * 8 * parray.dimension_sizes[2],
                  parray.dimension_sizes[0] * 16 * parray.dimension_sizes[2])

    ac_time, sram_area, sram_r_cost, sram_w_cost = get_cacti_cost(cacti_path, 0.028, "sram", sram_size, sram_bw, hd_hash=str(hash((sram_size, sram_bw, random.randbytes(8)))))
    # ac_time, sram_area, sram_r_cost, sram_w_cost = get_cacti_cost(cacti_path, 0.028, "sram", sram_size, sram_bw)

    sram_256KB_256_3r_3w = MemoryInstance(
        name="sram_256KB",
        size=sram_size * 8,  # byte -> bit
        r_bw=sram_bw,
        w_bw=sram_bw,
        r_cost=sram_r_cost,
        w_cost=sram_w_cost,
        area=sram_area,
        r_port=4,
        w_port=4,
        rw_port=0,
        latency=1,
        min_r_granularity=sram_bw//16,  # assume there are 16 sub-banks
        min_w_granularity=sram_bw//16,  # assume there are 16 sub-banks
    )

    #######################################################################################################################

    # dram_size = 1*1024*1024*1024 # unit: byte
    # dram_size = 1 * 1024 * 1024  # unit: byte (change to 1MB to fit for carbon estimation for tinyml perf workloads)
    # dram_ac_cost_per_bit = 3.7  # unit: pJ/bit
    dram_bw = parray.dimension_sizes[0] * 8 * parray.dimension_sizes[2]
    dram_3r_3w = MemoryInstance(
        name="dram_1GB",
        size=dram_size*8, # byte -> bit
        r_bw=dram_bw,
        w_bw=dram_bw,
        r_cost=dram_ac_cost_per_bit*dram_bw,  # pJ/access
        w_cost=dram_ac_cost_per_bit*dram_bw,  # pJ/access
        area=0,
        r_port=3,
        w_port=3,
        rw_port=0,
        latency=1,
        min_r_granularity=dram_bw // 16,  # assume there are 16 sub-banks
        min_w_granularity=dram_bw // 16,  # assume there are 16 sub-banks
    )

    memory_hierarchy_graph = MemoryHierarchy(operational_array=parray)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    ## Following part is different from pdigtial_ws
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_O1,
        operands=("O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},),
        served_dimensions=set(),
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_W1,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(0, 1, 0)},
    )
    ## Above part is different from pdigtial_ws
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_I1,
        operands=("I1",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(1, 0, 0)},
    )

    ##################################### on-chip highest memory hierarchy initialization #####################################

    memory_hierarchy_graph.add_memory(
        memory_instance=sram_256KB_256_3r_3w,
        operands=("I1", "I2", "O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_2", "tl": "r_port_2", "fl": None, "th": None},
            {"fh": "w_port_3", "tl": "r_port_3", "fl": "w_port_4", "th": "r_port_4"},
        ),
        served_dimensions="all",
    )

    ####################################################################################################################

    memory_hierarchy_graph.add_memory(
        memory_instance=dram_3r_3w,
        operands=("I1", "I2", "O"),
        # operands=("I2",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_2", "tl": "r_port_2", "fl": None, "th": None},
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_3", "th": "r_port_3"},
        ),
        served_dimensions="all",
    )

    if visualize:
        from zigzag.visualization.graph.memory_hierarchy import (
            visualize_memory_hierarchy_graph,
        )

        visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph

def memory_hierarchy_dut_for_pdigital_ws(parray, visualize=False, sram_size=256*1024, dram_size=1 * 1024 * 1024,
                                         dram_ac_cost_per_bit=3.7, enable_weight_dram_removal=False):
    # This function defines the memory hierarchy in the hardware template.
    # @para parray: digital pe array object
    # @para visualize: whether illustrate teh memory hierarchy
    # @para sram_size: define the on-chip sram size, unit: byte
    # @para dram_size: define the off-chip dram size, unit: byte
    # @para dram_ac_cost_per_bit: dram access cost per bit, unit: pJ
    # @para enable_weight_dram_removal: if remove dram level for weight and put weight also in on-chip SRAM.
    """ [OPTIONAL] Get w_cost of imc cell group from CACTI if required """
    cacti_path = "zigzag/classes/cacti/cacti_master"

    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """
    reg_W1 = MemoryInstance(
        name="reg_W1",
        size=8,
        r_bw=8,
        w_bw=8,
        r_cost=0,
        w_cost=0.7 * 3 / 1e3 * (0.9**2) * 8,  # unit: pJ/weight
        area=0.614 * 6 / 1e6 * 8,
        r_port=1,  # 1 standalone read port
        w_port=1,  # 1 standalone write port
        rw_port=0,  # no port for both reading and writing
        latency=1,  # 1 extra clock cycle required
    )
    reg_I1 = MemoryInstance(
        name="rf_I1",
        size=8,
        r_bw=8,
        w_bw=8,
        r_cost=0,
        w_cost=0.7 * 3 / 1e3 * (0.9**2) * 8,  # pJ/access
        area=0.614 * 6 / 1e6 * 8,  # mm^2
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
    )

    reg_O1 = MemoryInstance(
        name="rf_O1",
        size=16,
        r_bw=16,
        w_bw=16,
        r_cost=0,
        w_cost=0.7 * 3 / 1e3 * (0.9**2) * 16, # pJ/access
        area=0.614 * 6 / 1e6 * 16, # mm^2
        r_port=2,
        w_port=2,
        rw_port=0,
        latency=1,
    )

    ##################################### on-chip memory hierarchy building blocks #####################################

    # sram_size = 256 * 1024 # unit: byte
    # dimension_sizes:
    # [0]: D1, [1]: D2, [2]: D3
    sram_bw = max(parray.dimension_sizes[1] * 8 * parray.dimension_sizes[2],
                  parray.dimension_sizes[0] * 16 * parray.dimension_sizes[2])

    ac_time, sram_area, sram_r_cost, sram_w_cost = get_cacti_cost(cacti_path, 0.028, "sram", sram_size, sram_bw, hd_hash=str(hash((sram_size, sram_bw, random.randbytes(8)))))

    sram_256KB_256_3r_3w = MemoryInstance(
        name="sram_256KB",
        size=sram_size * 8, # byte -> bit
        r_bw=sram_bw,
        w_bw=sram_bw,
        r_cost=sram_r_cost,
        w_cost=sram_w_cost,
        area=sram_area,
        r_port=4,
        w_port=4,
        rw_port=0,
        latency=1,
        min_r_granularity=sram_bw//16, # assume there are 16 sub-banks
        min_w_granularity=sram_bw//16, # assume there are 16 sub-banks
    )

    #######################################################################################################################

    # dram_size = 1*1024*1024*1024 # unit: byte
    # dram_size = 1 * 1024 * 1024  # unit: byte (change to 1MB to fit for carbon estimation for tinyml perf workloads)
    # dram_ac_cost_per_bit = 3.7 # unit: pJ/bit
    dram_bw = parray.dimension_sizes[0] * 8 * parray.dimension_sizes[2]
    dram_3r_3w = MemoryInstance(
        name="dram_1GB",
        size=dram_size*8, # byte -> bit
        r_bw=dram_bw,
        w_bw=dram_bw,
        r_cost=dram_ac_cost_per_bit*dram_bw, # pJ/access
        w_cost=dram_ac_cost_per_bit*dram_bw, # pJ/access
        area=0,
        r_port=3,
        w_port=3,
        rw_port=0,
        latency=1,
        min_r_granularity=dram_bw // 16,  # assume there are 16 sub-banks
        min_w_granularity=dram_bw // 16,  # assume there are 16 sub-banks
    )

    memory_hierarchy_graph = MemoryHierarchy(operational_array=parray)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_W1,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions=set(),
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_I1,
        operands=("I1",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(1, 0, 0)},
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_O1,
        operands=("O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},),
        served_dimensions={(0, 1, 0)},
    )

    ##################################### on-chip highest memory hierarchy initialization #####################################

    if enable_weight_dram_removal:
        memory_hierarchy_graph.add_memory(
            memory_instance=sram_256KB_256_3r_3w,
            operands=("I1", "I2", "O",),
            port_alloc=(
                {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
                {"fh": "w_port_4", "tl": "r_port_4", "fl": None, "th": None},
                {"fh": "w_port_2", "tl": "r_port_2", "fl": "w_port_3", "th": "r_port_3"},
            ),
            served_dimensions="all",
        )
    else:
        memory_hierarchy_graph.add_memory(
            memory_instance=sram_256KB_256_3r_3w,
            operands=("I1", "O",),
            port_alloc=(
                {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
                {"fh": "w_port_2", "tl": "r_port_2", "fl": "w_port_3", "th": "r_port_3"},
            ),
            served_dimensions="all",
        )

    ####################################################################################################################

    if enable_weight_dram_removal:
        memory_hierarchy_graph.add_memory(
            memory_instance=dram_3r_3w,
            operands=("I1", "O"),
            # operands=("I2",),
            port_alloc=(
                {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
                # {"fh": "w_port_2", "tl": "r_port_2", "fl": None, "th": None},
                {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_3", "th": "r_port_3"},
            ),
            served_dimensions="all",
        )
    else:
        memory_hierarchy_graph.add_memory(
            memory_instance=dram_3r_3w,
            operands=("I1", "I2", "O"),
            # operands=("I2",),
            port_alloc=(
                {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
                {"fh": "w_port_2", "tl": "r_port_2", "fl": None, "th": None},
                {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_3", "th": "r_port_3"},
            ),
            served_dimensions="all",
        )

    if visualize:
        from zigzag.visualization.graph.memory_hierarchy import (
            visualize_memory_hierarchy_graph,
        )

        visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph

def memory_hierarchy_dut_for_imc(imc_array, visualize=False, sram_size=256*1024, dram_size=1 * 1024 * 1024,
                                 dram_ac_cost_per_bit=3.7, enable_weight_dram_removal=False):
    # This function defines the memory hierarchy in the hardware template.
    # @para imc_array: imc pe array object
    # @para visualize: whether illustrate teh memory hierarchy
    # @para sram_size: define the on-chip sram size, unit: byte
    # @para dram_size: define the off-chip dram size, unit: byte
    # @para dram_ac_cost_per_bit: dram access cost per bit, unit: pJ
    # @para enable_weight_dram_removal: if remove dram level for weight and put weight also in on-chip SRAM.
    """ [OPTIONAL] Get w_cost of imc cell group from CACTI if required """
    cacti_path = "zigzag/classes/cacti/cacti_master"
    tech_param = imc_array.unit.logic_unit.tech_param
    hd_param = imc_array.unit.hd_param
    dimensions = imc_array.unit.dimensions
    output_precision = hd_param["input_precision"] + hd_param["weight_precision"]
    if hd_param["enable_cacti"]:
        # unit: pJ/weight writing
        w_cost_per_weight_writing = get_w_cost_per_weight_from_cacti(cacti_path, tech_param, hd_param, dimensions)
    else:
        w_cost_per_weight_writing = hd_param["w_cost_per_weight_writing"] # user-provided value (unit: pJ/weight)

    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """
    cell_group = MemoryInstance(
        name="cell_group",
        size=hd_param["weight_precision"] * hd_param["group_depth"],
        r_bw=hd_param["weight_precision"],
        w_bw=hd_param["weight_precision"],
        r_cost=0,
        w_cost=w_cost_per_weight_writing,  # unit: pJ/weight
        area=0,  # this area is already included in imc_array
        r_port=0,  # no standalone read port
        w_port=0,  # no standalone write port
        rw_port=1,  # 1 port for both reading and writing
        latency=0,  # no extra clock cycle required
    )
    reg_I1 = MemoryInstance(
        name="rf_I1",
        size=hd_param["input_precision"],
        r_bw=hd_param["input_precision"],
        w_bw=hd_param["input_precision"],
        r_cost=0,
        w_cost=tech_param["dff_cap"] * (tech_param["vdd"] ** 2) * hd_param["input_precision"], # pJ/access
        area=tech_param["dff_area"] * hd_param["input_precision"], # mm^2
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
    )

    reg_O1 = MemoryInstance(
        name="rf_O1",
        size=output_precision,
        r_bw=output_precision,
        w_bw=output_precision,
        r_cost=0,
        w_cost=tech_param["dff_cap"] * (tech_param["vdd"] ** 2) * output_precision, # pJ/access
        area=tech_param["dff_area"] * output_precision, # mm^2
        r_port=2,
        w_port=2,
        rw_port=0,
        latency=1,
    )

    ##################################### on-chip memory hierarchy building blocks #####################################

    # sram_size = 256 * 1024 # unit: byte
    sram_bw = max(imc_array.unit.bl_dim_size * hd_param["input_precision"] * imc_array.unit.nb_of_banks,
                  imc_array.unit.wl_dim_size * output_precision * imc_array.unit.nb_of_banks)
    ac_time, sram_area, sram_r_cost, sram_w_cost = get_cacti_cost(cacti_path, tech_param["tech_node"], "sram",
                                                                  sram_size, sram_bw,
                                                                  hd_hash=str(hash((sram_size, sram_bw, random.randbytes(8)))))
    sram_256KB_256_3r_3w = MemoryInstance(
        name="sram_256KB",
        size=sram_size * 8, # byte -> bit
        r_bw=sram_bw,
        w_bw=sram_bw,
        r_cost=sram_r_cost,
        w_cost=sram_w_cost,
        area=sram_area,
        r_port=4,
        w_port=4,
        rw_port=0,
        latency=1,
        min_r_granularity=sram_bw//16, # assume there are 16 sub-banks
        min_w_granularity=sram_bw//16, # assume there are 16 sub-banks
    )

    #######################################################################################################################

    # dram_size = 1*1024*1024*1024 # unit: byte
    # dram_size = 1 * 1024 * 1024  # unit: byte (change to 1MB to fit for carbon estimation for tinyml perf workloads)
    # dram_ac_cost_per_bit = 3.7 # unit: pJ/bit
    dram_bw = imc_array.unit.wl_dim_size * hd_param["weight_precision"] * imc_array.unit.nb_of_banks
    dram_3r_3w = MemoryInstance(
        name="dram_1GB",
        size=dram_size*8, # byte -> bit
        r_bw=dram_bw,
        w_bw=dram_bw,
        r_cost=dram_ac_cost_per_bit*dram_bw, # pJ/access
        w_cost=dram_ac_cost_per_bit*dram_bw, # pJ/access
        area=0,
        r_port=3,
        w_port=3,
        rw_port=0,
        latency=1,
        min_r_granularity=dram_bw // 16,  # assume there are 16 sub-banks
        min_w_granularity=dram_bw // 16,  # assume there are 16 sub-banks
    )

    memory_hierarchy_graph = MemoryHierarchy(operational_array=imc_array)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    memory_hierarchy_graph.add_memory(
        memory_instance=cell_group,
        operands=("I2",),
        port_alloc=({"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},),
        served_dimensions=set(),
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_I1,
        operands=("I1",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(1, 0, 0)},
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_O1,
        operands=("O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},),
        served_dimensions={(0, 1, 0)},
    )

    ##################################### on-chip highest memory hierarchy initialization #####################################

    if enable_weight_dram_removal:
        memory_hierarchy_graph.add_memory(
            memory_instance=sram_256KB_256_3r_3w,
            operands=("I1", "I2", "O",),
            port_alloc=(
                {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
                {"fh": "w_port_4", "tl": "r_port_4", "fl": None, "th": None},
                {"fh": "w_port_2", "tl": "r_port_2", "fl": "w_port_3", "th": "r_port_3"},
            ),
            served_dimensions="all",
        )
    else:
        memory_hierarchy_graph.add_memory(
            memory_instance=sram_256KB_256_3r_3w,
            operands=("I1", "O",),
            port_alloc=(
                {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
                {"fh": "w_port_2", "tl": "r_port_2", "fl": "w_port_3", "th": "r_port_3"},
            ),
            served_dimensions="all",
        )

    ####################################################################################################################

    if enable_weight_dram_removal:
        memory_hierarchy_graph.add_memory(
            memory_instance=dram_3r_3w,
            operands=("I1", "O"),
            # operands=("I2",),
            port_alloc=(
                {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
                # {"fh": "w_port_2", "tl": "r_port_2", "fl": None, "th": None},
                {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_3", "th": "r_port_3"},
            ),
            served_dimensions="all",
        )
    else:
        memory_hierarchy_graph.add_memory(
            memory_instance=dram_3r_3w,
            operands=("I1", "I2", "O"),
            # operands=("I2",),
            port_alloc=(
                {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
                {"fh": "w_port_2", "tl": "r_port_2", "fl": None, "th": None},
                {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_3", "th": "r_port_3"},
            ),
            served_dimensions="all",
        )

    if visualize:
        from zigzag.visualization.graph.memory_hierarchy import (
            visualize_memory_hierarchy_graph,
        )

        visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph

def get_accelerator(acc_type, tech_param, hd_param, dims, sram_size=256*1024, dram_size_GB=1/1024, workload="resnet8",
                    dram_ac_cost_per_bit=3.7, workload_size=1e9,):
    # @para dram_size: dram size (unit: GB)
    # @para layer_size: workload size for weight. Unit: Byte
    assert acc_type in ["pdigital_ws", "pdigital_os", "AIMC", "DIMC"], f"acc_type {acc_type} not in [pdigital_ws, pdigital_os, AIMC, DIMC]"
    dram_size = int(math.ceil(dram_size_GB * 1024 * 1024 * 1024))  # unit: B
    # check if dram level for weight should be removed
    if sram_size > workload_size:
        enable_weight_dram_removal = True
    else:
        enable_weight_dram_removal = False
    if acc_type in ["AIMC", "DIMC"]:
        imc_array = ImcArray(tech_param, hd_param, dims)
        mem_hier = memory_hierarchy_dut_for_imc(imc_array, sram_size=sram_size, dram_size=dram_size,
                                                dram_ac_cost_per_bit=dram_ac_cost_per_bit,
                                                enable_weight_dram_removal=enable_weight_dram_removal)
        pe_area_total = 0  # no use parameter for IMC
    else:
        parray, multiplier_energy, pe_area_total = digital_array(acc_type, dims)
        if acc_type == "pdigital_ws":
            mem_hier = memory_hierarchy_dut_for_pdigital_ws(parray, sram_size=sram_size, dram_size=dram_size,
                                                            dram_ac_cost_per_bit=dram_ac_cost_per_bit,
                                                            enable_weight_dram_removal=enable_weight_dram_removal)
        else:
            mem_hier = memory_hierarchy_dut_for_pdigital_os(parray, sram_size=sram_size, dram_size=dram_size,
                                                            dram_ac_cost_per_bit=dram_ac_cost_per_bit)
        imc_array = parray

    core = {Core(1, imc_array, mem_hier)}
    acc_name = os.path.basename(__file__)[:-3]
    accelerator = Accelerator(acc_name, core)

    return accelerator, pe_area_total

def get_imc_param_setting(acc_type="DIMC", D1=32, D2=32, D3=1):
    ## type: pdigital, DIMC or AIMC
    # D1: int, can divide with 8
    # D2: int
    # D3: int
    assert acc_type in ["pdigital_ws", "pdigital_os", "AIMC", "DIMC"], f"acc_type {acc_type} not in [pdigital_ws, pdigital_os, AIMC, DIMC]"

    ##################
    ## dimensions
    dimensions = {
        # "D1": cols // 8,  # wordline dimension
        "D1": D1,  # wordline dimension
        "D2": D2,  # bitline dimension
        "D3": D3,  # nb_macros
    }  # {"D1": ("K", 4), "D2": ("C", 32),}

    if acc_type in ["pdigital_ws", "pdigital_os"]:
        tech_param_28nm = {}
        hd_param = {}
        return tech_param_28nm, hd_param, dimensions

    ##################
    ## tech_param
    tech_param_28nm = {
        "tech_node": 0.028,  # unit: um
        "vdd": 0.9,  # unit: V
        "nd2_cap": 0.7 / 1e3,  # unit: pF
        "xor2_cap": 0.7 * 1.5 / 1e3,  # unit: pF
        "dff_cap": 0.7 * 3 / 1e3,  # unit: pF
        "nd2_area": 0.614 / 1e6,  # unit: mm^2
        "xor2_area": 0.614 * 2.4 / 1e6,  # unit: mm^2
        "dff_area": 0.614 * 6 / 1e6,  # unit: mm^2
        "nd2_dly": 0.0478,  # unit: ns
        "xor2_dly": 0.0478 * 2.4,  # unit: ns
        # "dff_dly":  0.0478*3.4,         # unit: ns
    }

    ##################
    ## hd_param
    assert acc_type in ["DIMC", "AIMC"], f"acc_type {acc_type} not in range [DIMC, AIMC]"
    if acc_type == "DIMC":
        imc = "digital"
        in_bits_per_cycle = 1
    else:
        imc = "analog"
        in_bits_per_cycle = 2
    hd_param = {
        "pe_type": "in_sram_computing",  # for in-memory-computing. Digital core for different values.
        "imc_type": imc,  # "digital" or "analog"
        "input_precision": 8,  # activation precision
        "weight_precision": 8,  # weight precision
        "input_bit_per_cycle": in_bits_per_cycle,  # nb_bits of input/cycle (treated as DAC resolution)
        "group_depth": 1,  # m factor
        "adc_resolution": 8,  # ADC resolution
        "wordline_dimension": "D1",
        # hardware dimension where wordline is (corresponds to the served dimension of input regs)
        "bitline_dimension": "D2",
        # hardware dimension where bitline is (corresponds to the served dimension of output regs)
        "enable_cacti": True,  # use CACTI to estimated cell array area cost (cell array exclude build-in logic part)
        # Energy of writing weight. Required when enable_cacti is False.
        # "w_cost_per_weight_writing": 0.08,  # [OPTIONAL] unit: pJ/weight.
    }
    hd_param["adc_resolution"] = hd_param["input_bit_per_cycle"] + 0.5 * int(math.log2(dimensions["D2"]))
    ##################
    ## return para
    return tech_param_28nm, hd_param, dimensions

def digital_array(acc_type, dims):
    # Data is from envision paper (https://ieeexplore.ieee.org/abstract/document/7870353)
    # Tech: 28nm UTBB FD-SOI
    # PE array area is estimated from the chip picture, with total size: 0.75mm x 1.29mm for 16 x 16 int16 PEs
    # TOP/s/w: 2
    # Supply voltage: 1 V
    # Clock freq: 200 MHz
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.25  # unit: pJ/mac
    multiplier_area = 0.75*1.29/256/4  # mm2/PE
    area_total = multiplier_area * np.prod([dim for dim in dims.values()])
    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, dims)
    if acc_type in ["pdigital_ws"]:
        # add adder area
        nd2_area = 0.614 / 1e6  # unit: mm^2
        xor2_area = 0.614 * 2.4 / 1e6  # unit: mm^2
        adder_area_fa = 3 * nd2_area + 2 * xor2_area
        adder_input_pres = np.prod(multiplier_input_precision)
        nb_inputs_of_adder = dims["D2"]
        adder_depth = math.log2(nb_inputs_of_adder)
        nb_of_1b_adder = nb_inputs_of_adder * (adder_input_pres + 1) - (adder_input_pres + adder_depth + 1)
        single_adder_tree_area = adder_area_fa * nb_of_1b_adder
        adder_tree_counts = dims["D2"] * dims["D3"]
        total_adder_tree_area = single_adder_tree_area * adder_tree_counts
        area_total += total_adder_tree_area

    return multiplier_array, multiplier_energy, area_total

def adder_tree_delay(dims, adder_output_pres):
    nd2_dly = 0.0478  # unit: ns
    xor2_dly = 0.0478 * 2.4  # unit: ns
    nb_inputs_of_adder = dims["D2"]
    adder_dly_in2sum = 2 * xor2_dly
    adder_dly_in2cout = xor2_dly + 2 * nd2_dly
    adder_dly_cin2cout = 2 * nd2_dly
    adder_depth = math.log2(nb_inputs_of_adder)
    dly_adders = (adder_depth - 1) * adder_dly_in2sum + \
                 adder_dly_in2cout + \
                 (adder_output_pres - 1 - 1) * adder_dly_cin2cout
    return dly_adders

def calc_cf(energy, lat, area, nb_of_ops, lifetime=3, chip_yield=0.95, fixed_work_period=4e+6, dram_size=1/1024, dram_cost_removal=False):  # 4 ms by default
    #######################################################
    # This function is to calculate g, CO2 per operation for two scenarios: fixed-time and fixed-work.
    # Fixed-time scenario: assuming the design is fully activated during the entire lifetime.
    # Fixed-work scenario: assuming the design is only activated one time within a fixed period time.
    # Used model: ACT
    # Ref: Gupta, Udit, et al. "ACT: Designing sustainable computer systems with an architectural carbon modeling tool." Proceedings of the 49th Annual International Symposium on Computer Architecture. 2022.
    # NOTE: If setting nb_of_ops = 1, the result will correspond to g, CO2/TASK!
    # @para energy: energy cost, unit: pJ
    # @para lat: latency cost, unit: ns
    # @para area: area cost, unit: mm^2
    # @para nb_of_ops: number of operations
    # @para lifetime: lifetime of entire chip, unit: year
    # @para chip_yield: chip fabrication yield, range: float within (0, 1]
    # @para fixed_work_period: the period length for fixed-work scenario, unit: ns
    # @para dram_size: utilized dram size. Unit: GB. Default: 1MB for all four tinyml perf workloads (ceiled from 571680 B, assuming 8b/weight)
    #######################################################
    assert 0 < chip_yield <= 1, f"yield {chip_yield} is not in range (0,1]"
    #################
    ## para calc (specific for 28 nm)
    # unit: 
    # OP_CF -> g, CO2 per op from operational footprint
    # E_SOC -> g, CO2 per op from processors fabrication
    # E_PKG -> g, CO2 per op from chip package (assume there is only 1 soc chip)
    # E_DRAM -> g, CO2 per op from DRAM fabrication
    # E_CF = E_SOC + E_PKG + E_DRAM
    # TOTAL_CF -> g, CO2 in total per op

    # Below: calc carbon footprint for fixed-time scenario

    # Operational cost
    OP_CF = 301 * energy / (3.6 * (10 ** 18)) / nb_of_ops  # g, CO2/kWh * energy (pJ) / 3.6E18 / nb_of_ops

    runtime = lat  # unit: ns
    scale_factor = runtime / (lifetime * (
                3.1536 * 10 ** 16)) / nb_of_ops  # ns / (yr * 3.1536E16) / nb_of_ops (by now nb_of_ops is set to 1 in the upper-level script)

    # SOC fabrication cost
    E_SOC_EPA = 1 / chip_yield * (301 * 0.9 / 100) * area * scale_factor  # 1/Y * (301 * 0.9/100) (g, CO2/mm2) * area
    E_SOC_GPA = 1 / chip_yield * (100 / 100) * area * scale_factor
    E_SOC_MPA = 1 / chip_yield * (500 / 100) * area * scale_factor
    E_SOC = E_SOC_EPA + E_SOC_GPA + E_SOC_MPA  # g, CO2/op
    # Chip pacakge cost
    chip_nbs = 1
    E_PKG = 0.15 * 1000 * chip_nbs * scale_factor  # g, CO2/chip * chip_nbs
    # External memory cost
    # DRAM type: LPDDR3, SK hynix
    # CPS data source: SK hynix. 2018. SK hynix Sustainability Report 2018: DBL, SKHynix's New Growth Strategy. https://www.skhynix.com/sustainability/UI-FR-SA1601
    CPS = 230 * ((28/30)**2)  # cabon-per-size factor: g, CO2/GB
    if dram_cost_removal:
        E_DRAM = 0
    else:
        # dram_size = 1/1024  # 1MB for all four tinyml perf workloads (ceiled from 571680 B, assuming 8b/weight)
        E_DRAM = CPS * dram_size * scale_factor  # g, CO2
    # Total fabrication cost
    E_CF = E_SOC + E_PKG + E_DRAM
    # Total carbon cost, including operational cost and fabrication cost
    CF_PER_OP_fixed_time = OP_CF + E_CF  # g, CO2/op
    tt_cf_bd_fixed_time = {  # total cf breakdown details (fixed-time scenario)
        "ops": nb_of_ops,
        "runtime": runtime,
        "lifetime": lifetime,
        "scale_factor": scale_factor,
        "opcf": OP_CF,
        "soc_epa": E_SOC_EPA,
        "soc_gpa": E_SOC_GPA,
        "soc_mpa": E_SOC_MPA,
        "package": E_PKG,
        "dram": E_DRAM,
    }
    CF_PER_OP_fixed_time_ex_pkg = CF_PER_OP_fixed_time - E_PKG  # cost excluding package

    # Below: calc carbon footprint for fixed-work scenario

    # Compared to fixed-time scenario:
    # - by default, the runtime is fixed at 4ms, meaning 4ms/task for whatever hardware or workloads.
    # - OP_CF will not change.
    # - scale_factor will be fixed for all fabrication cost.
    runtime = fixed_work_period  # ns
    scale_factor = runtime / (lifetime * (3.1536 * 10 ** 16)) / nb_of_ops
    # What if fixed_work_period < lat? Then we need multiple chips, so the embodied footprint will increase.
    if fixed_work_period < lat:
        scale_factor *= lat/fixed_work_period
    # Operational cost
    OP_CF = OP_CF
    # SOC fabrication cost
    E_SOC_EPA = 1 / chip_yield * (301 * 0.9 / 100) * area * scale_factor  # 1/Y * (301 * 0.9/100) (g, CO2/mm2) * area
    E_SOC_GPA = 1 / chip_yield * (100 / 100) * area * scale_factor
    E_SOC_MPA = 1 / chip_yield * (500 / 100) * area * scale_factor
    E_SOC = E_SOC_EPA + E_SOC_GPA + E_SOC_MPA  # g, CO2/op
    # Chip pacakge cost
    chip_nbs = 1
    E_PKG = 0.15 * 1000 * chip_nbs * scale_factor  # g, CO2/chip * chip_nbs
    # External memory cost
    # DRAM type: LPDDR3
    CPS = 200 * ((28 / 30) ** 2)  # cabon-per-size factor: g, CO2/GB
    E_DRAM = CPS * dram_size * scale_factor  # g, CO2
    # Total fabrication cost
    E_CF = E_SOC + E_PKG + E_DRAM
    # Total carbon cost, including operational cost and fabrication cost
    CF_PER_OP_fixed_work = OP_CF + E_CF  # g, CO2/op

    tt_cf_bd_fixed_work = {  # total cf breakdown details (fixed-work scenario)
        "ops": nb_of_ops,
        "runtime": runtime,  # (ns) assume 4ms per task (this is the runtime required by AIMC 1024x1024 on mb-v1, which is the worst case in the experiment on tinyml perf workloads)
        "lifetime": lifetime,  # unit: year
        "scale_factor": scale_factor,
        "opcf": OP_CF,
        "soc_epa": E_SOC_EPA,
        "soc_gpa": E_SOC_GPA,
        "soc_mpa": E_SOC_MPA,
        "package": E_PKG,
        "dram": E_DRAM,
    }
    CF_PER_OP_fixed_work_ex_pkg = CF_PER_OP_fixed_work - E_PKG  # cost excluding package

    return CF_PER_OP_fixed_time, tt_cf_bd_fixed_time, CF_PER_OP_fixed_work, tt_cf_bd_fixed_work, CF_PER_OP_fixed_time_ex_pkg,CF_PER_OP_fixed_work_ex_pkg

def plot_area_trend_in_literature(data):
    # This function is to check if the area versus years follows an ascending order.
    # TODO: scale area to tech node 28nm
    areas_input = data[:, 10].tolist()  # die area
    techs = data[:, 3].astype(int).tolist()
    netsizes = data[:, 11].tolist()  # unit: B
    array_parallelisms = data[:, 8].tolist()  # unit: #ops/cycle
    topsws = data[:, 5].tolist()
    dois = data[:, 0].tolist()
    cit_idxs = data[:, 12].tolist()  # paper citation idx
    # remove elements when netsize is None
    clean_netsizes = []
    clean_areas = []
    clean_techs = []
    clean_parallelisms = []
    clean_topsws = []
    clean_dois = []
    clean_idxs = []
    for idx in range(len(netsizes)):
        if netsizes[idx] is not None and areas_input[idx] is not None:
            # exclude the case for Transformer
            clean_netsizes.append(netsizes[idx])
            clean_areas.append(areas_input[idx])
            clean_techs.append(techs[idx])
            clean_parallelisms.append(array_parallelisms[idx])
            clean_topsws.append(topsws[idx])
            clean_dois.append(dois[idx])
            clean_idxs.append(cit_idxs[idx])
        else:
            continue
    # scale area to 28nm according to techs
    area_scaled = []
    for idx in range(len(clean_techs)):
        tech = clean_techs[idx]
        if tech == 28:
            area_scaled.append(clean_areas[idx])
        elif tech > 28:  # square scale for planar tech
            area_scaled.append( clean_areas[idx] * ((28/tech)**2) )
        elif tech < 28:  # linear scale for FinFET tech
            area_scaled.append( clean_areas[idx] * 28/tech)
    # convert netsize: B -> MB
    clean_netsizes = np.array(clean_netsizes)
    clean_netsizes = clean_netsizes/1024/1024
    clean_netsizes = clean_netsizes.tolist()
    # sort netsize in ascending order
    sorted_indices = sorted(range(len(clean_netsizes)), key=lambda k: clean_netsizes[k])
    netsizes_sorted = [clean_netsizes[i] for i in sorted_indices]
    areas_sorted = [area_scaled[i] for i in sorted_indices]
    parallelisms_sorted = [clean_parallelisms[i] for i in sorted_indices]
    topsws_sorted = [clean_topsws[i] for i in sorted_indices]
    dois_sorted = [clean_dois[i] for i in sorted_indices]
    cit_idxs_sorted = [clean_idxs[i] for i in sorted_indices]

    ####################
    # plot figures
    colors = cm.hsv(np.linspace(0, 255, np.array(areas_sorted).shape[0]).astype(int))
    scale_factor = 20 * np.pi
    blob_area = scale_factor * np.array(areas_sorted)
    fig, ax = plt.subplots(figsize=(10, 8))
    fontsize = 24
    ax.scatter(netsizes_sorted, topsws_sorted, s=blob_area, marker="o", edgecolors="black", c=colors)
    ax.scatter(netsizes_sorted, topsws_sorted, s=2, marker="o", c="white")
    print(dois_sorted)

    # text to label #parallelism
    for x, y, cit_idx in zip(netsizes_sorted, topsws_sorted, cit_idxs_sorted):
        ax.text(1.1 * x, y, f"[{cit_idx}]", ha="center", fontsize=12)

    ## Plot legend blobs
    legend_x = np.asarray([2, 5, 10, 30, 80])
    legend_y = np.asarray([100, 100, 100, 100, 100])
    legend_values = np.asarray([0.1, 1, 5, 10, 50])
    legend_names = np.asarray(["0.1 mm$^2$", "1 mm$^2$", "5 mm$^2$", "10 mm$^2$", "50 mm$^2$"])
    ax.scatter(legend_x, legend_y, s=legend_values*scale_factor, c="white", edgecolors="black")
    ax.annotate("Area:", xy=(legend_x[0]*0.5, legend_y[0]), verticalalignment="center", horizontalalignment="center", fontsize=18)
    for x, y, txt in zip(legend_x, legend_y, legend_names):
        if x < 80:
            ax.text(x, y+3, f"{txt}", ha="center", fontsize=12)
        else:
            ax.text(x, y, f"{txt}", ha="center", fontsize=12)



    # Figure configuration
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # ax.set_ylabel(f"TOP/s/W (macro-level)", fontsize=fontsize)
    ax.set_ylabel(f"TOP/s/W", fontsize=fontsize)
    ax.set_xlabel("Model size [MB]", fontsize=fontsize)
    ax.set_xscale("log")
    ax.grid("both")
    plt.tight_layout()
    plt.show()

def cacti_sanity_check(sram_size, d):
    row_counts_in_sram = sram_size / (2 * d)  # sram_size * precision / partial_output_precision
    if row_counts_in_sram < 32:
        return False
    else:
        return True

def zigzag_similation_and_result_storage(workloads: list, acc_types: list, sram_sizes: list, Dimensions: list, periods: dict,
                                         pkl_name: str, dram_size: float, dram_ac_cost_per_bit: float,
                                         possible_dram_energy_removal: bool, size_workloads: dict, d1_equal_d2: bool,
                                         workload_suit: str):
    # Run zigzag simulation for peak and tinyml workloads.
    # workloads: peak, ds_cnn, ae, mobilenet, resnet8
    # acc_types: AIMC, DIMC, pdigital_ws, pdigital_os
    # sram_sizes: int
    # Dimensions: int
    # periods: float
    # @para dram_size: utilized dram size
    # @para dram_ac_cost_per_bit: dram access cost per bit (unit: pJ)
    # @para possible_dram_energy_removal: remove dram energy cost if on-chip weight regs size > workload size
    # @para d1_equal_d2: Ture: D1=D2, False: D1=D2/8
    trig_time = time.time()
    data_vals = []
    if workload_suit in ["tiny", "mobile"]:
        output_dir = f"outputs_{workload_suit}"
    else:
        output_dir = f"outputs"

    os.makedirs(output_dir, exist_ok=True)
    os.system(f"rm -rf {output_dir}/*")

    workloads_dir = {
        "ds_cnn": "zigzag/inputs/examples/workload/mlperf_tiny/ds_cnn.onnx",
        "ae": "zigzag/inputs/examples/workload/mlperf_tiny/deepautoencoder.onnx",
        "mobilenet": "zigzag/inputs/examples/workload/mlperf_tiny/mobilenet_v1.onnx",
        "resnet8": "zigzag/inputs/examples/workload/mlperf_tiny/resnet8.onnx",
        "resnet18": "zigzag/inputs/examples/workload/resnet18.onnx",
        "deeplabv3": "zigzag/inputs/examples/workload/mlperf_mobile/deeplabv3_mnv2_ade20k_inferred.onnx",
        "mobilebert": "zigzag/inputs/examples/workload/mlperf_mobile/mobilebert_inferred.onnx",
        "mobilenet_edgetpu": "zigzag/inputs/examples/workload/mlperf_mobile/mobilenet_edgetpu_inferred.onnx",
        "mobilenet_v2": "zigzag/inputs/examples/workload/mlperf_mobile/ssd_mobilenet_v2_inferred.onnx",
    }
    for workload in workloads:
        for acc_type in acc_types:
            for sram_size in sram_sizes:
                for d in Dimensions:
                    # if workload == workloads[-1] and acc_type == acc_types[-1]:  # show info only at last, to lower the simulation time overhead
                    iter_start_time = time.time()
                    print(f"workload: {workload}, acc: {acc_type}, sram (KB): {sram_size//1024}, dim: {d}", end=", ")

                    # sanity check: if cacti support curren sram size
                    check_pass = cacti_sanity_check(sram_size=sram_size, d=d)
                    if not check_pass:
                        print(f"case skipped: [workload: {workload}, acc: {acc_type}, sram_size: {sram_size}B, dim: {d}]")
                        continue

                    if d1_equal_d2:
                        tech_param, hd_param, dims = get_imc_param_setting(acc_type=acc_type, D1=d, D2=d, D3=1)
                    else:
                        assert d//8 == d/8, f"current dim size {d} cannot divide 8."
                        tech_param, hd_param, dims = get_imc_param_setting(acc_type=acc_type, D1=d//8, D2=d, D3=1)
                    if workload == "peak":
                        # peak performance assessment below
                        if acc_type in ["AIMC", "DIMC"]:
                            imc = ImcArray(tech_param, hd_param, dims)
                            area_bd = imc.area_breakdown  # dict
                            area_total = imc.total_area  # float (mm2)
                            tclk_bd = imc.tclk_breakdown  # dict
                            tclk_total = imc.tclk  # float (ns)
                            peak_en_bd = imc.unit.get_peak_energy_single_cycle()
                            peak_en_total = sum([v for v in peak_en_bd.values()])  # float (pJ)
                            nb_of_ops = np.prod([x for x in dims.values()]) * hd_param["input_bit_per_cycle"] / hd_param[
                                "input_precision"] * 2
                        else:  # acc_type == "pdigital_ws" or "pdigital_os"
                            parray, multiplier_energy, area_total = digital_array(acc_type, dims)
                            area_bd = {"pe": area_total}
                            tclk_bd = {"pe": 5}  # ns
                            tclk_total = 5  # ns
                            if acc_type in ["pdigital_ws"]:
                                additional_tclk = adder_tree_delay(dims=dims, adder_output_pres=16)
                                tclk_total += additional_tclk
                            peak_en_bd = {"pe": multiplier_energy * np.prod(parray.dimension_sizes)}
                            peak_en_total = np.prod(parray.dimension_sizes)  # pJ
                            nb_of_ops = np.prod(parray.dimension_sizes) * 2

                        cf_total_fixed_time, cf_bd_fixed_time, cf_total_fixed_work, cf_bd_fixed_work, cf_total_fixed_time_ex_pkg, cf_total_fixed_work_ex_pkg = calc_cf(
                            energy=peak_en_total, lat=tclk_total, area=area_total,
                            nb_of_ops=nb_of_ops, lifetime=3, chip_yield=0.95,
                            fixed_work_period=periods["peak"])  # unit: g, CO2/MAC

                        res = {
                            "workload": workload,
                            "acc_type": acc_type,
                            "sram_size": sram_size,
                            "dim": d,
                            "ops": nb_of_ops,
                            "area": area_bd,  # area breakdown (dict)
                            "lat": 1,
                            "tclk": tclk_bd,
                            "en": peak_en_bd,
                            "cf_ft": cf_bd_fixed_time,
                            "cf_fw": cf_bd_fixed_work,
                            "t_area": area_total,  # total area (float)
                            "t_lat": 1,
                            "t_tclk": tclk_total,
                            "t_en": peak_en_total,
                            "t_cf_ft": cf_total_fixed_time,  # unit: g, CO2/MAC
                            "t_cf_fw": cf_total_fixed_work,  # unit: g, CO2/MAC
                            "t_cf_ft_ex_pkg": cf_total_fixed_time_ex_pkg,  # unit: g, CO2/MAC
                            "t_cf_fw_ex_pkg": cf_total_fixed_work_ex_pkg,  # unit: g, CO2/MAC
                            "cme": None,
                        }
                        data_vals.append(res)
                    else:
                        # real workload performance assessment below
                        workload_dir = workloads_dir[workload]

                        if acc_type == "pdigital_os":
                            mapping = {
                                "default": {
                                    "core_allocation": 1,
                                    "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
                                    "spatial_mapping_hint": {"D1": ["K"], "D2": ["OX", "OY"]},
                                }
                            }
                        else:
                            mapping = {
                                "default": {
                                    "core_allocation": 1,
                                    "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
                                    "spatial_mapping_hint": {"D1": ["K", "OX", "OY"], "D2": ["C", "FX", "FY"]},
                                }
                            }
                        workload_size = size_workloads[workload]
                        accelerator, pe_area_total = get_accelerator(acc_type, tech_param, hd_param, dims, sram_size, dram_size, workload,
                                                      dram_ac_cost_per_bit, workload_size)

                        # Call API
                        hw_name = acc_type
                        wl_name = re.split(r"/|\.", workload_dir)[-1]
                        if wl_name == "onnx":
                            wl_name = re.split(r"/|\.", workload_dir)[-2]
                        experiment_id = f"{hw_name}-{wl_name}"
                        cmes_pkl_name = f"{experiment_id}-saved_list_of_cmes"
                        os.system(f"rm -rf {output_dir}/{experiment_id}-layer_overall_complete.json")

                        ans = get_hardware_performance_zigzag(
                            workload_dir,
                            accelerator,
                            mapping,
                            opt="EDP",
                            dump_filename_pattern=f"{output_dir}/{experiment_id}-layer_?.json",
                            pickle_filename=f"{output_dir}/{cmes_pkl_name}.pickle",
                        )
                        # Read output
                        with open(f"{output_dir}/{experiment_id}-layer_overall_complete.json", "r") as fp:
                            dat = json.load(fp)
                        en_total = dat["outputs"]["energy"]["energy_total"]  # float: pJ
                        lat_total = dat["outputs"]["latency"]["computation"]  # float: cycles
                        dram_cost_removal = False
                        if acc_type in ["AIMC", "DIMC"]:
                            area_total = dat["outputs"]["area (mm^2)"]["total_area"]  # float: mm2
                            tclk_total = dat["outputs"]["clock"]["tclk (ns)"]  # float: ns
                            # breakdown
                            en_bd = {
                                "array": dat["outputs"]["energy"]["operational_energy"],  # pJ
                                "mem": dat["outputs"]["energy"]["memory_energy"],  # pJ
                            }
                            lat_bd = {
                                "mac_computation": dat["outputs"]["latency"]["computation_breakdown"]["mac_computation"],
                                "mem_stalling": dat["outputs"]["latency"]["computation_breakdown"]["weight_loading"],
                            }
                            area_bd = dat["outputs"]["area (mm^2)"]["total_area_breakdown:"]
                            tclk_bd = dat["outputs"]["clock"]["tclk_breakdown (ns)"]
                            if possible_dram_energy_removal:
                                w_reg_size = np.prod([v for v in dims.values()])  # unit: weight element
                                if w_reg_size >= size_workloads[workload]:
                                    dram_energy = dat["outputs"]["energy"]["memory_energy_breakdown_per_level"]["W"][-1]
                                    en_total = en_total - dram_energy
                                    en_bd["mem"] = en_bd["mem"] - dram_energy
                                    dram_cost_removal = True
                        else:
                            # pe_array_area = accelerator.cores[0].operational_array.total_area  # only multipliers, no adders
                            mem_area = np.sum(accelerator.cores[0].memory_level_area)
                            area_total = pe_area_total + mem_area  # mm2
                            tclk_mac = 5  # ns
                            if acc_type in ["pdigital_ws"]:
                                additional_tclk = adder_tree_delay(dims=dims, adder_output_pres=16)
                                tclk_total = additional_tclk + tclk_mac
                            else:
                                additional_tclk = 0
                                tclk_total = tclk_mac
                            en_bd = {
                                "array": dat["outputs"]["energy"]["operational_energy"],  # pJ
                                "mem": dat["outputs"]["energy"]["memory_energy"],  # pJ
                            }
                            lat_bd = {
                                "mac_computation": dat["outputs"]["latency"]["computation_breakdown"][
                                    "mac_computation"],
                                "mem_stalling": dat["outputs"]["latency"]["computation_breakdown"]["mem_stalling"],
                            }
                            area_bd = {"mem_area": mem_area, "pe_area": pe_area_total}
                            tclk_bd = {"mac": tclk_mac, "additional_tclk": additional_tclk}
                        # calc CF (carbon footprint) (below is for g, CO2/op)
                        # cf_total, cf_bd = calc_cf(energy=en_total, lat=lat_total*tclk_total, area=area_total, nb_of_ops=ops_workloads[workload], lifetime=3, chip_yield=0.95)
                        # below is for g, CO2/inference
                        if lat_total * tclk_total > periods[workload]:  # does not meet the fixed-work requirement
                            fw_period_meet = False
                        else:
                            fw_period_meet = True
                        cf_total_fixed_time, cf_bd_fixed_time, cf_total_fixed_work, cf_bd_fixed_work, cf_total_fixed_time_ex_pkg, cf_total_fixed_work_ex_pkg = calc_cf(
                            energy=en_total, lat=lat_total * tclk_total, area=area_total,
                            nb_of_ops=1, lifetime=3, chip_yield=0.95, fixed_work_period=periods[workload],
                            dram_size=dram_size, dram_cost_removal=dram_cost_removal)
                        res = {
                            "workload": workload,
                            "acc_type": acc_type,
                            "sram_size": sram_size,
                            "dim": d,
                            "ops": ops_workloads[workload],
                            "area": area_bd,  # dict: mm2
                            "lat": lat_bd,  # dict: cycles
                            "tclk": tclk_bd,  # dict: ns
                            "en": en_bd,  # dict: pJ
                            "cf_ft": cf_bd_fixed_time,  # dict: g, CO2
                            "cf_fw": cf_bd_fixed_work,
                            "t_area": area_total,
                            "t_lat": lat_total,
                            "t_tclk": tclk_total,
                            "t_en": en_total,
                            "t_cf_ft": cf_total_fixed_time,
                            "t_cf_fw": cf_total_fixed_work,
                            "t_cf_ft_ex_pkg": cf_total_fixed_time_ex_pkg,
                            "t_cf_fw_ex_pkg": cf_total_fixed_work_ex_pkg,
                            "fw_period_meet": fw_period_meet,
                            "cme": ans[2],
                        }
                        data_vals.append(res)
                    iter_end_time = time.time()
                    print(f"elapsed time: {round(iter_end_time - iter_start_time)} sec")

    df = pd.DataFrame(data_vals)

    #########################################
    ## Add geo-mean and calculation for topsw/tops/topsmm2 to the results gotten above
    # Targeted workloads: mlperf-tiny benchmarks (4 workloads: ae, ds_cnn, mobilenet-v1, resnet8)
    # Formula:
    # TOP/s/W   = 1/(energy: pJ/op)
    # TOP/s     = nbs_of_ops / (time: ps)
    # TOP/s/mm2 = TOP/s / area: mm2
    # Deduction (for a single inference):
    # OP_CF = 301 * energy / (3.6 * (10**18)) / nb_of_ops # g, CO2/kWh * energy (pJ) / 3.6E18 / nb_of_ops
    # => OP_CF = 301 / (3.6E18) * pJ/op = 301/(3.6E18) / (TOP/s/W) , (g, CO2)
    # Fixed-time scenario:
    # E_SOC = 1/chip_yield * (301*0.9+100+500)/100 * area: mm2 * runtime: ns / (lifetime: year * (3.1536 * 10 ** 16))
    #       = 1/chip_yield * 8.71 * area:mm2 * runtime: ns / (9.4608E16) , (g, CO2)
    # => E_SOC = 1/chip_yield * (0.921/E16) * mm2 * ns = 1/chip_yield * (0.921/E13) * nbs_of_ops/(TOP/s/mm2)
    data_vals = []
    for acc_type in acc_types:
        for sram_size in sram_sizes:
            for dim in Dimensions:
                # sanity check: if cacti support curren sram size
                check_pass = cacti_sanity_check(sram_size=sram_size, d=dim)
                if not check_pass:
                    print(f"case skipped: [workload: {workload}, acc: {acc_type}, sram_size: {sram_size}B, dim: {dim}]")
                    continue
                geo_topsw = 1
                geo_tops = 1
                geo_topsmm2 = 1
                geo_ops = 1
                geo_lat = 1
                geo_en = 1
                geo_cf_ft = 1
                geo_cf_fw = 1
                geo_cf_ft_ex_pkg = 1
                geo_cf_fw_ex_pkg = 1
                geo_period_meet = True
                for workload in workloads:
                    if workload == "geo":  # skip if it's for geo-mean (will be re-calculated)
                        continue
                    dff = df[(df.workload == workload) & (df.acc_type == acc_type) & (df.sram_size == sram_size) & (
                                df.dim == dim)]
                    if dff.fw_period_meet.to_list()[0] == False:
                        geo_period_meet = False  # some workload does not pass fixed-work requirement
                    # re-arrange the stored dict and add metrics
                    new_res = {
                        "workload": dff.workload.to_list()[0],
                        "acc_type": dff.acc_type.to_list()[0],
                        "sram_size": dff.sram_size.to_list()[0],
                        "dim": dff.dim.to_list()[0],
                        "ops": dff.ops.to_list()[0],
                        "area": dff.area.to_list()[0],
                        "lat": dff.lat.to_list()[0],
                        "tclk": dff.tclk.to_list()[0],
                        "en": dff.en.to_list()[0],
                        "cf_ft": dff.cf_ft.to_list()[0],
                        "cf_fw": dff.cf_fw.to_list()[0],
                        "t_area": dff.t_area.to_list()[0],
                        "t_lat": dff.t_lat.to_list()[0],
                        "t_tclk": dff.t_tclk.to_list()[0],
                        "t_en": dff.t_en.to_list()[0],
                        "t_cf_ft": dff.t_cf_ft.to_list()[0],
                        "t_cf_fw": dff.t_cf_fw.to_list()[0],
                        "t_cf_ft_ex_pkg": dff.t_cf_ft_ex_pkg.to_list()[0],  # exclude package cost
                        "t_cf_fw_ex_pkg": dff.t_cf_fw_ex_pkg.to_list()[0],  # exclude package cost
                        "topsw": 1 / (dff.t_en.to_list()[0] / dff.ops.to_list()[0]),  # 1/(pJ/op)
                        "tops": dff.ops.to_list()[0] / (dff.t_lat.to_list()[0] * dff.t_tclk.to_list()[0] * 1000),
                        # ops/ps
                        "topsmm2": dff.ops.to_list()[0] / (dff.t_lat.to_list()[0] * dff.t_tclk.to_list()[0] * 1000) /
                                   dff.t_area.to_list()[0],
                        "fw_period_meet": dff.fw_period_meet.to_list()[0],
                        "cme": dff.cme.to_list()[0],
                    }
                    if ((workload_suit == "tiny") and (workload in ["ae", "ds_cnn", "mobilenet", "resnet8"])) or (
                            (workload_suit == "mobile") and (
                            workload in ["deeplabv3", "mobilebert", "mobilenet_edgetpu", "mobilenet_v2"])):
                        geo_topsw *= new_res["topsw"]
                        geo_tops *= new_res["tops"]
                        geo_topsmm2 *= new_res["topsmm2"]
                        geo_ops *= new_res["ops"]
                        geo_lat *= new_res["t_lat"]
                        geo_en *= new_res["t_en"]
                        geo_cf_ft *= new_res["t_cf_ft"]
                        geo_cf_fw *= new_res["t_cf_fw"]
                        geo_cf_ft_ex_pkg *= new_res["t_cf_ft_ex_pkg"]
                        geo_cf_fw_ex_pkg *= new_res["t_cf_fw_ex_pkg"]
                    data_vals.append(new_res)

                if workloads != ["resnet18"]:
                    geo_ops = geo_ops ** (1 / 4)
                    geo_lat = geo_lat ** (1 / 4)
                    geo_en = geo_en  ** (1 / 4)
                    geo_topsw = geo_topsw ** (1 / 4)  # each suit has 4 networks
                    geo_tops = geo_tops ** (1 / 4)
                    geo_topsmm2 = geo_topsmm2 ** (1 / 4)
                    geo_cf_ft = geo_cf_ft ** (1 / 4)
                    geo_cf_fw = geo_cf_fw ** (1 / 4)
                    geo_cf_ft_ex_pkg = geo_cf_ft_ex_pkg ** (1 / 4)
                    geo_cf_fw_ex_pkg = geo_cf_fw_ex_pkg ** (1 / 4)
                    geo_res = {
                        "workload": "geo",
                        "acc_type": acc_type,
                        "sram_size": sram_size,
                        "dim": dim,
                        "ops": geo_ops,
                        "area": new_res["area"],
                        "lat": None,
                        "tclk": new_res["tclk"],
                        "en": None,
                        "cf_ft": None,
                        "cf_fw": None,
                        "t_area": new_res["t_area"],
                        "t_lat": geo_lat,
                        "t_tclk": new_res["t_tclk"],
                        "t_en": geo_en,
                        "t_cf_ft": geo_cf_ft,
                        "t_cf_fw": geo_cf_fw,
                        "t_cf_ft_ex_pkg": geo_cf_ft_ex_pkg,
                        "t_cf_fw_ex_pkg": geo_cf_fw_ex_pkg,
                        "topsw": geo_topsw,
                        "tops": geo_tops,
                        "topsmm2": geo_topsmm2,
                        "fw_period_meet": geo_period_meet,
                    }
                    data_vals.append(geo_res)
    new_df = pd.DataFrame(data_vals)
    df = new_df

    # save df to pickle
    save_as_pickle(df, pkl_name)
    df = df.drop("cme", axis=1)  # delete cme to reduce file size
    save_as_pickle(df, f"no_cme_{pkl_name}")
    print(f"SIMULATION done. Turn pickle_exist to True to enable the figure display.")
    targ_time = time.time()
    sim_time = round(targ_time-trig_time, 1)
    print(f"Simulation time: {sim_time} sec.")

def experiment_1_literature_trend():
    """
    Experiment 1: plot carbon for papers in literature
    Left figure: carbon under continuous-active scenario. Right figure: carbon under periodic-active scenario (figures shown in the CarbonSpot paper)
    """
    ## Step 1: extract m factor (carbon/area) for different tech nodes. Data comes from ACT paper above.
    # plot_m_factor_across_techs()

    ## Step 2: calculate activation period, #mac/inference for tinyml benchmarks
    # ds-cnn (keyword spotting): 10 inference/second, 2_656_768 macs/inference
    # mobilenet (visual weak words): 0.75 FPS (1.3s/inference), 7_489_644 macs/inference
    # resnet8 (imagenet): 25 FPS, 12_501_632 macs/inference
    # autoencoder (anomaly detection): 1 inference/second, 264_192 macs/inference
    # geo-mean: 3.7 Hz, 10535996 macs/inference

    ## Papers
    # rows that are commented out are papers excluded due to the page limit of conference template
    data = np.array([
            #           DOI                                             Year    IMC  Tech  topsw(peak)                      topsmm2 (peak)              #ops/cycle (peak)                   area(die) (mm2)             paper_citation_idx
            #                                                                            Pres               tops(peak)                                                  area (macro) (mm2)                  targeted network size (B)
            ["10.1109/ISSCC42613.2021.9365766",                         2021, "DIMC", 22, 8, 24.7,           0.917,             0.917/0.202,              64*1024/8/8,       0.202,             None,       None,               1], ###
            # ["10.1109/ISSCC42614.2022.9731762",                         2022, "DIMC", 28, 8, 30.8,           1.35,              1.43,                     12*8*1024/8/8,     1.35/1.43,         6.69,       25.6*1024*1024,     8],   # resnet50
            # ["10.1109/ISSCC42614.2022.9731754",                         2022, "DIMC", 5,  8, 63,             2.95/4,            55,                       8*8*1024/8/8,      (2.95/4)/55,       None,       None,               23],
            ["10.1109/ISSCC42614.2022.9731645",                         2022, "DIMC", 28, 8, 12.5,           1.48,              0.221,                    24*8*1024/8/8,     1.48/0.221,        6.8266,     108*1024*1024,      2],  # BERT-Base (Transformer) ###
            # ["10.1109/JSSC.2021.3061508",                               2021, "DIMC", 65, 8, 2.06*4,         0.006*4,           0.006*4/0.2272,           2*8*1024/8/8,      0.2272,            1,          60*1024,            3],  # LeNet-5
            ["10.1109/VLSITechnologyandCir46769.2022.9830438",          2022, "DIMC", 12, 8, 30.3,           0.336,             41.6/4,                   1*8*1024/8/8,      0.336/(41.6/4),    None,       None,               3], ###
            # ["10.1109/ISSCC42614.2022.9731545",                         2022, "DIMC", 28, 8, 27.38,          0.0055,            0.0055/0.03,              4*8*1024/8/8,      0.03,              None,       None,               25],
            ["10.1109/ISSCC42614.2022.9731659",                         2022, "DIMC", 28, 8, 1108/64,        9.175/64,          9.175/64/0.033,           2*8*1024/8/8,      0.033,             1,          None,               4],
            ["10.1109/ESSCIRC59616.2023.10268774",                      2023, "DIMC", 16, 8, 23.8,           0.182,             0.364,                    128*8*1024/8/8,    0.182/0.364,       0.5056,     None,               5],  # Weijie's design
            ["10.1109/ISSCC42613.2021.9365788",                         2021, "AIMC", 16, 8, 30.25,          2.95,              0.118,                    589*8*1024/8/8,    2.95/0.118,        25,         25.6*1024*1024,     6],  # resnet50
            # ["10.1109/ESSCIRC59616.2023.10268725",                      2023, "DIMC", 28, 8, 22.4,           1.46*0.0159,       1.46,                     2*8*1024/8/8,      0.0159,            None,       None,               28],
            ["10.1109/ISSCC42615.2023.10067360",                        2023, "DIMC", 28, 8, 2.52,           3.33,              0.85,                     144*8*1024/8/8,    3.33/0.85,         3.93,       None,               7],
            # ["10.1109/ISSCC19947.2020.9062985",                         2020, "AIMC", 7,  8, 321/4,          0.455/4,           0.455/4/0.0032,           0.5*8*1024/8/8,    0.0032,            None,       None,               17],
            # ["10.1109/CICC51472.2021.9431575",                          2021, "AIMC", 22, 8, 1050/16,        23.5/16,           12.1/16,                  64*8*1024/8/8,     23.5/12.1,         4.02,       None,               18],
            ["10.1109/CICC57935.2023.10121308",                         2023, "DIMC", 28, 8, 711/2/8,        1.152/16,          1.152/16/(0.636*0.148),   1.152*8*1024/8/8, (0.636*0.148),      None,       268346/2,           8],  # 4b resnet20
            # ["10.1109/CICC57935.2023.10121243",                         2023, "AIMC", 65, 8, 1.4,            1.104/64,          1.104/64/7,               16*8*1024/8/8,     7,                 7,          268346,             5],  # 4-8b resnet20
            ["10.1109/CICC57935.2023.10121221",                         2023, "DIMC", 28, 8, 40.16*(1/0.36), 0.318,             1.25,                     32*8*1024/8/8,     0.318/1.25,        2.03,       77360,              9],
            # ["10.1109/CICC57935.2023.10121213",                         2023, "AIMC", 65, 8, 95.4*(9/64),    0.8136*(9/64),     0.8136*(9/64)/0.26,       13.5*8*1024/8/8,   0.26,              None,       None,               -1],
            # ["10.1109/JSSC.2020.3005754",                               2020, "AIMC", 55, 8, 0.6,            0.00514,           0.00514/(2.34*2.54),      0.5*8*1024/8/8,    (2.34*2.54),       5.9436,     268346/8,           2],  # binary resnet20
            ["10.23919/VLSICircuits52068.2021.9492444",                 2021, "AIMC", 28, 8, 5796/64,        6.144/64,          6.144/64/0.51,            36.8*8*1024/8/8,   0.51,              None,       8970,               10],  # custom networks
            ["https://ieeexplore.ieee.org/abstract/document/9508673",   2021, "DIMC", 28, 8, 588/64,         4.9/64,            4.9/64/20.9,              432*8*1024/8/8,    20.9,              20.9,       11.7*1024*1024/8,   11],  # resnet18
            ["10.1109/ISSCC42614.2022.9731657",                         2022, "AIMC", 28, 8, 45.7/2*3/8,     0.97/2*3/8,        0.97/2*3/8/(0.436*0.212), 170*8*1024/8/8,    (0.436*0.212),     8.702,      16*1024*1024,       12],  # YOLO-Tiny
            ["10.1109/TCSI.2023.3244338",                               2023, "AIMC", 28, 8, 16.1/4,         0.0128/4,          0.0128/4/(0.22*0.26),     2*8*1024/8/8,      (0.22*0.26),       None,       5642,               13],  # custom network
            # ["10.1109/TCSI.2023.3241385",                               2023, "AIMC", 28, 8, 942.9/4/8,      59.584/4/8*0.0385, 59.584/4/8,               2*8*1024/8/8,     0.0385,             None,       None,               21],
        ])

    ## Step 3: plot carbon cost in literature
    # mlperf-tiny
    plot_carbon_footprint_in_literature(data=data,
                                        period=10 ** 9 / 3.7,  # unit: ns
                                        op_per_task=10535996 * 2)  # unit: ops/inference (mlperf-tiny)
    # mlperf-mobile
    plot_carbon_footprint_in_literature(data=data,
                                        period=10 ** 9 / 25,  # unit: ns
                                        op_per_task=6101039461)  # unit: ops/inference (mlperf-mobile)
    # plot_area_trend_in_literature(data=data)  # hardware area trend in literature
    pass

def save_as_pickle(df, filename):
    with open(filename, "wb") as fp:
        pickle.dump(df, fp)

def read_pickle(filename="result.pkl"):
    with open(filename, "rb") as fp:
        df = pickle.load(fp)
    return df

if __name__ == "__main__":
    """
    #########################################
    ## TOP DESCRIPTION
    This file is to do carbon footprint assessment based on CarbonSpot. The carbon parameter is from ACT, which is from META.
    For details of ACT, find here the paper: Gupta, Udit, et al. "ACT: Designing sustainable computer systems with an architectural carbon modeling tool." Proceedings of the 49th Annual International Symposium on Computer Architecture. 2022.
    NOTE:
    To speed up the simulation time, the result will be saved to expr_res.pkl.
    So, you can enable pickle_exist = True, to skip the zigzag-imc call and read inputs from the pkl file.
    If simulation is required, set pickle_exist = False.
    #########################################
    """
    """
    Experiment 1: carbon for papers in literature
    """
    experiment_1_literature_trend()
    """
    Experiment 2: Carbon for different area, for AIMC, DIMC, pure digital PEs
    """
    ## Step 0: simulation parameter setting
    # workload: peak & ae from MLPerf Tiny
    # variables: cols_nbs => [32, 64, .., 1024]
    # variables: rows_nbs => rows_nbs = cols_nbs
    # variables:

    ############# fix parameters #############
    # period definition # unit: ns (geo means geometric mean of mlperf-tiny workloads; resnet18's period is set to be same as resnet8.
    # assuming 25FPS for tinyml-mobile workloads
    periods = {"peak": 1, "ae": 1e+9/1, "ds_cnn": 1e+9/10, "mobilenet": 1e+9/0.75, "resnet8": 1e+9/25, "geo":1e+9/((1*10*0.75*25)**0.25),
               "resnet18": 1e+9/25,
               "deeplabv3": 1e+9/25, "mobilebert": 1e+9/25, "mobilenet_edgetpu": 1e+9/25, "mobilenet_v2": 1e+9/25,}
    ops_workloads = {'ae': 532512, 'ds_cnn': 5609536, 'mobilenet': 15907840, 'resnet8': 25302272,  # include batch and relu
                     "deeplabv3": 10554605568, "mobilebert": 17667194880, "mobilenet_edgetpu": 1981340928, "mobilenet_v2": 3750131680,}
    size_workloads = {"ae": 264192, "ds_cnn": 22016, "mobilenet": 208112, "resnet8": 77360, "resnet18": 11678912,
                      "deeplabv3": 2238576, "mobilebert": 23004160, "mobilenet_edgetpu": 4052832, "mobilenet_v2": 16795712}  # unit: Byte
    dram_ac_cost_per_bit = 3.7  # pJ/bit
    possible_dram_energy_removal = False  # (not useful anymore, weight dram will be removed within zigzag if allowed) remove dram energy cost if on-chip weight mem size > workload size, only for weight-stationary dataflow
    d1_equal_d2 = True  # True [D1=D2=dim], False [D1=dim//8, D2=dim]

    ############# design space #############
    acc_types = ["pdigital_os", "pdigital_ws", "AIMC",
                 "DIMC"]  # pdigital_ws (pure digital, weight stationary), (pure digital, output stationary), AIMC, DIMC
    Dimensions = [2 ** x for x in range(10, 4, -1)]  # options of cols_nbs, rows_nbs
    workload_suits = ["tiny", "mobile"]  # mlperf-tiny, mlperf-mobile
    # workloads = ["peak", "ae", "ds_cnn", "mobilenet", "resnet8"]  # peak: macro-level peak  # options of workloads

    ############# simulation button #############
    pickle_exist = True  # False: start simulation; False: check output

    if pickle_exist == False:
        #########################################
        ## Simulation
        for workload_suit in workload_suits:
            if workload_suit == "tiny":
                pkl_name = "expr_res_tiny.pkl"  # this file will take 192 MB disk space
                dram_size = 1 / 1024  # 1MB. unit: GB
                workloads = ["ae", "ds_cnn", "mobilenet", "resnet8"]
                sram_sizes = [8 * 1024, 32 * 1024, 128 * 1024, 512 * 1024, 1024 * 1024]  # unit: B
            elif workload_suit == "mobile":
                pkl_name = "expr_res_mobile.pkl"  # this file will take 3 GB disk space
                # pkl_name = "expr_res_mobile_append.pkl"
                dram_size = 25 / 1024  # 20MB. unit: GB
                workloads = ["deeplabv3", "mobilebert", "mobilenet_edgetpu", "mobilenet_v2"]
                sram_sizes = [1*1024*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024, 512*1024*1024]  # unit: B
                # sram_sizes = [64 * 1024 * 1024, 512 * 1024 * 1024]
            else:  # left for debug mode
                assert Exception(f"Illegal workload suit: {workload_suit} (not tiny or mobile)")

            zigzag_similation_and_result_storage(workloads=workloads, acc_types=acc_types, sram_sizes=sram_sizes,
                                                 Dimensions=Dimensions, periods=periods, pkl_name=pkl_name,
                                                 dram_size=dram_size, dram_ac_cost_per_bit=dram_ac_cost_per_bit,
                                                 possible_dram_energy_removal=possible_dram_energy_removal,
                                                 size_workloads=size_workloads,
                                                 d1_equal_d2=d1_equal_d2, workload_suit=workload_suit)
    else:
        ## Step 1: load df from pickle
        workload_suit = "tiny"

        if workload_suit == "tiny":
            df = read_pickle("no_cme_expr_res_tiny.pkl")
            workloads = ["ae", "ds_cnn", "mobilenet", "resnet8"]  # legal workload keywords
            sram_sizes = [8 * 1024, 32 * 1024, 128 * 1024, 512 * 1024, 1024 * 1024]  # unit: B
            workloads.append("geo")  # append geo so that plotting for geo is also supported
        elif workload_suit == "mobile":
            df = read_pickle("no_cme_expr_res_mobile.pkl")
            workloads = ["deeplabv3", "mobilebert", "mobilenet_edgetpu", "mobilenet_v2"]  # legal workload keywords
            sram_sizes = [1 * 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]  # unit: B
            workloads.append("geo")  # append geo so that plotting for geo is also supported
        else:  # debugging branch
            df = read_pickle(f"expr_res_{workload_suit}.pkl")

        ## Step 2:
        #########################################
        ## Visualization (Experiment playground)
        #######################
        ## Experiment 1: sweeping imc size, fixing sram size
        # The idea is:
        # (1) there will be an optimal area point regarding carbon footprint;
        # (2) the optimum will shift when increasing the task complexity under fixed-work scenarios, but it's not true
        #       for fixed-time scenarios.
        # @para d1_equal_d2: True [D1=D2=dim], False [D1=dim//8, D2=dim]
        #######################
        ## Set the two parameters below to check specific results
        workload = "geo"
        sram_size = 512*1024
        #######################
        i_df = df[(df.workload == workload) & (df.sram_size == sram_size)]
        assert workload in workloads, f"Legal workload: {workloads}"
        assert sram_size in sram_sizes, f"Legal sram size: {sram_sizes}"
        assert workload != "peak", "The color of the plot has not been fixed when workload == peak. Now the color " \
                                   "display is in a mess order. The cause is the elements in AIMC and DIMC are " \
                                   "different to each other."
        ## (1) [check] if performance value makes sense (x axis: dimension size) (note: workload != geo)
        plot_performance_bar(i_df=i_df, acc_types=acc_types, workload=workload, sram_size=sram_size, d1_equal_d2=True, breakdown=False)
        ## (2) [check] performance together with carbon (x axis: dimension size)
        ## plot_curve below is for plotting TOPsw, TOPs, TOPsmm2, carbon curve for a fixed workload and sram size
        # plot_curve(i_df=i_df, acc_types=acc_types, workload=workload, sram_size=sram_size, d1_equal_d2=True)
        ## (3) [check] carbon breakdown of different carbon components for different area (x axis: area) (note: workload != geo)
        # plot_carbon_breakdown_in_curve(i_df=i_df, acc_types=acc_types, workload=workload, sram_size=sram_size, d1_equal_d2=True)
        ## (4) plot carbon comparison across 4 scenarios
        raw_data = {"data": df, "workloads": workloads, "periods": periods,}
        # plot_total_carbon_curve_four_cases(i_df=i_df, acc_types=acc_types, workload=workload, sram_size=sram_size, complexity=13, raw_data=raw_data, plot_breakdown=True, d1_equal_d2=True)
        ## (5) plot carbon in scatter across 4 scenarios (can include the sweep for different sram size)
        # scatter_carbon_cost_four_cases(acc_types=acc_types, sram_sizes=sram_sizes, workload=workload, complexity=50, raw_data=raw_data, d1_equal_d2=True, active_plot=True)
        acc_types = ["pdigital_os", "DIMC"]
        scatter_performance(acc_types=acc_types, sram_sizes=sram_sizes, workload=workload, raw_data=raw_data, d1_equal_d2=True, active_plot=False)
        # breakpoint()

        #######################
        ## Experiment 2: sweeping sram size, fixing imc size
        # The idea is:
        # (1) the optimum for energy and carbon is not the same, where big on-chip memory is preferred from the energy
        #       perspective but small on-chip memory is preferred from the carbon perspective.
        #######################
        ## Set the two parameters below to check specific results
        # workload = "mobilenet"
        # imc_dim = 128
        #######################
        # assert imc_dim in Dimensions, f"Legal dimensions: {Dimensions}"
        # assert workload in workloads, f"Legal workload: {workloads}"
        # assert workload != "peak", "The color of the plot has not been fixed when workload == peak. Now the color " \
        #                            "display is in a mess order. The cause is the elements in AIMC and DIMC are " \
        #                            "different to each other."
        # i_df = df[(df.workload == workload) & (df.dim == imc_dim)]
        #
        # sram_size = [64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024]
        ## (1) check if performance value makes sense (x axis: dimension size) (note: workload != geo)
        # plot_performance_bar(i_df=i_df, acc_types=acc_types, workload=workload, sram_size=sram_size, d1_equal_d2=True, breakdown=True)
        ## (4) plot carbon comparison across 4 scenarios
        # raw_data = {
        #     "data": df,
        #     "workloads": workloads,
        #     "periods": periods,
        #     "dim": imc_dim,
        # }
        # plot_total_carbon_curve_four_cases(i_df=i_df, acc_types=acc_types, workload=workload, sram_size=sram_size, complexity=50, raw_data=raw_data, plot_breakdown=True, d1_equal_d2=True)
        # breakpoint()
