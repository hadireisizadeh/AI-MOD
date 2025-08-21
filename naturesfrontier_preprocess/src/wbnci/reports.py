from __future__ import annotations

import os
import sys
import json
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import osgeo.gdal as gdal
import h5py
import pygeoprocessing.geoprocessing as geo
import taskgraph
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .solution_mapping import (raster_from_solution_archive, lulc_tif_to_png, 
                               rasterize_sdu_shapefile, make_legend, lulc_to_activity,
                               make_activity_legend)
from .utils import get_headers


BASELINE = 0
MINIMIZATION = -1
MAXIMIZATION = 1

SOL_COLORS = {
    "nearest": "tomato",
    "extreme": "midnightblue",
    "production_value": "gold",
    "net_econ_value": "gold",
    "biodiversity": "green",
    "net_ghg_co2e": "mediumorchid",
    "noxn_in_drinking_water": "cornflowerblue"
}

VAR_LABELS = {
    "net_econ_value": "Net economic value (billion $)",
    "production_value": "Production value (billion $)",
    "biodiversity": "Biodiversity",
    "net_ghg_co2e": "CO2e",
    "noxn_in_drinking_water": "[N] drinking water",
}

LEGEND_LABELS = {
    "current": "Current",
    "nearest": "Nearest",
    "extreme": "Extreme",
    "net_econ_value": "Net $ value",
    "production_value": "Production value",
    "biodiversity": "Biodiversity",
    "net_ghg_co2e": "CO2e",
    "noxn_in_drinking_water": "[N] drinking water"
}


def summarize_value_tables(country_root: str) -> pd.DataFrame:
    """
    Create a table that sums SDU values for each metric and each scenario. Structure is scenarios as
    rows, metrics as columns. Returns with "management" column as index.
    """
    vtroot = os.path.join(country_root, "ValueTables")

    value_tables = sorted(glob.glob(os.path.join(vtroot, "*.csv")))
    cols = ["management"] + get_headers(value_tables[0], drop_cols=["SDUID"])
    summary = pd.DataFrame(columns=cols)
    summary.set_index("management", inplace=True)
    
    for table_file in value_tables:
        management = os.path.splitext(os.path.basename(table_file))[0]
        df = pd.read_csv(table_file)
        for col in cols[1:]:
            summary.loc[management, col] = sum(df[col])
    
    return summary


def merge_baseline_and_optimizations(country_folder: str, value_columns: list[str],
                                     optimization_folder="OptimizationResults",
                                     include_minimizations=False) -> pd.DataFrame:
    """
    Merge sustainable_current value table with outputs from optimizations. Adds "sense" column
    that indicates 0 -> Baseline, 1 -> Maximization, -1 -> Minimization.
    """
    optim_folder = os.path.join(country_folder, optimization_folder)

    baseline_value_table_file = os.path.join(country_folder, "ValueTables", "sustainable_current.csv")
    base_df = pd.read_csv(baseline_value_table_file)
    base_sum = base_df[value_columns].sum().to_frame().transpose()
    base_sum["sense"] = BASELINE
    base_sum["ID"] = 1
    
    max_df = pd.read_csv(os.path.join(optim_folder, "summary_table_maximize.csv"))
    max_df["sense"] = MAXIMIZATION
    df = pd.concat([base_sum, max_df]).reset_index()
    if include_minimizations:
        min_df = pd.read_csv(os.path.join(optim_folder, "summary_table_minimize.csv"))
        min_df["sense"] = MINIMIZATION
        df = pd.concat([df, min_df]).reset_index()
    
    # merge data tables
    df = df[["ID", "sense"] + value_columns]
    
    return df


def make_summary_tables(country_folder: str, value_columns: list[str], nci_columns: list[str],
                        include_minimizations=False, optimization_folder="OptimizationResults") -> None:
    """
    Create:
        - a table (value_table_summary.csv) containing the summed SDU values for each 
        homogeneous management scenario. 
        - a table (`optimization_folder`/merged_summary_table.csv) summarizing the scores
        for the baseline and optimization solutions. For this table, also calculate normed
        values, distance from baseline to each point, and nearest point ID.
    
    Normed values: 
        - For most columns, we norm based on range from zero to max
        - For biodiversity we norm based on two alternative "worst case" scenarios, one
        where all land is flipped to urban, one where all economically profitable conversions
        are made. 
    
    Note that for the maximization component of the frontier, we count `net_econ_value` as the
    economic objective, while for the minimization frontier we count `ag_value` (ignoring
    `transition_cost` as a component of `net_econ_value`). This means we have to juggle some column
    names to get the different results we need. 
    """
    
    optim_folder = os.path.join(country_folder, optimization_folder)

    # calculate and save the summed SDU values for each homogeneous management scenario
    scenario_sums = summarize_value_tables(country_folder)
    scenario_sums.to_csv(os.path.join(country_folder, "value_table_summary.csv"))

    # load baseline and optimization scenarios
    df = merge_baseline_and_optimizations(
        country_folder, value_columns,
        include_minimizations=include_minimizations,
        optimization_folder=optimization_folder)
    
    # Fill "econ" column depending on whether the solution is for maximization or
    # minimization frontier. Max gets net, min gets gross
    def econ(net, ag, sense):
        if sense == MAXIMIZATION:
            return net
        else:
            return ag
    df["econ"] = [econ(net, ag, sense) for net, ag, sense in zip(
        df["net_econ_value"], df["production_value"], df["sense"])]

    for col in value_columns+["econ"]:
        if col == "noxn_in_drinking_water":
            # special case where we want to flip
            df[f"{col}_normed"] = (df[col].max() - df[col])/(df[col].max() - df[col].min())
        else:
            # default case
            df[f"{col}_normed"] = df[col] / max(df[col])
    
    # Add additional biodiversity normed vals
    bref = scenario_sums.loc["all_urban", "biodiversity"]
    df["biodiversity_normed_urban"] = (df["biodiversity"] - bref) / (max(df["biodiversity"]) - bref)
    bref = scenario_sums.loc["all_econ", "biodiversity"]
    df["biodiversity_normed_econ"] = (df["biodiversity"] - bref) / (max(df["biodiversity"]) - bref)
    
    # calculate distance from baseline for each point (in normed coords)
    base_pt = df[df["sense"] == BASELINE].loc[0].to_dict()
    norm_cols = [f"{col}_normed" for col in nci_columns]
    norm_cols_urban = norm_cols.copy()
    norm_cols_urban[1] = "biodiversity_normed_urban"
    norm_cols_econ = norm_cols.copy()
    norm_cols_econ[1] = "biodiversity_normed_econ"
    
    distances = np.zeros(len(df))
    distances_urban = np.zeros(len(df))
    distances_econ = np.zeros(len(df))
    
    for i, row in df.iterrows():
        distances[i] = np.sqrt(sum([(base_pt[c] - row[c]) ** 2 for c in norm_cols]))
        distances_urban[i] = np.sqrt(sum([(base_pt[c] - row[c]) ** 2 for c in norm_cols_urban]))
        distances_econ[i] = np.sqrt(sum([(base_pt[c] - row[c]) ** 2 for c in norm_cols_econ]))
    
    df["distance"] = distances
    df["distance_urban"] = distances_urban
    df["distance_econ"] = distances_econ

    # Identify Pareto points
    pareto_columns = ["biodiversity", "net_ghg_co2e", "net_econ_value", "noxn_in_drinking_water"]
    ecb_cols = ["biodiversity", "net_ghg_co2e", "net_econ_value"]
    pareto_direction = [1, 1, 1, -1]
    baseline_values = {
        col: df[col][0] for col in pareto_columns
    }
    pareto_ecbw = np.ones(len(df), dtype=int)
    pareto_ecb = np.ones(len(df), dtype=int)
    pareto_dims = np.zeros(len(df), dtype=int)
    for col, dir in zip(pareto_columns, pareto_direction):
        pareto_ecbw = np.all([
            pareto_ecbw,
            dir * df[col] > dir * baseline_values[col]
        ], axis=0)
        if col in ecb_cols:
            pareto_ecb = np.all([
                pareto_ecb,
                dir * df[col] > dir * baseline_values[col]
            ], axis=0)

        pareto_dims += dir * df[col] > dir * baseline_values[col]
    df["pareto_ecbw"] = pareto_ecbw
    df["pareto_ecb"] = pareto_ecb
    df["pareto_dims"] = pareto_dims

    df.to_csv(os.path.join(optim_folder, "merged_summary_table.csv"), index=False)
    sol_file = os.path.join(optim_folder, "solutions.h5")
    df.to_hdf(sol_file, key="summary_table")
    
    # calculate the NCI
    ddf = df[["sense", "distance"]].groupby("sense").min()
    alpha = ddf.loc[MAXIMIZATION, "distance"]
    nearest_max_pts = df[(df["sense"] == MAXIMIZATION) & (df["distance"] == alpha)]
    nearest_max_id = int(list(nearest_max_pts["ID"])[0])
    
    s = h5py.File(sol_file, "a")
    t = s["summary_table"]
    t.attrs["nearest_max_id"] = nearest_max_id

    if include_minimizations:
        beta = ddf.loc[MINIMIZATION, "distance"]
        nearest_min_pts = df[(df["sense"] == MINIMIZATION) & (df["distance"] == beta)]
        nearest_min_id= int(list(nearest_min_pts["ID"])[0])
        t.attrs["nearest_min_id"] = nearest_min_id


def make_lulc_legend(map_folder):
    legend_file = os.path.join(map_folder, "pngs", "legend.pdf")
    make_legend(legend_file)


def make_reference_point_lulc_maps(country_folder, target_folder, reference_points, 
                                   sdu_file, reference_lulc_file, optimization_scenarios,
                                   optimization_folder="OptimizationResults"):
    """
    Make maps of LULC for the scenarios in `reference_points`, which should have keys 
    "overall", "pareto", and "nearest".
    """
    scenario_folder = os.path.join(country_folder, "ScenarioMaps")
    results_file = os.path.join(country_folder, optimization_folder, "solutions.h5")
    sdu_raster = os.path.join(os.path.dirname(sdu_file), "sdu.tif")

    if not os.path.isfile(sdu_raster):
        rasterize_sdu_shapefile(sdu_file, reference_lulc_file)
    
    sdu = gpd.read_file(sdu_file)
    sdu_list = list(sdu['SDUID'])
    
    # Make map images
    tif_dir = os.path.join(target_folder, 'tifs')
    png_dir = os.path.join(target_folder, 'pngs')

    for d in [tif_dir, png_dir]:
        if not os.path.isdir(d):
            os.makedirs(d)
    
    # for overall "corners"
    for var, ptid in reference_points["overall"].items():
        tif_file = os.path.join(tif_dir, f'overall_max_{var}.tif')
        png_file = os.path.join(png_dir, f'overall_max_{var}.png')
        raster_from_solution_archive(results_file, ptid-1, sdu_list,
                                    sdu_raster, scenario_folder, 
                                    optimization_scenarios, tif_file)
        lulc_tif_to_png(tif_file, dstfile=png_file)

    # for Pareto-space "corners"
    for var, ptid in reference_points["pareto"].items():
        tif_file = os.path.join(tif_dir, f'pareto_max_{var}.tif')
        png_file = os.path.join(png_dir, f'pareto_max_{var}.png')
        raster_from_solution_archive(results_file, ptid-1, sdu_list,
                                    sdu_raster, scenario_folder, 
                                    optimization_scenarios, tif_file)
        lulc_tif_to_png(tif_file, dstfile=png_file)
    
    # for nearest point
    tif_file = os.path.join(tif_dir, "nearest.tif")
    png_file = os.path.join(png_dir, "nearest.png")
    nearest_max_id = reference_points["nearest"]
    raster_from_solution_archive(results_file, nearest_max_id-1, sdu_list,
                                    sdu_raster, scenario_folder, 
                                    optimization_scenarios, tif_file)
    lulc_tif_to_png(tif_file, dstfile=png_file)
    
    # for sustainable current
    srcfile = os.path.join(scenario_folder, 'sustainable_current.tif')
    dstfile = os.path.join(png_dir, 'sustainable_current.png')
    lulc_tif_to_png(srcfile, dstfile=dstfile, needs_color_table=True)


def get_pareto_extreme_ids(results_file, pareto_vars, minimize_vars=[], extra_vars={}):
    """
    Find the "corner" points for each of the `pareto_vars`. 

    Also (optionally) find the points that maximize/minimize the variables in `extra_vars`, which 
    should be a dict like `{variable: "max" | "min"}`
    """
    df = pd.read_hdf(results_file, 'summary_table')
    basedf = df[df['sense'] == 0]
    maxdf = df[df['sense'] == 1].copy()
    maxdf.fillna(value=0, inplace=True)
    
    pareto_point = np.ones(len(maxdf))
    for var in pareto_vars:
        if var in minimize_vars:
            pareto_point = pareto_point * (maxdf[var] <= basedf.loc[0, var])
        else:
            pareto_point = pareto_point * (maxdf[var] >= basedf.loc[0, var])

    maxdf['pareto'] = pareto_point
    
    pardf = maxdf[maxdf['pareto'] == 1]
    
    pareto_extreme_ids = {}
    for var in pareto_vars:
        extreme_val = pardf[var].max() if var not in minimize_vars else pardf[var].min()
        if np.isnan(extreme_val):
            extreme_pt_id = 1
        else:
            extreme_pt_id = int(pardf[pardf[var] == extreme_val]['ID'].min())
        pareto_extreme_ids[var] = extreme_pt_id
    
    for var, dir in extra_vars.items():
        if dir not in ["max", "min"]:
            raise ValueError("extra_vars direction must be 'max' or 'min'")
        extreme_val = pardf[var].max() if dir=="max" else pardf[var].min()
        if np.isnan(extreme_val):
            extreme_pt_id = 1
        else:
            extreme_pt_id = int(pardf[pardf[var] == extreme_val]['ID'].min())
        pareto_extreme_ids[var] = extreme_pt_id

    return pareto_extreme_ids


def get_overall_extreme_ids(results_file, objective_vars, minimize_vars=[], extra_vars={}):
    df = pd.read_hdf(results_file, 'summary_table')
    basedf = df[df['sense'] == 0]
    maxdf = df[df['sense'] == 1].copy()
    
    extreme_ids = {}
    for var in objective_vars:
        extreme_val = maxdf[var].max() if var not in minimize_vars else maxdf[var].min()
        extreme_pt_id = int(maxdf[maxdf[var] == extreme_val]['ID'].min()) # take the min because there might be more than one
        extreme_ids[var] = extreme_pt_id
    
    for var, sense in extra_vars.items():
        extreme_val = maxdf[var].max() if sense=="max" else maxdf[var].min()
        extreme_pt_id = int(maxdf[maxdf[var] == extreme_val]['ID'].min()) # take the min because there might be more than one
        extreme_ids[var] = extreme_pt_id

    return extreme_ids


def get_nearest_max_id(results_file):
    with h5py.File(results_file, 'r') as s:
        t = s['summary_table']
        nearest_max_id = t.attrs['nearest_max_id']
    return nearest_max_id


def make_nci_table(country_name, df, target_folder, econ_col, non_econ_cols,
                   overall_extreme_ids, pareto_extreme_ids, nearest_max_id):
    """
    Calculate % of Max, % of Pareto Max measured relative to minimum value and to zero.
    """
    basedf = df[df['sense'] == BASELINE]
    maxdf = df[df['sense'] == MAXIMIZATION].set_index('ID')
    
    objectives = [econ_col] + non_econ_cols
        
    output_file = os.path.join(target_folder, "nci_scores.csv")
    with open(output_file, 'w') as f:
        f.write("country,objective,pct_max,pct_pareto_max,pct_max_zero,pct_pareto_max_zero\n")
        for objective in objectives:
            # Percent of maximum relative to minimum value 
            if objective == "nitrate_cancer_cases":
                pct_max = 1 - basedf[f"{objective}_normed"][0]
            else:
                pct_max = basedf[f"{objective}_normed"][0]
            
            # Percent of maximum in Pareto space
            ptid = pareto_extreme_ids[objective]
            if objective == "nitrate_cancer_cases":
                pct_pareto_max = (1-basedf[f"{objective}_normed"][0]) / (1-maxdf.loc[ptid, f"{objective}_normed"])
            else:
                pct_pareto_max = basedf[f"{objective}_normed"][0] / maxdf.loc[ptid, f"{objective}_normed"]

            # Percent of maximum relative zero 
            if objective == "nitrate_cancer_cases":
                pct_max_zero = (max(maxdf[objective]) - basedf[objective][0]) / (max(maxdf[objective]) - min(maxdf[objective]))
            else:
                pct_max_zero = basedf[objective][0] / max(maxdf[objective])
            
            # Percent of maximum in Pareto space relative to zero
            ptid = pareto_extreme_ids[objective]
            if objective == "nitrate_cancer_cases":
                print(basedf)
                print(f"baseline: {basedf[objective][0]}")
                print(f"pareto: {maxdf.loc[ptid, objective]}")
                print(f"maximum: {max(maxdf[objective])}")
                print(f"minimum: {min(maxdf[objective])}")
                
                pct_pareto_max_zero = (max(maxdf[objective]) - basedf[objective][0]) / (max(maxdf[objective]) - maxdf.loc[ptid,objective])
            else:
                pct_pareto_max_zero = basedf[objective][0] / maxdf.loc[ptid, objective]

            
            f.write(f"{country_name},{objective},{pct_max},{pct_pareto_max},{pct_max_zero},{pct_pareto_max_zero}\n")



def make_frontier_plot(country_folder: str, output_file: str, econ_col: str, 
                       non_econ_cols: list[str], reference_points: dict,
                       suffix: str = "", suptitle: str|None = None, df=None,
                       optimization_folder="OptimizationResults",
                       **kwargs):
    """
    Plots each of non_econ_cols against econ_col.
    Saves to output_folder/econ_vs_non_econ{suffix}.{kwargs["fmt"]}
    
    optional kwargs:
        - `dpi`: specify the output file dpi (default: 200)
        - `fmt`: specify the output file type (default: "png")
    """

    results_file = os.path.join(country_folder, optimization_folder, "solutions.h5")
    overall_extreme_ids = reference_points["overall"]
    pareto_extreme_ids = reference_points["pareto"]
    nearest_max_id = reference_points["nearest"]

    if df is None:
        df = pd.read_hdf(results_file, 'summary_table')

    df[econ_col] = df[econ_col] / 1000000000
    
    basedf = df[df['sense'] == BASELINE]
    maxdf = df[df['sense'] == MAXIMIZATION].set_index('ID')
    ecb_pts = maxdf[maxdf["pareto_ecb"] == True]
    # mindf = df[df['sense'] == MINIMIZATION]
    
    corner_pts = list(overall_extreme_ids.values())
    cornerdf = maxdf[maxdf.index.isin(corner_pts)]
    n_non_econ = len(non_econ_cols)
    
    if suptitle:
        height = 4.5
    else:
        height = 4

    fig, axes = plt.subplots(1, n_non_econ, figsize=(4.5*n_non_econ+2.5, height))

    if suptitle:
        sup = fig.suptitle(suptitle, size='xx-large', weight='semibold')
    else:
        sup = None

    # colors = ["#cccccc", "#feebe2","#fbb4b9","#f768a1","#ae017e"]
    colors = ["#cccccc", '#edf8fb','#b3cde3','#8c96c6','#88419d']

    for ax, ne_col in zip(axes, non_econ_cols):
        for i in range(1, 5):
            parpts = maxdf[maxdf["pareto_dims"] == i]
            ax.scatter(parpts[econ_col], parpts[ne_col], c=colors[i], s=25, alpha=0.8,
                       linewidth=0, label=f"Pareto for {i} objs")
        # ax.scatter(maxdf[econ_col], maxdf[ne_col], c="blue", s=25, alpha=0.6, linewidth=0)
        # ax.scatter(ecb_pts[econ_col], ecb_pts[ne_col], facecolor="none", s=25, edgecolor="black", linewidth=0.1)
        ax.scatter(basedf[econ_col], basedf[ne_col], c="orange", edgecolor="black", s=64, label="Current")
        ax.set_title(ne_col.capitalize().replace("_", " "))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
        ax.set_xlabel(VAR_LABELS[econ_col])
        ax.set_ylabel(VAR_LABELS[ne_col])
        
        ax.scatter([maxdf.loc[nearest_max_id, econ_col]], [maxdf.loc[nearest_max_id, ne_col]],
                    c=SOL_COLORS['nearest'], edgecolor='black', s=36, label='Nearest')
        # ax.scatter(cornerdf[econ_col], cornerdf[ne_col],
        #             c=SOL_COLORS['extreme'], edgecolor='black', s=36, label='Extreme')
        for var, ptid in pareto_extreme_ids.items():
            ax.scatter([maxdf.loc[ptid, econ_col]], [maxdf.loc[ptid, ne_col]],
                       c=SOL_COLORS[var], edgecolor='black', s=36, label=LEGEND_LABELS[var])
        
        if ne_col == non_econ_cols[0]:
            leg = ax.legend(
                title="Highlighted\nscenarios",
                loc='center left',
                bbox_to_anchor=(-0.7, 0.5))
    
    # process output kwargs
    dpi = kwargs["dpi"] if "dpi" in kwargs else 200
    fmt = kwargs["fmt"] if "fmt" in kwargs else "png"
    artists = [leg]
    if sup is not None:
        artists.append(sup)
    
    plt.savefig(output_file, dpi=dpi, 
                bbox_extra_artists=artists, bbox_inches='tight')
    plt.close()
    

def make_activity_maps(country_folder: str, lu_table_file: str, optimization_folder="OptimizationResults") -> None:
    """
    Needs to be run after `make_summary_tables`
    """
    lulc_map_folder = os.path.join(country_folder, optimization_folder, "FiguresAndMaps", "LU Maps", "tifs")
    lulc_maps = glob.glob(os.path.join(lulc_map_folder, "*.tif"))
    lulc_maps.append(os.path.join(country_folder, "ScenarioMaps", "sustainable_current.tif"))
    
    output_folder = os.path.join(country_folder, optimization_folder, "FiguresAndMaps", "LU Category Maps")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    legend_file = os.path.join(output_folder, "activity_legend.png")
    make_activity_legend(legend_file)
    
    for m in lulc_maps:
        target_file = os.path.join(output_folder, os.path.basename(m))
        lulc_to_activity(m, lu_table_file, target_file)
