from __future__ import annotations

import os
import sys
import json
import shutil
import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from multiprocessing import Pool


from wbnci.preprocessing import read_country_list
from wbnci.reports import (make_summary_tables, make_lulc_legend,
                           make_reference_point_lulc_maps, get_nearest_max_id, 
                           get_overall_extreme_ids, get_pareto_extreme_ids,
                           make_frontier_plot, make_activity_maps)
from wbnci.agreement_maps import make_agreement_maps



BASELINE = 0
MINIMIZATION = -1
MAXIMIZATION = 1


def main(config_file):

    with open(config_file, "r") as f:
        args = yaml.safe_load(f)

    workspace = args["workspace"]
    if os.path.isfile(args["country_list"]):
        countries = read_country_list(args["country_list"])
    else:
        countries = [c.strip() for c in args["country_list"].split(",")]
    
    if "n_workers" in args:
        n_workers = int(args["n_workers"])
    else:
        n_workers = 12

    if "optimization_scenarios" in args:
        optimization_scenarios = args["optimization_scenarios"]
    else:
        optimization_scenarios = [
            "extensification_bmps_irrigated",
            "extensification_bmps_rainfed",
            "extensification_current_practices",
            "extensification_intensified_irrigated",
            "extensification_intensified_rainfed",
            "fixedarea_bmps_irrigated",
            "fixedarea_bmps_rainfed",
            "fixedarea_intensified_irrigated",
            "fixedarea_intensified_rainfed",
            "forestry_expansion",
            "grazing_expansion",
            "restoration",
            "sustainable_current",
        ]
    
    if "value_columns" in args:
        value_columns = args["value_columns"]
    else:
        value_columns = [
            "net_econ_value",
            "biodiversity",
            "net_ghg_co2e",
            "carbon",
            "cropland_value",
            "forestry_value",
            "grazing_value",
            "grazing_methane",
            "production_value",
            "transition_cost",
            "nitrate_cancer_cases",
            "ground_noxn",
            "noxn_in_drinking_water",
            "surface_noxn"
        ]

    tgdb = os.path.join(workspace, "taskgraph_data.db")
    if os.path.isfile(tgdb):
        os.remove(tgdb)

    pool = Pool()
    tasks = []
    for country in countries:
        print(country)
        country_folder = os.path.join(workspace, country)
        # postprocess_country(country_folder, args, optimization_scenarios, value_columns)
        tasks.append(pool.apply_async(postprocess_country, [country_folder, args, optimization_scenarios, value_columns]))
    for t in tasks:
        t.wait()


def postprocess_country(country_folder: str, args: dict, optimization_scenarios: list[str], value_columns: list[str]):
    """
    Generate the results for `country_folder`.
    """
    # Make summary tables
    nci_columns = ["econ", "biodiversity", "net_ghg_co2e"]   # note that "econ" is built inside `make_summary_tables`
    make_summary_tables(country_folder, value_columns, nci_columns)
    results_file = os.path.join(country_folder, "OptimizationResults", "solutions.h5")

    target_folder = os.path.join(country_folder, "OptimizationResults", "FiguresAndMaps")
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    sdu_file = os.path.join(country_folder, "sdu_map", "sdu.shp")
    reference_lulc_file = os.path.join(country_folder, "InputRasters", "current_lulc.tif")
    scenario_folder = os.path.join(country_folder, "ScenarioMaps")

    # FIND REFERENCE POINTS
    pareto_vars = ["production_value", "biodiversity", "net_ghg_co2e"]
    minimize_vars = []
    reference_points = {}
    reference_points["overall"] = get_overall_extreme_ids(
        results_file, pareto_vars, minimize_vars=minimize_vars)
    reference_points["pareto"] = get_pareto_extreme_ids(
        results_file, pareto_vars, minimize_vars=minimize_vars)
    reference_points["nearest"] = get_nearest_max_id(results_file)
        
    # MAKE FRONTIER PLOT
    econ_col = "production_value"
    non_econ_cols = ["biodiversity", "net_ghg_co2e", "noxn_in_drinking_water"]
    make_frontier_plot(country_folder, target_folder, econ_col, non_econ_cols, reference_points)

    # MAKE MAPS FOR REFERENCE POINTS
    lu_map_folder = os.path.join(target_folder, "LU Maps")
    make_reference_point_lulc_maps(country_folder, lu_map_folder, reference_points, sdu_file,
                                   reference_lulc_file, optimization_scenarios)
    make_lulc_legend(lu_map_folder)

    # MAKE ACTIVITY MAPS
    make_activity_maps(country_folder, args["lu_table_file"])
    
    # MAKE AGREEMENT MAPS
    make_agreement_maps(country_folder)
    



###### UNUSED

def make_water_quality_plot(country_dir):
    table_file = os.path.join(country_dir, "OptimizationResults", "merged_summary_table.csv")
    if not os.path.isfile(table_file):
        return
    df = pd.read_csv(table_file)
    curr = df[df["sense"] == 0]
    smax = df[df["sense"] == 1]
    
    wq_cols = ["ground_noxn", "noxn_in_drinking_water", "surface_noxn", "nitrate_cancer_cases"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    for i, col in enumerate(wq_cols):
        r = int(i/2)
        c = np.mod(i, 2)
        ax = axes[r][c]
        ax.scatter(smax["production_value"], smax[col], facecolor="blue", alpha=0.3)
        ax.scatter(curr["production_value"], curr[col], facecolor="orange", edgecolor="black")
        ax.set_title(col)
        ax.set_xlabel("prod_value")
        ax.set_ylabel(col)
    
    plt.savefig(os.path.join(country_dir, "OptimizationResults", "ToShare", "wq_plot.pdf"), bbox_inches="tight")


def upload_country_files(country):
    src_root = "/Users/peterhawthorne/Projects/WBNCI/scratch-may-2020/packages"
    dst_root = "/Volumes/GoogleDrive/Shared drives/NCI/May 2020 results/May24/CountryResults"
    or_src = os.path.join(src_root, country, "OptimizationResults_May24")
    or_dst = os.path.join(dst_root, country, "OptimizationResults_May24")
    if not os.path.exists(or_dst):
        os.makedirs(or_dst)
    or_files = [
        os.path.join(or_src, "merged_summary_table.csv"),
        os.path.join(or_src, "solutions.h5")]
    for src in or_files:
        dst = os.path.join(or_dst, os.path.basename(src))
        shutil.copy(src, dst)


def upload_summary_table():
    src_root = "/Users/peterhawthorne/Projects/WBNCI/scratch-may-2020/packages"
    dst_root = "/Volumes/GoogleDrive/Shared drives/NCI/May 2020 results/Preliminary"
    basename = "nci_summary_table.csv"
    shutil.copy(os.path.join(src_root, basename),
                os.path.join(dst_root, basename))



# MAIN
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Usage: python preprocess.py [config file path]")
    
    if not os.path.isfile(sys.argv[1]):
        raise Exception(f"Error: config file {sys.argv[1]} does not exist")
    
    main(sys.argv[1])
    
