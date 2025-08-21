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
            # "nitrate_cancer_cases",
            # "ground_noxn",
            "noxn_in_drinking_water",
            # "surface_noxn"
        ]

    tgdb = os.path.join(workspace, "taskgraph_data.db")
    if os.path.isfile(tgdb):
        os.remove(tgdb)

    for country in countries:
        print(country)
        country_folder = os.path.join(workspace, country)
        postprocess_country(country_folder, args, optimization_scenarios, value_columns)


def postprocess_country(country_folder: str, args: dict, optimization_scenarios: list[str], value_columns: list[str]):
    """
    Generate the results for `country_folder`.
    """
    country_name = os.path.basename(country_folder)

    if "optimization_output_folder" in args:
        results_folder_name = args["optimization_output_folder"]
    else:
        results_folder_name = "OptimizationResults"
    print(results_folder_name)

    # Make summary tables
    nci_columns = ["econ", "biodiversity", "net_ghg_co2e"]   # note that "econ" is built inside `make_summary_tables`
    make_summary_tables(country_folder, value_columns, nci_columns, optimization_folder=results_folder_name)
    results_file = os.path.join(country_folder, results_folder_name, "solutions.h5")

    target_folder = os.path.join(country_folder, results_folder_name, "FiguresAndMaps")
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    sdu_file = os.path.join(country_folder, "sdu_map", "sdu.shp")
    reference_lulc_file = os.path.join(country_folder, "InputRasters", "current_lulc.tif")
    scenario_folder = os.path.join(country_folder, "ScenarioMaps")

    # FIND REFERENCE POINTS
    # pareto_vars = ["net_econ_value", "biodiversity", "net_ghg_co2e"]
    # minimize_vars = []
    pareto_vars = args["objectives"]
    minimize_vars = []
    if "minimize_vars" in args:
        for i in args["minimize_vars"]:
            minimize_vars.append(pareto_vars[i-1])
    if "noxn_in_drinking_water" in pareto_vars:
        extra_vars = {}
    else:
        extra_vars={"noxn_in_drinking_water": "min"}
    
    reference_points = {}
    reference_points["overall"] = get_overall_extreme_ids(
        results_file, pareto_vars, minimize_vars=minimize_vars,
        extra_vars=extra_vars)
    reference_points["pareto"] = get_pareto_extreme_ids(
        results_file, pareto_vars, minimize_vars=minimize_vars,
        extra_vars=extra_vars)
    reference_points["nearest"] = get_nearest_max_id(results_file)
        
    # MAKE FRONTIER PLOT
    econ_col = "net_econ_value"
    # econ_col = "production_value"
    non_econ_cols = ["biodiversity", "net_ghg_co2e", "noxn_in_drinking_water"]
    target_file = os.path.join(target_folder, "econ_vs_non_econ.png")
    make_frontier_plot(country_folder, target_file, econ_col, non_econ_cols, 
                       reference_points, suptitle=country_name,
                       optimization_folder=results_folder_name)

    # # MAKE MAPS FOR REFERENCE POINTS
    lu_map_folder = os.path.join(target_folder, "LU Maps")
    make_reference_point_lulc_maps(country_folder, lu_map_folder, reference_points, sdu_file,
                                   reference_lulc_file, optimization_scenarios,
                                   optimization_folder=results_folder_name)
    make_lulc_legend(lu_map_folder)

    # # MAKE ACTIVITY MAPS
    # make_activity_maps(country_folder, args["lu_table_file"], optimization_folder=results_folder_name)
    
    # # MAKE AGREEMENT MAPS
    # make_agreement_maps(country_folder, optimization_folder=results_folder_name)
    


# MAIN
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Usage: python preprocess.py [config file path]")
    
    if not os.path.isfile(sys.argv[1]):
        raise Exception(f"Error: config file {sys.argv[1]} does not exist")
    
    main(sys.argv[1])
    
