from itertools import count
import os
import sys
import shutil
import yaml
from wbnci.preprocessing import read_country_list


def main(config_file):

    with open(config_file, "r") as f:
        args = yaml.safe_load(f)

    workspace = args["workspace"]
    if os.path.isfile(args["country_list"]):
        countries = read_country_list(args["country_list"])
    else:
        countries = [c.strip() for c in args["country_list"].split(",")]

    frontier_folder = os.path.join(workspace, "_frontiers")
    if not os.path.isdir(frontier_folder):
        os.makedirs(frontier_folder)
    summary_folder = os.path.join(workspace, "_summaries")
    if not os.path.isdir(summary_folder):
        os.makedirs(summary_folder)

    for country in countries:
        # copy frontier
        src = os.path.join(workspace, country, "OptimizationResults",
                           "FiguresAndMaps", "econ_vs_non_econ.png")
        dst = os.path.join(frontier_folder, f"{country}.png")
        shutil.copyfile(src, dst)
        # copy summary_table.csv
        src = os.path.join(workspace, country, "OptimizationResults", "merged_summary_table.csv")
        dst = os.path.join(summary_folder, f"{country}.csv")
        shutil.copyfile(src, dst)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Usage: python preprocess.py [config file path]")
    
    if not os.path.isfile(sys.argv[1]):
        raise Exception(f"Error: config file {sys.argv[1]} does not exist")
    
    main(sys.argv[1])
    
