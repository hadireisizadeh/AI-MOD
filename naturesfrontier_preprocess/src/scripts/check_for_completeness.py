import os
import glob
import sys


workspace = sys.argv[1]

country_folders = sorted([d for d in glob.glob(os.path.join(workspace, "*")) if os.path.isdir(d)])

files_to_check = [
    "OptimizationResults/merged_summary_table.csv",
    "OptimizationResults/solutions.h5",
    "OptimizationResults/FiguresAndMaps/econ_vs_non_econ.png",
    "OptimizationResults/FiguresAndMaps/Agreement Maps/modal_sol.png",
    "OptimizationResults/FiguresAndMaps/LU Maps/pngs/nearest.png"
]


for cf in country_folders:
    for f in files_to_check:
        if not os.path.isfile(os.path.join(cf, f)):
            print(cf)


