import yaml
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.data_handler import DataHandler
import numpy as np


script_dir = Path(__file__).parent.absolute()
config_file = script_dir / "config.yml"
print(f"Reading config from: {config_file}")

with open(config_file, 'r') as f:
    args = yaml.safe_load(f)
print("Config loaded successfully")

workspace = args["workspace"]
country_list = args["country_list"].split(",")
objectives = args.get("objectives", ["net_econ_value", "biodiversity", "net_ghg_co2e"])
country = "Suriname_hexgon"


table_folder = Path(workspace) / country / "ValueTables"
file_list = [
    "extensification_bmps_irrigated.csv",
    "extensification_bmps_rainfed.csv",
    "extensification_current_practices.csv",
    "extensification_intensified_irrigated.csv",
    "extensification_intensified_rainfed.csv",
    "fixedarea_bmps_irrigated.csv",
    "fixedarea_bmps_rainfed.csv",
    "fixedarea_intensified_irrigated.csv",
    "fixedarea_intensified_rainfed.csv",
    "forestry_expansion.csv",
    "grazing_expansion.csv",
    "restoration.csv",
    "sustainable_current.csv"
]

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
    "noxn_in_drinking_water"
]

    
data_dict = DataHandler.load_objective_tables(str(table_folder), file_list, value_columns)

plot_output_dir = Path(".")  # Save in current directory

for obj, arr in data_dict.items():
    if isinstance(arr, np.ndarray):
        plt.figure()
        plt.plot(arr, label=obj)
        plt.title(f"{obj} over solutions")
        plt.xlabel("Index")
        plt.ylabel(obj)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_path = plot_output_dir / f"{obj}.png"
        plt.savefig(plot_path)
        print(f"Saved plot for {obj} to {plot_path}")
        plt.close()
    else:
        print(f"Skipped {obj}: not a NumPy array.")
