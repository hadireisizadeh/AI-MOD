# run_nci.py
from src.nci import do_nci
import yaml
import os
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt

# Get the directory where run_nci.py is located
script_dir = Path(__file__).parent.absolute()
config_file = script_dir / "config.yml"
print(f"Reading config from: {config_file}")

methods = ["gaussian", "fibonacci_lattice", "dirichlet"]
pts = 5000
cell_options = ["hexgon", "pixel", "coarser_grid"]
T_values = [1e0, 1e2, 1e3, 1e8]
colors = ["b", "g", "r", "k"] 

with open(config_file, 'r') as f:
    args = yaml.safe_load(f)

workspace = args["workspace"]
country_list = args["country_list"].split(",")
objectives = args.get("objectives", ["net_econ_value", "biodiversity", "net_ghg_co2e"])
results_folder = args.get("optimization_output_folder", "OptimizationResults")
sign_obj = args.get("sign_obj", [1,1,-1])


for method in methods:
    for T in T_values:
        results_folder_T = f"{results_folder}_T{T:.0e}"
        for country in country_list:
            print(f"Processing country: {country}, method={method}, T={T:.0e}")
            do_nci(workspace, country, objectives, sign_obj, method, T, pts, results_folder_T)

    for c in cell_options:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        for T, color in zip(T_values, colors):
            path = os.path.expanduser(
                f"~/naturesfrontier/runs/example/Suriname_{c}/OptimizationResults_T{T:.0e}/summary_table_maximize.csv"
            )

            if not os.path.exists(path):
                print(f"Warning: File not found at {path}")
                continue

            df = pd.read_csv(path)
            df["net_econ_value"] = df["net_econ_value"].fillna(0)
            df["biodiversity"] = df["biodiversity"].fillna(0)
            df["net_ghg_co2e"] = df["net_ghg_co2e"].fillna(0)

            x = df["net_econ_value"]
            y = df["biodiversity"]
            z = df["net_ghg_co2e"]

            ax.scatter(x, y, z, c=color, marker='o', alpha=0.85, label=f"T={T:.0e}")

        ax.set_xlabel("Net Economic Value")
        ax.set_ylabel("Biodiversity")
        ax.set_zlabel("Net GHG CO2e")
        ax.set_title(f"{c}-{method}")
        ax.legend(loc="best")

        output_path = f"frontiers_{c}_{method}_{pts}_T{T:.0e}.pdf"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"Saved {output_path}")
