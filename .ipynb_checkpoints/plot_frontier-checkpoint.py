import pandas as pd 
import matplotlib.pyplot as plt
import os

cell_options = ["hexgon", "pixel", "coarser_grid"]
method = "fibonacci_lattice"
#"gaussian"
minpts = "5000"

for c in cell_options:
    path = os.path.expanduser(f"~/naturesfrontier/runs/example/Suriname_{c}/OptimizationResults/summary_table_maximize.csv")
    if not os.path.exists(path):
        print(f"Warning: File not found at {path}")
        continue
    
    df = pd.read_csv(path)
    df["net_econ_value"] = df["net_econ_value"].fillna(0)
    df["biodiversity"] = df["biodiversity"].fillna(0)
    df["net_ghg_co2e"] = df["net_ghg_co2e"].fillna(0)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    x = df["net_econ_value"]
    y = df["biodiversity"]
    z = df["net_ghg_co2e"]

    # Scatter plot
    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_xlabel("Net Economic Value")
    ax.set_ylabel("Biodiversity")
    ax.set_zlabel("Net GHG CO2e")
    ax.set_title(f"3D Plot of Pareto front - {c}")
    output_path = f"frontier_{c}_{method}_{minpts}.png"
    
    plt.savefig(output_path)
    plt.show()
