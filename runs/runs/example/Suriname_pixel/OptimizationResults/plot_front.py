import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("summary_table_maximize.csv")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Extract objective values
x = df["net_econ_value"]
y = df["biodiversity"]
z = df["net_ghg_co2e"]

# Scatter plot in 3D
ax.scatter(x, y, z, c='b', marker='o')

# Labels and title
ax.set_xlabel("Net Economic Value")
ax.set_ylabel("Biodiversity")
ax.set_zlabel("Net GHG CO2e")
ax.set_title("3D Plot of Objective Values")

# Save in the current directory
output_path = "objective_plot_3D_4.png"
plt.savefig(output_path)
plt.show()

print(f"Saved 3D plot: {output_path}")
