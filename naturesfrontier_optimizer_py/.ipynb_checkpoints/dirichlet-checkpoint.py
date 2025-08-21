import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_simplex_points(n=50):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X - Y
    mask = (Z >= 0)
    x = X[mask]
    y = Y[mask]
    z = Z[mask]
    return x, y, z

# Generate points on the 2-simplex surface
x, y, z = generate_simplex_points()

# 3D plot of the simplex
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='black', s=1)

# Triangle edges
vertices = np.array([
    [1, 0, 0],  # x=1
    [0, 1, 0],  # y=1
    [0, 0, 1],  # z=1
    [1, 0, 0]   # close loop
])
ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', linewidth=2)

# Labels and axes
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('2-Simplex: x + y + z = 1')
ax.view_init(elev=30, azim=120)
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.savefig("simplex_3d.png", dpi=300)
plt.show()
