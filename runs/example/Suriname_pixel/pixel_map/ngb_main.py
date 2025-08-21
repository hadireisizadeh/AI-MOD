import numpy as np
import pandas as pd
import rasterio
import json

def rook_neighbors_dict(raster_path: str) -> dict[int, dict[str, int | None]]:
    """
    Return a dictionary mapping each pixel SUID to its N/E/S/W neighbors.
    Missing neighbors (border or nodata) are None.
    Example: {399: {"N": 654, "E": 400, "S": None, "W": None}, ...}
    """
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        nodata = src.nodata

    valid = np.ones_like(arr, dtype=bool)
    if nodata is not None:
        valid &= (arr != nodata)

    N = np.full(arr.shape, np.nan)
    E = np.full(arr.shape, np.nan)
    S = np.full(arr.shape, np.nan)
    W = np.full(arr.shape, np.nan)

    # Fill neighbors by shifting
    N[1:, :] = arr[:-1, :]
    S[:-1, :] = arr[1:, :]
    W[:, 1:] = arr[:, :-1]
    E[:, :-1] = arr[:, 1:]

    if nodata is not None:
        for neigh in (N, E, S, W):
            neigh[neigh == nodata] = np.nan

    # Build dictionary
    result = {}
    for (i, j), val in np.ndenumerate(arr):
        if not valid[i, j]:
            continue
        suid = int(val)
        result[suid] = {
            "N": int(N[i, j]) if not np.isnan(N[i, j]) else None,
            "E": int(E[i, j]) if not np.isnan(E[i, j]) else None,
            "S": int(S[i, j]) if not np.isnan(S[i, j]) else None,
            "W": int(W[i, j]) if not np.isnan(W[i, j]) else None,
        }
    return result


pixel_grid = "pixel_grid.tif"
neighbors_dict = rook_neighbors_dict(pixel_grid)
# neighbors_dict = rook_neighbors_dict("pixel_grid.tif")  # from previous step

with open("neighbors.json", "w") as f:
    json.dump(neighbors_dict, f, indent=2)

