import os
import glob
import pandas as pd
import numpy as np
import osgeo.gdal as gdal
import pygeoprocessing.geoprocessing as pygeo


def execute(args):
    """
    Reimplementation of Justin's cython carbon model to use pygeoprocessing. 
    The original model read in the full rasters, which was causing memory use
    issues when multiprocessing. This version uses the raster calculator which
    is memory efficient.
    
    This version also checks whether the carbon value for a non-natural class
    is greater than the carbon value for the potential vegetation, and if so, 
    reduces the non-natural carbon
    
    args should contain:
    + lu_raster
    + pixel_area
    + potential_vegetation
    + target_folder
    + carbon_zone_file
    + carbon_table_file
    """
    
    output_folder = args['target_folder']
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    carbon_table_path = args["carbon_table_file"]
    carbon_zones_path = args["carbon_zone_file"]

    # set up lookup tables
    lookup_table_df = pd.read_csv(carbon_table_path, index_col=0)
    table_shape = (len(lookup_table_df.index), len(lookup_table_df.columns))
    lookup_table = np.float32(lookup_table_df.values)
    # row_names = {int(v): int(c) for c, v in enumerate(lookup_table_df.index)}
    # col_names = {int(v): int(c) for c, v in enumerate(lookup_table_df.columns)}
    rows = [int(r) for r in lookup_table_df.index]
    cols = [int(r) for r in lookup_table_df.columns]

    row_idx = np.zeros(max(rows)+1, dtype='int')
    col_idx = np.zeros(max(cols)+1, dtype='int')

    for i, r in enumerate(rows):
        row_idx[r] = i
    for i, c in enumerate(cols):
        col_idx[c] = i

    def local_op(lu, pv, cz, pa):
        lu_carb = lookup_table[row_idx[cz], col_idx[lu]] * 100 * pa
        pv_carb = lookup_table[row_idx[cz], col_idx[pv]] * 100 * pa
        # 100 scaling is for ha -> km2 (pixel area in ha, carbon in tons/km2)

        result = np.zeros_like(pa)
        result = np.minimum(lu_carb, pv_carb)
        bmp_mask = np.any([lu==16, lu==26], axis=0)
        result[bmp_mask] = 0.9 * lu_carb[bmp_mask] + 0.1 * pv_carb[bmp_mask]
        return result

    raster_band_list = [
        (args["lu_raster"], 1),
        (args["potential_vegetation"], 1),
        (args["carbon_zone_file"], 1),
        (args["pixel_area"], 1)
    ]
    sname = os.path.splitext(os.path.basename(args["lu_raster"]))[0]
    target_file = os.path.join(output_folder, f'{sname}_carbon.tif')

    pygeo.raster_calculator(
        raster_band_list,
        local_op, 
        target_file,
        gdal.GDT_Float32,
        -9999.0
    )
        

    