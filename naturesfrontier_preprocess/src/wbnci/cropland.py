import os
import sys
import json
import glob
import math
from typing import List
import shutil
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import osgeo.gdal as gdal
import pygeoprocessing.geoprocessing as pygeo
import taskgraph

"""
This stage of the preprocessing will:
    - Construct the ag value scenario maps
    - Construct global carbon scenario maps?
    - Copy country-level ecoregion maps to country packages
"""

GDT_Byte_NODATA = 7
all_lu_codes = [10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90,
                100, 110, 120, 121, 122, 130, 140, 150, 152, 153, 160, 170, 180, 190,
                200, 201, 202, 210, 220, 15, 16, 25, 26]

ag_lucodes = [10, 11, 12, 15, 16, 20, 25, 26, 30, 40]
natural_lucodes = [50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110, 120, 121,
                   122, 130, 150, 151, 152, 153, 180]


def execute(args):
    """
    `args` should be a dictionary with keys:
    + `lu_raster`
    + `target_folder`
    + `lu_codes_table`
    + `cropland_current_value`
    + `cropland_irrigated_value`
    + `cropland_rainfed_value`
    + `pixel_area`
    + `set_zero_floor`
    """
    
    sname = os.path.splitext(os.path.basename(args['lu_raster']))[0]
    
    target_file = os.path.join(
        args['target_folder'], f"{sname}_cropland_value.tif")
    
    szf = args["set_zero_floor"] if "set_zero_floor" in args else False

    apply_ag_values(
        args['lu_raster'],
        target_file,
        args['cropland_current_value'],
        args['cropland_rainfed_value'],
        args['cropland_irrigated_value'],
        args['palm_oil_value'],
        args['pixel_area'],
        set_zero_floor = szf
    )


def apply_ag_values(scenario_file, output_file, current_practices, 
                    intensified_rainfed, intensified_irrigated, palm_oil,
                    pixel_area, set_zero_floor=False):
    
    print(os.path.basename(scenario_file))
       
    ref_raster_info = pygeo.get_raster_info(current_practices)
    
    raster_band_list = [(scenario_file, 1), (current_practices, 1),
                        (intensified_rainfed, 1), (intensified_irrigated, 1),
                        (palm_oil, 1), (pixel_area, 1)]    
    
    local_op = _ag_values_local_op_zero_floor if set_zero_floor else _ag_values_local_op
    
    pygeo.raster_calculator(
        raster_band_list, 
        local_op,
        output_file,
        gdal.GDT_Float32,
        ref_raster_info['nodata'][0],
        raster_driver_creation_tuple=('GTIFF', ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))

    )


current_codes = np.array([10, 11, 12, 20])
mosaic_75_code = 30
mosaic_25_code = 40
int_rf_code = 15
int_rf_bmps_code = 16
palm_oil_code = 29
int_irr_code = 25
int_irr_bmps_code = 26


def _ag_values_local_op(scen, cur, int_rf, int_irr, oil, pa):
    # TODO: this is inefficient - would make more sense to loop through elements once, but I'm 
    # TODO: not sure the best way to do that in python without incurring typecheck etc hits.
    result = np.zeros(scen.shape)
    
    # current practices
    mask = np.isin(scen, current_codes)
    result[mask] = cur[mask]
    # intensified rainfed
    mask = scen == int_rf_code
    result[mask] = int_rf[mask]
    # intensified rainfed with BMPs
    mask = scen == int_rf_bmps_code
    result[mask] = 0.9 * int_rf[mask]
    # intensified irrigated
    mask = scen == int_irr_code
    result[mask] = int_irr[mask]
    # intensified irrigated with BMPs
    mask = scen == int_irr_bmps_code
    result[mask] = 0.9 * int_irr[mask]
    # mosaic 75% cropland
    mask = scen == mosaic_75_code
    result[mask] = 0.75 * cur[mask]
    # mosaic 25% cropland
    mask = scen == mosaic_25_code
    result[mask] = 0.25 * cur[mask]
    # palm oil
    mask = scen == palm_oil_code
    result[mask] = oil[mask]

    return result * pa * 100  # pixel area is km2, ag value is per hectare


def _ag_values_local_op_zero_floor(scen, cur, int_rf, int_irr, oil, pa):
    # TODO: this is inefficient - would make more sense to loop through elements once, but I'm 
    # TODO: not sure the best way to do that in python without incurring typecheck etc hits.
    result = np.zeros(scen.shape)
    
    # current practices
    mask = np.isin(scen, current_codes)
    result[mask] = cur[mask]
    # intensified rainfed
    mask = scen == int_rf_code
    result[mask] = int_rf[mask]
    # intensified rainfed with BMPs
    mask = scen == int_rf_bmps_code
    result[mask] = 0.9 * int_rf[mask]
    # intensified irrigated
    mask = scen == int_irr_code
    result[mask] = int_irr[mask]
    # intensified irrigated with BMPs
    mask = scen == int_irr_bmps_code
    result[mask] = 0.9 * int_irr[mask]
    # mosaic 75% cropland
    mask = scen == mosaic_75_code
    result[mask] = 0.75 * cur[mask]
    # mosaic 25% cropland
    mask = scen == mosaic_25_code
    result[mask] = 0.25 * cur[mask]
    # palm oil
    mask = scen == palm_oil_code
    result[mask] = oil[mask]
    
    result[result<0] = 0
    return result * pa * 100  # pixel area is km2, ag value is per hectare

