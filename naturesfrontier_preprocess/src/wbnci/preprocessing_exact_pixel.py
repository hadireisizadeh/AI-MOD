from __future__ import annotations
import os
import sys
import json
import glob
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import osgeo.gdal as gdal
import osgeo.ogr as ogr
import osgeo.osr as osr
import pygeoprocessing.geoprocessing as pygeo
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import tempfile
import shutil

GDT_Byte_NODATA = 7


class RootPreprocessingError(Exception):
    pass


def read_country_list(list_file):
    """
    Read a list of country names (one name per line) and return a list
    """
    result = []
    with open(list_file) as f:
        for line in f:
            result.append(line.strip())
    return result


def make_scenario_map(source_file, target_file, transition_codes, new_code, all_codes):
    """
    Creates a new raster (target_file) where all values contained in transition_codes
    are remapped to new_code. Preserves data type and nodata from source raster.
    """
    value_map = {c: c for c in all_codes}
    for c in transition_codes:
        value_map[c] = new_code
    source_raster_info = pygeo.get_raster_info(source_file)

    pygeo.reclassify_raster(
        (source_file, 1),
        value_map,
        target_file,
        source_raster_info['datatype'],
        source_raster_info['nodata'][0]
    )


def make_scenario_map_pv(source_file, target_file, transition_codes, pv_raster_file, all_codes):
    """
    Creates a scenario map using potential vegetation values
    """
    source_nodata = pygeo.get_raster_info(source_file)['nodata'][0]

    def _reclass_op(src, pv):
        result = np.empty(src.shape)
        replace_mask = np.isin(src, transition_codes)
        result[replace_mask] = pv[replace_mask]
        result[~replace_mask] = src[~replace_mask]
        return result

    pygeo.raster_calculator(
        [(source_file, 1), (pv_raster_file, 1)],
        _reclass_op,
        target_file,
        gdal.GDT_Int16,
        source_nodata
    )


def make_union_mask(mask_list: List[str], target_file: str):
    """
    Creates a new raster at `target_file` which has value 1 for a pixel if any of the rasters
    in `mask_list` have a valid (non-NoData) value for that pixel and nodata elsewhere.
    """
    brpbcl = [(p, 1) for p in mask_list]

    def maskfn(*mask_list):
        result = np.ones(mask_list[0].shape)
        valid_pix = np.any([m != GDT_Byte_NODATA for m in mask_list], axis=0)
        result[~valid_pix] = GDT_Byte_NODATA
        return result

    pygeo.raster_calculator(
        brpbcl,
        maskfn,
        target_file,
        gdal.GDT_Byte,
        GDT_Byte_NODATA
    )


def make_intersection_mask(mask_list: List[str], target_file: str):
    """
    Creates a new raster at `target_file` which has the value 1 for a pixel if
    all of the rasters in `mask_list` have a valid (non-NoData or 0) value for that pixel
    and has value NoData elsewhere.
    """
    brpbcl = [(p, 1) for p in mask_list]

    def maskfn(*mask_list):
        result = np.ones(mask_list[0].shape)
        valid_pix = np.all([m > 0 for m in mask_list], axis=0)
        result[~valid_pix] = GDT_Byte_NODATA
        return result
    
    pygeo.raster_calculator(
        brpbcl,
        maskfn,
        target_file,
        gdal.GDT_Byte,
        GDT_Byte_NODATA
    )


# New function to create a pixel grid raster instead of SDU hexagons
def create_pixel_grid(country_root, args):
    """
    Creates a raster where each pixel has a unique identifier (SDUID).
    This is used in place of the hexagonal SDU grid.
    
    Parameters:
        country_root (string): path to the country folder
        args (dict): configuration parameters
        
    Returns:
        None
    """
    input_folder = os.path.join(country_root, "InputRasters")
    target_folder = os.path.join(country_root, "pixel_map")
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Use the masked LULC file as a base
    base_raster_path = os.path.join(input_folder, "current_lulc_masked.tif")
    target_file = os.path.join(target_folder, "pixel_grid.tif")
    
    # Get raster info
    raster_info = pygeo.get_raster_info(base_raster_path)
    nodata = raster_info['nodata'][0]
    
    # Generate a unique ID for each pixel
    def create_pixel_id_raster(*args):
        source_raster = args[0]
        rows, cols = source_raster.shape
        
        # Create a 2D array of unique IDs
        # We'll use row-major ordering (row * width + col)
        pixel_ids = np.zeros((rows, cols), dtype=np.int32)
        for r in range(rows):
            for c in range(cols):
                pixel_ids[r, c] = r * cols + c + 1  # +1 to avoid 0 as an ID
        
        # Apply mask from source raster
        mask = source_raster == nodata
        pixel_ids[mask] = -1  # Use -1 as nodata value for pixel IDs
        
        return pixel_ids

    # Use raster calculator to create the pixel ID raster
    pygeo.raster_calculator(
        [(base_raster_path, 1)],
        create_pixel_id_raster,
        target_file,
        gdal.GDT_Int32,
        -1  # -1 as nodata value for pixel IDs
    )
    
    print(f"Created pixel grid with unique IDs at {target_file}")
    return target_file


# New function to aggregate values at pixel level (replacing aggregate_marginal_values)
def aggregate_pixel_values(pixel_grid_path: str,
                           id_field_name: str,
                           mask_raster_path: str,
                           value_raster_lookup: dict) -> dict:
    """
    Extract values for each pixel from the value rasters.
    
    Parameters:
        pixel_grid_path (string): path to the pixel grid raster with unique IDs
        id_field_name (string): field name for the pixel IDs (e.g., 'SDUID')
        mask_raster_path (string): path to mask raster
        value_raster_lookup (dict): keys are marginal value IDs, values are paths to rasters
        
    Returns:
        A dictionary where keys are pixel IDs and values contain extracted values for each pixel
    """
    print('value_raster_lookup: {}'.format(value_raster_lookup))
    marginal_value_ids = list(value_raster_lookup.keys())
    
    # Open pixel grid raster
    pixel_grid = gdal.Open(pixel_grid_path)
    pixel_band = pixel_grid.GetRasterBand(1)
    pixel_nodata = pixel_band.GetNoDataValue()
    
    # Open mask raster
    mask_raster = gdal.Open(mask_raster_path)
    mask_band = mask_raster.GetRasterBand(1)
    mask_nodata = mask_band.GetNoDataValue()
    
    # Calculate pixel area in hectares
    geotransform = pixel_grid.GetGeoTransform()
    pixel_area_m2 = float((geotransform[1]) ** 2)  # Assuming square pixels
    pixel_area_ha = pixel_area_m2 / 10000.0
    
    # Open all value rasters
    marginal_value_rasters = [
        gdal.Open(value_raster_lookup[marginal_value_id])
        for marginal_value_id in marginal_value_ids
    ]
    marginal_value_bands = [
        raster.GetRasterBand(1) for raster in marginal_value_rasters
    ]
    marginal_value_nodata_list = [
        band.GetNoDataValue() for band in marginal_value_bands
    ]
    
    # Initialize results dictionary
    # Structure: {pixel_id: [pixel_area_ha, {marginal_value_id: value, ...}]}
    pixel_values = {}
    
    # Process by blocks for memory efficiency
    for block_offset, pixel_id_block in pygeo.iterblocks((pixel_grid_path, 1)):
        # Read mask and marginal value blocks for this area
        mask_block = mask_band.ReadAsArray(**block_offset)
        marginal_value_blocks = [
            band.ReadAsArray(**block_offset) for band in marginal_value_bands
        ]
        
        # Process each unique pixel ID in this block
        for unique_id in np.unique(pixel_id_block):
            if unique_id == pixel_nodata:
                continue
                
            # Create mask for this pixel ID
            pixel_mask = pixel_id_block == unique_id
            
            # Check if pixel is in the mask
            valid_mask = mask_block[pixel_mask] != mask_nodata
            if not np.any(valid_mask):
                continue
                
            # Initialize pixel entry with area
            pixel_values[int(unique_id)] = [pixel_area_ha, {}]
            
            # Extract values for each marginal value
            for mv_id, mv_nodata, mv_block in zip(
                    marginal_value_ids, marginal_value_nodata_list, marginal_value_blocks):
                
                value = mv_block[pixel_mask]
                if value.size > 0 and value[0] != mv_nodata:
                    pixel_values[int(unique_id)][1][mv_id] = float(value[0])
                else:
                    pixel_values[int(unique_id)][1][mv_id] = 0.0
    
    # Clean up
    pixel_band = None
    pixel_grid = None
    mask_band = None
    mask_raster = None
    for band in marginal_value_bands:
        band = None
    for raster in marginal_value_rasters:
        raster = None
        
    return pixel_values


# New function to build score table at pixel level
def build_pixel_score_table(
        sdu_col_name, activity_list, activity_name, pixel_values,
        pixel_serviceshed_coverage, target_ip_table_path, baseline_table=False):
    """
    Build a table for optimization using pixel-level values.
    
    Parameters:
        sdu_col_name (string): name of the column for pixel IDs
        activity_list (list): list of activity names
        activity_name (string): name of the current activity
        pixel_values (dict): dictionary with pixel IDs as keys and values as [area, {service_values}]
        pixel_serviceshed_coverage (dict): serviceshed coverage info (or None)
        target_ip_table_path (string): path to the output CSV file
        baseline_table (bool): whether this is a baseline table
    """
    if activity_name is not None:
        try:
            activity_index = activity_list.index(activity_name)
        except ValueError:
            msg = 'activity_name not found in activity_list'
            raise RootPreprocessingError(msg)
    else:
        activity_index = None

    with open(target_ip_table_path, 'w') as target_file:
        # Write header
        target_file.write(f"{sdu_col_name},pixel_area_ha")
        target_file.write(",%s_ha" * len(activity_list) % tuple(activity_list))
        target_file.write(',exclude')
        
        # Get marginal value IDs
        if len(pixel_values) > 0:
            first_entry = next(iter(pixel_values.values()))
            marginal_value_ids = sorted(first_entry[1].keys())
            n_mv_ids = len(marginal_value_ids)
            target_file.write((",%s" * n_mv_ids) % tuple(marginal_value_ids))
        else:
            marginal_value_ids = []
            
        # Handle serviceshed coverage (if provided)
        if pixel_serviceshed_coverage is not None and len(pixel_serviceshed_coverage) > 0:
            first_serviceshed_lookup = next(iter(pixel_serviceshed_coverage.values()))
            serviceshed_ids = sorted(first_serviceshed_lookup.keys())
            target_file.write((",%s" * len(serviceshed_ids)) % tuple(serviceshed_ids))
            
            value_ids = {
                sid: sorted(first_serviceshed_lookup[sid][1].keys()) 
                for sid in serviceshed_ids
            }
            
            for serviceshed_id in serviceshed_ids:
                for value_id in value_ids[serviceshed_id]:
                    target_file.write(",%s_%s" % (serviceshed_id, value_id))
        else:
            serviceshed_ids = []
            
        target_file.write('\n')
        
        # Write each row (one per pixel ID)
        for pixel_id in sorted(pixel_values.keys()):
            # Write pixel ID and area
            target_file.write(f"{pixel_id},{pixel_values[pixel_id][0]}")
            
            # Write areas by activity
            areas = [0.0 for _ in range(len(activity_list))]
            if baseline_table is False and activity_index is not None:
                # For the active scenario, set this area to the pixel area
                areas[activity_index] = pixel_values[pixel_id][0]
                
            target_file.write(",%f" * len(areas) % tuple(areas))
            
            # Determine if pixel should be excluded
            if baseline_table is False and max(areas) == 0:
                target_file.write(',1')
            else:
                target_file.write(',0')
            
            # Write all marginal values
            for mv_id in marginal_value_ids:
                target_file.write(f",{pixel_values[pixel_id][1].get(mv_id, 0.0)}")
            
            # Write serviceshed values (if applicable)
            if pixel_serviceshed_coverage is not None and pixel_id in pixel_serviceshed_coverage:
                for serviceshed_id in serviceshed_ids:
                    target_file.write(
                        f",{pixel_serviceshed_coverage[pixel_id][serviceshed_id][0]}"
                    )
                
                for serviceshed_id in serviceshed_ids:
                    for value_id in value_ids[serviceshed_id]:
                        target_file.write(
                            f",{pixel_serviceshed_coverage[pixel_id][serviceshed_id][1][value_id]}"
                        )
            elif serviceshed_ids:
                # If no serviceshed coverage for this pixel, write zeros
                target_file.write(",0.0" * len(serviceshed_ids))
                total_value_entries = sum(len(value_ids[sid]) for sid in serviceshed_ids)
                target_file.write(",0.0" * total_value_entries)
                
            target_file.write('\n')


def join_table_to_grid(grid_path, table_path, target_file):
    """
    Join a table to a grid shapefile.
    
    Parameters:
        grid_path (string): path to grid shapefile
        table_path (string): path to table to join
        target_file (string): path to output shapefile
    """
    df = pd.read_csv(table_path)
    gdf = gpd.read_file(grid_path)
    gdf = pd.merge(gdf, df, on='SDUID')
    gdf.to_file(target_file)