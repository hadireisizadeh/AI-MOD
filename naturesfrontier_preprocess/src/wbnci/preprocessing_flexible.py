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
from typing import List, Dict, Tuple, Any, Optional, Union
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


# ---------- HEXAGONAL SDU FUNCTIONS ----------

def create_regular_sdu_grid(
        mask_raster_path, grid_type, cell_size, out_grid_vector_path,
        sdu_id_fieldname, remove_nonoverlapping=False):
    """Convert vector to a regular grid.

    Here the vector is gridded such that all cells are contained within the
    original vector.  Cells that would intersect with the boundary are not
    produced.

    Parameters:
        mask_raster_path (string): path to a single band raster where
            pixels valued at '1' are valid and invalid otherwise.
        grid_type (string): one of "square" or "hexagon"
        cell_size (float): dimensions of the grid cell in the projected units
            of `vector_path`; if "square" then this indicates the side length,
            if "hexagon" indicates the width of the horizontal axis.
        out_grid_vector_path (string): path to the output ESRI shapefile
            vector that contains a gridded version of `vector_path`, this file
            should not exist before this call
        sdu_id_fieldname (string): desired key id field
        remove_nonoverlapping (bool): default behavior is to make a rectangular grid.
            Change to True to filter to only polygons that overlap pixels in the mask
            raster.

    Returns:
        None
    """

    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(out_grid_vector_path):
        driver.DeleteDataSource(out_grid_vector_path)

    raster_mask = gdal.Open(mask_raster_path)
    spatial_reference = osr.SpatialReference(raster_mask.GetProjection())

    out_grid_vector = driver.CreateDataSource(out_grid_vector_path)
    grid_layer = out_grid_vector.CreateLayer(
        'grid', spatial_reference, ogr.wkbPolygon)
    grid_layer.CreateField(
        ogr.FieldDefn(str(sdu_id_fieldname), ogr.OFTInteger))
    grid_layer_defn = grid_layer.GetLayerDefn()

    geotransform = raster_mask.GetGeoTransform()
    # minx maxx miny maxy
    extent = [
        geotransform[0],
        (geotransform[0] +
         raster_mask.RasterXSize * geotransform[1] +
         raster_mask.RasterYSize * geotransform[2]),
        (geotransform[3] +
         raster_mask.RasterXSize * geotransform[4] +
         raster_mask.RasterYSize * geotransform[5]),
        geotransform[3]
        ]
    raster_mask = None

    # flip around if one direction is negative or not; annoying case that'll
    # always linger unless directly approached like this
    extent = [
        min(extent[0], extent[1]),
        max(extent[0], extent[1]),
        min(extent[2], extent[3]),
        max(extent[2], extent[3])]

    print(f"sdu extent: {extent}")

    if grid_type == 'hexagon':
        # calculate the inner dimensions of the hexagons
        grid_width = extent[1] - extent[0]
        grid_height = extent[3] - extent[2]
        delta_short_x = cell_size * 0.25
        delta_long_x = cell_size * 0.5
        delta_y = cell_size * 0.25 * (3 ** 0.5)

        # Since the grid is hexagonal it's not obvious how many rows and
        # columns there should be just based on the number of squares that
        # could fit into it.  The solution is to calculate the width and
        # height of the largest row and column.
        n_cols = int(math.floor(grid_width / (3 * delta_long_x)) + 1)
        n_rows = int(math.floor(grid_height / delta_y) + 1)

        print(f"sdu grid size: {n_rows}, {n_cols}")

        def _generate_polygon(col_index, row_index):
            """Generate a points for a closed hexagon."""
            if (row_index + 1) % 2:
                centroid = (
                    extent[0] + (delta_long_x * (1 + (3 * col_index))),
                    extent[2] + (delta_y * (row_index + 1)))
            else:
                centroid = (
                    extent[0] + (delta_long_x * (2.5 + (3 * col_index))),
                    extent[2] + (delta_y * (row_index + 1)))
            x_coordinate, y_coordinate = centroid
            hexagon = [(x_coordinate - delta_long_x, y_coordinate),
                       (x_coordinate - delta_short_x, y_coordinate + delta_y),
                       (x_coordinate + delta_short_x, y_coordinate + delta_y),
                       (x_coordinate + delta_long_x, y_coordinate),
                       (x_coordinate + delta_short_x, y_coordinate - delta_y),
                       (x_coordinate - delta_short_x, y_coordinate - delta_y),
                       (x_coordinate - delta_long_x, y_coordinate)]
            return hexagon
    elif grid_type == 'square':
        def _generate_polygon(col_index, row_index):
            """Generate points for a closed square."""
            square = [
                (extent[0] + col_index * cell_size + x,
                 extent[2] + row_index * cell_size + y)
                for x, y in [
                    (0, 0), (cell_size, 0), (cell_size, cell_size),
                    (0, cell_size), (0, 0)]]
            return square
        n_rows = int((extent[3] - extent[2]) / cell_size)
        n_cols = int((extent[1] - extent[0]) / cell_size)
    else:
        raise ValueError('Unknown polygon type: %s' % grid_type)

    for row_index in range(n_rows):
        for col_index in range(n_cols):
            polygon_points = _generate_polygon(col_index, row_index)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for xoff, yoff in polygon_points:
                ring.AddPoint(xoff, yoff)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            poly_feature = ogr.Feature(grid_layer_defn)
            poly_feature.SetGeometry(poly)
            poly_feature.SetField(
                str(sdu_id_fieldname), row_index * n_cols + col_index)
            grid_layer.CreateFeature(poly_feature)
    grid_layer.SyncToDisk()

    grid_layer = None
    out_grid_vector.Destroy()

    if remove_nonoverlapping:
        remove_nonoverlapping_sdus(out_grid_vector_path, mask_raster_path, sdu_id_fieldname)


def remove_nonoverlapping_sdus(vector_path, mask_raster_path, key_id_field):
    """Remove polygons in `vector_path` that don't overlap valid data.

    Parameters:
        vector_path (string): path to a single layer polygon shapefile
            that has a  unique key field named `key_id_field`.  This function
            modifies this polygon to remove any polygons.
        make_raster_path (string): path to a mask raster; polygons in
            `vector_path` that only overlap nodata pixels will be removed.
        key_id_field (string): name of key id field in the polygon vector.

    Returns:
        None.
    """
    with tempfile.NamedTemporaryFile(dir='.', delete=False) as id_raster_file:
        id_raster_path = id_raster_file.name

    id_nodata = -1
    pygeo.new_raster_from_base(
        mask_raster_path,
        id_raster_path,
        gdal.GDT_Int32,
        [id_nodata],
        fill_value_list=[id_nodata],
    )
    id_raster = gdal.Open(id_raster_path, gdal.GA_Update)

    tmp_vector_dir = tempfile.mkdtemp()
    vector_basename = os.path.basename(vector_path)
    vector_driver = ogr.GetDriverByName("ESRI Shapefile")
    base_vector = ogr.Open(vector_path)
    vector = vector_driver.CopyDataSource(
        base_vector, os.path.join(tmp_vector_dir, vector_basename))
    base_vector = None
    layer = vector.GetLayer()

    gdal.RasterizeLayer(
        id_raster, [1], layer, options=['ATTRIBUTE=%s' % key_id_field])
    id_band = id_raster.GetRasterBand(1)
    # mask_nodata = pygeoprocessing.get_nodata_from_uri(mask_raster_path)
    mask_nodata = pygeo.get_raster_info(mask_raster_path)['nodata']
    covered_ids = set()
    for mask_offset, mask_block in pygeo.iterblocks((mask_raster_path, 1)):
        id_block = id_band.ReadAsArray(**mask_offset)
        valid_mask = mask_block != mask_nodata
        covered_ids.update(np.unique(id_block[valid_mask]))

    # cleanup the ID raster since we're done with it
    id_band = None
    id_raster = None
    os.remove(id_raster_path)

    # now it's sufficient to check if the min value on each feature is defined
    # if so there are valid pixels underneath, otherwise none.
    for feature in layer:
        feature_id = feature.GetField(str(key_id_field))
        if feature_id not in covered_ids:
            layer.DeleteFeature(feature.GetFID())

    print('INFO: Packing Target SDU Grid')
    # remove target vector and create a new one in its place with same layer
    # and fields
    os.remove(vector_path)
    target_vector = vector_driver.CreateDataSource(vector_path)
    spatial_ref = osr.SpatialReference(layer.GetSpatialRef().ExportToWkt())
    target_layer = target_vector.CreateLayer(
        str(os.path.splitext(vector_basename)[0]),
        spatial_ref, ogr.wkbPolygon)
    layer_defn = layer.GetLayerDefn()
    for index in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(index)
        field_defn.SetWidth(24)
        target_layer.CreateField(field_defn)

    # copy over undeleted features
    layer.ResetReading()
    for feature in layer:
        target_layer.CreateFeature(feature)
    target_layer = None
    target_vector = None
    layer = None
    vector = None

    # remove unpacked vector
    shutil.rmtree(tmp_vector_dir)


def aggregate_marginal_values(sdu_grid_path: str,
                              sdu_key_id: str,
                              mask_raster_path: str,
                              value_raster_lookup: dict,
                              sdu_id_raster_path=None) -> dict:
    """Build table that indexes SDU ids with aggregated marginal values.

    Parameters:
        sdu_grid_path (string): path to single layer polygon vector with
            integer field id that uniquely identifies each polygon.
        sdu_key_id (string): field in `sdu_grid_path` that uniquely identifies
            each feature.
        mask_raster_path (string): path to a mask raster whose pixels are
            considered "valid" if they are not nodata.
        value_raster_lookup (dict): keys are marginal value IDs that
            will be used in the optimization table; values are paths to
            single band rasters.
        sdu_id_raster_path (string | None): if not None, this raster will
            be used as the aggregating index raster. 

    Returns:
        A dictionary that encapsulates stats about each polygon, mask coverage
        and marginal value aggregation and coverage. Each key in the dict is
        the SDU_ID for a polygon, while the value is a tuple that contains
        first polygon/mask stats, then another dict for marginal value stats.
        In pseudocode:
            { sdu_id0:
                (sdu area, sdu pixel coverage, mask pixel count,
                 mask pixel coverage in Ha),
                {marginal value id a: (
                    aggregated values, n pixels of coverage,
                    aggregated value per Ha of coverage),
                 marginal valud id b: ...},
              sdu_id1: ...
            }
    """
    print('value_raster_lookup: {}'.format(value_raster_lookup))
    marginal_value_ids = list(value_raster_lookup.keys())

    id_nodata = -1
    if sdu_id_raster_path is None:
        with tempfile.NamedTemporaryFile(dir='.', delete=False) as id_raster_file:
            id_raster_path = id_raster_file.name

        pygeo.new_raster_from_base(
            value_raster_lookup[marginal_value_ids[0]],
            id_raster_path,
            gdal.GDT_Int32,
            [id_nodata],
            fill_value_list=[id_nodata])
        id_raster = gdal.Open(id_raster_path, gdal.GA_Update)

        vector = ogr.Open(sdu_grid_path, 1)  # open for reading
        layer = vector.GetLayer()
        gdal.RasterizeLayer(
            id_raster, [1], layer, options=['ATTRIBUTE=%s' % sdu_key_id])
        id_raster = None
        layer = None
        vector = None
    else:
        id_raster_path = sdu_id_raster_path

    mask_raster = gdal.Open(mask_raster_path)
    mask_band = mask_raster.GetRasterBand(1)
    mask_nodata = mask_band.GetNoDataValue()
    geotransform = mask_raster.GetGeoTransform()
    # note: i'm assuming square pixels that are aligned NS and EW and
    # projected in meters as linear units
    pixel_area_m2 = float((geotransform[1]) ** 2)

    marginal_value_rasters = [
        gdal.Open(value_raster_lookup[marginal_value_id])
        for marginal_value_id in marginal_value_ids]
    marginal_value_bands = [
        raster.GetRasterBand(1) for raster in marginal_value_rasters]
    marginal_value_nodata_list = [
        band.GetNoDataValue() for band in marginal_value_bands]

    # first element in tuple is the coverage stats:
    # (sdu area, sdu pixel count, mask pixel count, mask pixel coverage in Ha)
    # second element 3 element list (aggregate sum, pixel count, sum/Ha)
    marginal_value_sums = defaultdict(
        lambda: (
            [0.0, 0, 0, 0.0],
            dict((mv_id, [0.0, 0, None]) for mv_id in marginal_value_ids)))

    # format of sdu_coverage is:
    # (sdu area, sdu pixel count, mask pixel count, mask pixel coverage in Ha)
    for block_offset, id_block in pygeo.iterblocks((id_raster_path, 1)):
        marginal_value_blocks = [
            band.ReadAsArray(**block_offset) for band in marginal_value_bands]
        mask_block = mask_band.ReadAsArray(**block_offset)
        for aggregate_id in np.unique(id_block):
            if aggregate_id == id_nodata:
                continue
            aggregate_mask = id_block == aggregate_id
            # update sdu pixel coverage
            # marginal_value_sums[aggregate_id][0] =
            #    (sdu area, sdu pixel count, mask pixel count, mask pixel Ha)
            marginal_value_sums[aggregate_id][0][1] += np.count_nonzero(
                aggregate_mask)
            valid_mask_block = mask_block[aggregate_mask]
            marginal_value_sums[aggregate_id][0][2] += np.count_nonzero(
                valid_mask_block != mask_nodata)
            for mv_id, mv_nodata, mv_block in zip(
                    marginal_value_ids, marginal_value_nodata_list,
                    marginal_value_blocks):
                valid_mv_block = mv_block[aggregate_mask]
                # raw aggregation of marginal value
                # marginal_value_sums[aggregate_id][1][mv_id] =
                # (sum, pixel count, pixel Ha)
                marginal_value_sums[aggregate_id][1][mv_id][0] += np.nansum(
                    valid_mv_block[np.logical_and(
                        valid_mv_block != mv_nodata,
                        valid_mask_block != mask_nodata)])
                # pixel count coverage of marginal value
                marginal_value_sums[aggregate_id][1][mv_id][1] += (
                    np.count_nonzero(np.logical_and(
                        valid_mv_block != mv_nodata,
                        valid_mask_block != mask_nodata)))
    # calculate SDU, mask coverage in Ha, and marginal value Ha coverage
    for sdu_id in marginal_value_sums:
        marginal_value_sums[sdu_id][0][0] = (
            marginal_value_sums[sdu_id][0][1] * pixel_area_m2 / 10000.0)
        marginal_value_sums[sdu_id][0][3] = (
            marginal_value_sums[sdu_id][0][2] * pixel_area_m2 / 10000.0)
        # calculate the 3rd tuple of marginal value per Ha
        for mv_id in marginal_value_sums[sdu_id][1]:
            if marginal_value_sums[sdu_id][1][mv_id][1] != 0:
                marginal_value_sums[sdu_id][1][mv_id][2] = (
                    marginal_value_sums[sdu_id][1][mv_id][0] / (
                        marginal_value_sums[sdu_id][1][mv_id][1] *
                        pixel_area_m2 / 10000.0))
            else:
                marginal_value_sums[sdu_id][1][mv_id][2] = 0.0
    del marginal_value_bands[:]
    del marginal_value_rasters[:]
    mask_band = None
    mask_raster = None
    if sdu_id_raster_path is None:
        os.remove(id_raster_path)
    return marginal_value_sums


def build_sdu_score_table(
        sdu_col_name, activity_list, activity_name, marginal_value_lookup,
        sdu_serviceshed_coverage, target_ip_table_path, baseline_table=False):
    """Build a table for Integer Programmer.

    Output is a CSV table with columns identifying the aggregating SDU_ID,
    stats about SDU and mask coverage, as well as aggregate values for
    marginal values.

    Parameters:
        sdu_col_name (string): desired name of the SDU id column in the
            target IP table.
        marginal_value_lookup (dict): in pseudocode:
         { sdu_id0:
                (sdu area, sdu pixel coverage, mask pixel count,
                 mask pixel coverage in Ha),
                {marginal value id a: (
                    aggreated values, n pixels of coverage,
                    aggregated value per Ha of covrage),
                 marginal value id b: ...},
              sdu_id1: ...
            }
        sdu_serviceshed_coverage (dict): in pseudocode:
            {
                sdu_id_0: {
                    "serviceshed_id_a":
                        [serviceshed coverage proportion for a on id_0,
                         {service_shed_a_value_i: sum of value_i multiplied
                          by proportion of coverage of sdu_id_0 with
                          servicshed _id_a.}]
                    "serviceshed_id_b": ....
                },
                sdu_id_1: {....
            }
        target_ip_table_path (string): path to target IP table that will
            have the columns:
                SDU_ID,pixel_count,area_ha,maskpixct,maskpixha,mv_ida,mv_ida_perHA
    """
    if activity_name is not None:
        try:
            activity_index = activity_list.index(activity_name)
        except ValueError:
            msg = 'activity_name not found in activity_list in _build_ip_table'
            raise RootPreprocessingError(msg)
    else:
        activity_index = None

    with open(target_ip_table_path, 'w') as target_ip_file:
        # write header
        target_ip_file.write(
            "{},pixel_count,area_ha".format(sdu_col_name))
        target_ip_file.write(",%s_ha" * len(activity_list) % tuple(activity_list))
        target_ip_file.write(',exclude')
        # target_ip_file.write(
        #     "{},pixel_count,area_ha,{}_px,{}_ha".format(
        #         sdu_col_name, activity_name, activity_name))
        # This gets the "first" value in the dict, then the keys of that dict
        # also makes sense to sort them so it's easy to navigate the CSV.
        first_marg_value_element = next(iter(marginal_value_lookup.values()))
        marginal_value_ids = sorted(first_marg_value_element[1].keys())

        n_mv_ids = len(marginal_value_ids)
        target_ip_file.write((",%s" * n_mv_ids) % tuple(marginal_value_ids))
        # target_ip_file.write(
        #     (",%s_perHA" * n_mv_ids) % tuple(marginal_value_ids))
        if sdu_serviceshed_coverage is not None:
            first_serviceshed_lookup = next(iter(sdu_serviceshed_coverage))
        else:
            first_serviceshed_lookup = {}
        serviceshed_ids = sorted(first_serviceshed_lookup.keys())
        target_ip_file.write(
            (",%s" * len(serviceshed_ids)) % tuple(serviceshed_ids))
        value_ids = {
            sid: sorted(first_serviceshed_lookup[sid][1].keys()) for
            sid in serviceshed_ids
            }
        for serviceshed_id in serviceshed_ids:
            for value_id in value_ids[serviceshed_id]:
                target_ip_file.write(",%s_%s" % (serviceshed_id, value_id))
        target_ip_file.write('\n')

        # write each row
        for sdu_id in sorted(marginal_value_lookup):
            # id, pixel count, total pixel area,
            target_ip_file.write(
                "%d,%d,%f" % (
                    sdu_id, marginal_value_lookup[sdu_id][0][1],
                    marginal_value_lookup[sdu_id][0][0]))

            # areas by activity
            areas = [0 for _ in range(len(activity_list))]
            if baseline_table is False and activity_index is not None:
                areas[activity_index] = marginal_value_lookup[sdu_id][0][3]
            target_ip_file.write(",%f" * len(areas) % tuple(areas))
            # if all areas are 0, that means in particular the current activity has 0 available area
            # and we want to exclude this SDU as an option
            if baseline_table is False and max(areas) == 0:
                target_ip_file.write(',1')
            else:
                target_ip_file.write(',0')

            # write out all the marginal value aggregate values
            for mv_id in marginal_value_ids:
                target_ip_file.write(
                    ",%f" % marginal_value_lookup[sdu_id][1][mv_id][0])
            # write out all marginal value aggregate values per Ha
            # for mv_id in marginal_value_ids:
            #     target_ip_file.write(
            #         ",%f" % marginal_value_lookup[sdu_id][1][mv_id][2])
            # serviceshed values
            for serviceshed_id in serviceshed_ids:
                target_ip_file.write(
                    (",%f" % sdu_serviceshed_coverage[sdu_id][serviceshed_id][0]))
            for serviceshed_id in serviceshed_ids:
                for value_id in value_ids[serviceshed_id]:
                    target_ip_file.write(
                        (",%f" % sdu_serviceshed_coverage[sdu_id][serviceshed_id][1][value_id]))
            target_ip_file.write('\n')


# ---------- PIXEL LEVEL FUNCTIONS ----------

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


# ---------- COARSER GRID FUNCTIONS ----------

def create_coarser_grid(country_root, args):
    """
    Creates a raster with a coarser grid of pixels, where each coarser pixel
    has a unique identifier. This is an intermediate approach between
    individual pixels and hexagons.
    
    Parameters:
        country_root (string): path to the country folder
        args (dict): configuration parameters
        
    Returns:
        None
    """
    input_folder = os.path.join(country_root, "InputRasters")
    target_folder = os.path.join(country_root, "coarser_grid_map")
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Use the masked LULC file as a base
    base_raster_path = os.path.join(input_folder, "current_lulc_masked.tif")
    target_file = os.path.join(target_folder, "coarser_grid.tif")
    
    # Get the coarsening factor from config
    coarsening_factor = args.get("coarsening_factor", 10)
    
    # Get raster info
    raster_info = pygeo.get_raster_info(base_raster_path)
    base_raster = gdal.Open(base_raster_path)
    base_geotransform = base_raster.GetGeoTransform()
    base_projection = base_raster.GetProjection()
    base_nodata = raster_info['nodata'][0]
    
    # Calculate new dimensions
    original_width = base_raster.RasterXSize
    original_height = base_raster.RasterYSize
    new_width = original_width // coarsening_factor
    new_height = original_height // coarsening_factor
    
    # Create new coarser grid raster
    driver = gdal.GetDriverByName('GTiff')
    coarser_raster = driver.Create(
        target_file, 
        new_width, 
        new_height, 
        1, 
        gdal.GDT_Int32
    )
    
    # Set geotransform for coarser raster
    new_geotransform = list(base_geotransform)
    new_geotransform[1] = base_geotransform[1] * coarsening_factor  # Pixel width
    new_geotransform[5] = base_geotransform[5] * coarsening_factor  # Pixel height
    coarser_raster.SetGeoTransform(new_geotransform)
    coarser_raster.SetProjection(base_projection)
    
    # Get band and set no data value
    band = coarser_raster.GetRasterBand(1)
    band.SetNoDataValue(-1)
    
    # Fill with unique IDs
    id_array = np.zeros((new_height, new_width), dtype=np.int32)
    grid_id = 1
    for y in range(new_height):
        for x in range(new_width):
            id_array[y, x] = grid_id
            grid_id += 1
    
    # Write data
    band.WriteArray(id_array)
    
    # Clean up
    band = None
    coarser_raster = None
    base_raster = None
    
    print(f"Created coarser grid with unique IDs at {target_file}")
    
    # Now create a version of the original base raster that's segmented into the coarser grid
    # This will be used to map original pixels to coarser grid cells
    target_file_detailed = os.path.join(target_folder, "coarser_grid_detailed.tif")
    
    def expand_coarser_grid(*args):
        base_data = args[0]
        rows, cols = base_data.shape
        
        # Create expanded ID array matching original raster size
        expanded_ids = np.ones((rows, cols), dtype=np.int32) * -1
        
        # For each coarser cell, assign its ID to all contained original pixels
        for y in range(0, rows):
            for x in range(0, cols):
                if base_data[y, x] != base_nodata:
                    coarse_y = y // coarsening_factor
                    coarse_x = x // coarsening_factor
                    
                    # Check if within bounds of the coarser grid
                    if coarse_y < new_height and coarse_x < new_width:
                        expanded_ids[y, x] = coarse_y * new_width + coarse_x + 1
        
        return expanded_ids
    
    # Use raster calculator to create the expanded grid
    pygeo.raster_calculator(
        [(base_raster_path, 1)],
        expand_coarser_grid,
        target_file_detailed,
        gdal.GDT_Int32,
        -1
    )
    
    print(f"Created detailed coarser grid mapping at {target_file_detailed}")
    return target_file_detailed


def aggregate_coarser_grid_values(
        coarser_grid_path: str,
        id_field_name: str,
        mask_raster_path: str,
        value_raster_lookup: dict) -> dict:
    """
    Aggregate values from original rasters to the coarser grid.
    
    Parameters:
        coarser_grid_path (string): path to the detailed coarser grid raster with IDs
        id_field_name (string): field name for the coarser grid IDs 
        mask_raster_path (string): path to mask raster
        value_raster_lookup (dict): keys are marginal value IDs, values are paths to rasters
        
    Returns:
        A dictionary where keys are coarser grid IDs and values contain aggregated values
    """
    print('value_raster_lookup: {}'.format(value_raster_lookup))
    marginal_value_ids = list(value_raster_lookup.keys())
    
    # Open grid raster
    grid_raster = gdal.Open(coarser_grid_path)
    grid_band = grid_raster.GetRasterBand(1)
    grid_nodata = grid_band.GetNoDataValue()
    
    # Open mask raster
    mask_raster = gdal.Open(mask_raster_path)
    mask_band = mask_raster.GetRasterBand(1)
    mask_nodata = mask_band.GetNoDataValue()
    
    # Calculate pixel area in hectares
    geotransform = grid_raster.GetGeoTransform()
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
    # Structure: {grid_id: [total_area_ha, pixel_count, {marginal_value_id: [sum, count, per_ha], ...}]}
    grid_values = defaultdict(lambda: [0.0, 0, {mv_id: [0.0, 0, 0.0] for mv_id in marginal_value_ids}])
    
    # Process by blocks for memory efficiency
    for block_offset, grid_id_block in pygeo.iterblocks((coarser_grid_path, 1)):
        # Read mask and marginal value blocks for this area
        mask_block = mask_band.ReadAsArray(**block_offset)
        marginal_value_blocks = [
            band.ReadAsArray(**block_offset) for band in marginal_value_bands
        ]
        
        # Process each unique grid ID in this block
        for unique_id in np.unique(grid_id_block):
            if unique_id == grid_nodata:
                continue
                
            # Create mask for this grid ID
            grid_mask = grid_id_block == unique_id
            
            # Count valid pixels in this grid cell
            valid_mask = np.logical_and(grid_mask, mask_block != mask_nodata)
            valid_pixel_count = np.sum(valid_mask)
            
            if valid_pixel_count == 0:
                continue
                
            # Update pixel count and area
            grid_values[int(unique_id)][1] += valid_pixel_count
            grid_values[int(unique_id)][0] += valid_pixel_count * pixel_area_ha
            
            # Aggregate values for each marginal value
            for mv_id, mv_nodata, mv_block in zip(
                    marginal_value_ids, marginal_value_nodata_list, marginal_value_blocks):
                
                valid_value_mask = np.logical_and(valid_mask, mv_block != mv_nodata)
                if np.any(valid_value_mask):
                    values = mv_block[valid_value_mask]
                    grid_values[int(unique_id)][2][mv_id][0] += np.sum(values)
                    grid_values[int(unique_id)][2][mv_id][1] += len(values)
    
    # Calculate per hectare values
    for grid_id in grid_values:
        for mv_id in marginal_value_ids:
            if grid_values[grid_id][2][mv_id][1] > 0:
                total_area_ha = grid_values[grid_id][2][mv_id][1] * pixel_area_ha
                if total_area_ha > 0:
                    grid_values[grid_id][2][mv_id][2] = grid_values[grid_id][2][mv_id][0] / total_area_ha
    
    # Clean up
    grid_band = None
    grid_raster = None
    mask_band = None
    mask_raster = None
    for band in marginal_value_bands:
        band = None
    for raster in marginal_value_rasters:
        raster = None
    
    # Convert from defaultdict to regular dict for consistency with other functions
    result = {}
    for grid_id, [total_area, pixel_count, mv_data] in grid_values.items():
        # Format similar to other aggregation functions: [area_data, marginal_value_data]
        # where area_data is [area_ha, pixel_count, valid_pixel_count, valid_area_ha]
        result[grid_id] = [[total_area, pixel_count, pixel_count, total_area], 
                           {mv_id: [mv_data[mv_id][0], mv_data[mv_id][1], mv_data[mv_id][2]] 
                            for mv_id in mv_data}]
    
    return result


def build_coarser_grid_score_table(
        sdu_col_name, activity_list, activity_name, grid_values,
        grid_serviceshed_coverage, target_ip_table_path, baseline_table=False):
    """
    Build a table for optimization using coarser grid cell values.
    
    This function is essentially the same as build_sdu_score_table but included
    for consistency with the other methods.
    """
    # Just use the existing SDU function since the data structure is the same
    build_sdu_score_table(
        sdu_col_name, activity_list, activity_name, grid_values,
        grid_serviceshed_coverage, target_ip_table_path, baseline_table
    )


# Common function to join table to vector
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


# Function to create spatial decision units based on selected method
def create_spatial_decision_units(country_root, args):
    """
    Create spatial decision units based on the method specified in the config file.
    
    Parameters:
        country_root (string): path to the country folder
        args (dict): configuration parameters
        
    Returns:
        Path to the created spatial decision units file
    """
    # Get the spatial unit type from config
    spatial_unit_type = args.get("spatial_unit_type", "hexagon")
    
    if spatial_unit_type == "pixel":
        # Use pixel-level processing
        return create_pixel_grid(country_root, args)
    elif spatial_unit_type == "coarser_grid":
        # Use coarser grid processing
        return create_coarser_grid(country_root, args)
    else:
        # Default to hexagonal SDUs
        output_folder = os.path.join(country_root, "sdu_map")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file = os.path.join(output_folder, "sdu.shp")
        
        # Get SDU area from config (default to 0.1 if not specified)
        sdu_area = args.get("sdu_area", 0.1)
        
        create_regular_sdu_grid(
            os.path.join(country_root, "InputRasters", "current_lulc_masked.tif"),
            "hexagon",
            sdu_area,
            output_file,
            "SDUID",
            True
        )
        return output_file


# Function to aggregate values based on selected method
def aggregate_values(spatial_unit_path, id_field_name, mask_raster_path, 
                     value_raster_lookup, spatial_unit_type):
    """
    Aggregate values from rasters to spatial decision units based on the selected method.
    
    Parameters:
        spatial_unit_path (string): path to the spatial unit file
        id_field_name (string): field name for the spatial unit IDs
        mask_raster_path (string): path to mask raster
        value_raster_lookup (dict): keys are value IDs, values are paths to rasters
        spatial_unit_type (string): type of spatial unit (pixel, coarser_grid, or hexagon)
        
    Returns:
        Dictionary of aggregated values
    """
    if spatial_unit_type == "pixel":
        return aggregate_pixel_values(
            spatial_unit_path, id_field_name, mask_raster_path, value_raster_lookup)
    elif spatial_unit_type == "coarser_grid":
        return aggregate_coarser_grid_values(
            spatial_unit_path, id_field_name, mask_raster_path, value_raster_lookup)
    else:
        return aggregate_marginal_values(
            spatial_unit_path, id_field_name, mask_raster_path, value_raster_lookup)


# Function to build score table based on selected method
def build_score_table(sdu_col_name, activity_list, activity_name, aggregated_values,
                      serviceshed_coverage, target_ip_table_path, spatial_unit_type, 
                      baseline_table=False):
    """
    Build a table for optimization using the selected spatial unit method.
    
    Parameters:
        sdu_col_name (string): name of the column for spatial unit IDs
        activity_list (list): list of activity names
        activity_name (string): name of the current activity
        aggregated_values (dict): dictionary of aggregated values
        serviceshed_coverage (dict): serviceshed coverage info (or None)
        target_ip_table_path (string): path to the output CSV file
        spatial_unit_type (string): type of spatial unit (pixel, coarser_grid, or hexagon)
        baseline_table (bool): whether this is a baseline table
    """
    if spatial_unit_type == "pixel":
        build_pixel_score_table(
            sdu_col_name, activity_list, activity_name, aggregated_values,
            serviceshed_coverage, target_ip_table_path, baseline_table)
    elif spatial_unit_type == "coarser_grid":
        build_coarser_grid_score_table(
            sdu_col_name, activity_list, activity_name, aggregated_values,
            serviceshed_coverage, target_ip_table_path, baseline_table)
    else:
        build_sdu_score_table(
            sdu_col_name, activity_list, activity_name, aggregated_values,
            serviceshed_coverage, target_ip_table_path, baseline_table)