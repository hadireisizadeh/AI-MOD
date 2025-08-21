from __future__ import annotations
import os
import sys
import json
import glob
import math
from typing import List
from collections import defaultdict
import shutil
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import osgeo.gdal as gdal
import osgeo.ogr as ogr
import osgeo.osr as osr
# import gdal, ogr, osr
import pygeoprocessing.geoprocessing as pygeo

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

    :param source_file:
    :param target_file:
    :param transition_codes:
    :param new_code:
    :param all_codes:
    :return:
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

    :param source_file:
    :param target_file:
    :param transition_codes:
    :param pv_raster_file:
    :param all_codes:
    :return:
    """

    source_nodata = pygeo.get_raster_info(source_file)['nodata'][0]
    # pv_nodata = pygeo.get_raster_info(pv_raster_file)['nodata'][0]

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
    The purpose of this function is to construct the spatial union of a set of mask rasters.
    Creates a new raster at `target_file` which has value 1 for a pixel if any of the rasters
    in `mask_list` have a valid (non-NoData) value for that pixel and nodata elsewhere.

    :param mask_list:
    :param target_file:
    :return:
    """
    brpbcl = [(p, 1) for p in mask_list]  #base_raster_path_band_const_list

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
    Intersects the raster files in `mask_list`.
    Creates a new raster at `target_file` which has the value 1 for a pixel if
    all of the rasters in `mask_list` have a valid (non-NoData or 0) value for that pixel
    and has value NoData elsewhere.
    """
    brpbcl = [(p, 1) for p in mask_list]  #base_raster_path_band_const_list

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

    pygeo.new_raster_from_base(
        mask_raster_path,
        id_raster_path,
        gdal.GDT_Int32,
        [-1],
        fill_value_list=[-1],
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
    # TODO: drop activity mask, get activity from the rasters - require nodata for non-transition pixels

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


def join_table_to_grid(grid_path, table_path, target_file):
    """

    :param grid_path:
    :param table_path:
    :param target_file:
    :return:
    """

    df = pd.read_csv(table_path)
    gdf = gpd.read_file(grid_path)
    gdf = pd.merge(gdf, df, on='SDUID')
    gdf.to_file(target_file)