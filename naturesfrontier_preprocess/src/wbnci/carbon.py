import os
import glob
import pandas as pd
import numpy as np
import osgeo.gdal as gdal
import pygeoprocessing.geoprocessing as pygeo
from .carbon_unilever_cython_functions import write_carbon_table_to_array

# note to self: to compile cython function:
# import numpy in python console
# run numpy.get_include()
# in terminal, export CFLAGS=-I/numpy/path
# cythonize -a -i carbon_unilever_cython_functions.pyx


def execute(args):
    """
    args should contain:
    + lu_raster
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
    row_names = {int(v): int(c) for c, v in enumerate(lookup_table_df.index)}
    col_names = {int(v): int(c) for c, v in enumerate(lookup_table_df.columns)}

    sfile = args["lu_raster"]
    sname = os.path.splitext(os.path.basename(sfile))[0]
        
    carbon_output_path = os.path.join(output_folder, f'{sname}_carbon.tif')
    
    lulc_array, lulc_array_nodata = read_to_array(sfile)
    lulc_array = lulc_array.astype(np.float32)
    lulc_array[lulc_array == lulc_array_nodata] = np.nan
    
    carbon_zones_array, cz_nodata = read_to_array(carbon_zones_path)
    carbon_zones_array = carbon_zones_array.astype(np.float32)
    carbon_zones_array[carbon_zones_array == cz_nodata] = np.nan

    pixel_area, pixel_area_nodata = read_to_array(args['pixel_area'])
    pixel_area = pixel_area.astype(np.float32) * 100  # convert km2 to ha
    pixel_area[pixel_area == pixel_area_nodata] = np.nan

    output_carbon = write_carbon_table_to_array(
        lulc_array,
        carbon_zones_array,
        pixel_area,
        lookup_table,
        row_names,
        col_names,
    )

    pygeo.new_raster_from_base(carbon_zones_path, carbon_output_path, gdal.GDT_Float32, [-9999])
    outds = gdal.OpenEx(carbon_output_path, 1)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(output_carbon)
    outband.FlushCache()
    outband = None
    del outds




def batch(args):
    cfolder = args['country_folder']
    output_folder = args['output_folder']
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    carbon_table_path = os.path.join(cfolder, "exhaustive_carbon_table.csv")
    carbon_zones_path = os.path.join(cfolder, 'Projected', 'carbon', 'carbon_zones.tif')

    # set up lookup tables
    lookup_table_df = pd.read_csv(carbon_table_path, index_col=0)
    table_shape = (len(lookup_table_df.index), len(lookup_table_df.columns))
    lookup_table = np.float32(lookup_table_df.values)
    row_names = {int(v): int(c) for c, v in enumerate(lookup_table_df.index)}
    col_names = {int(v): int(c) for c, v in enumerate(lookup_table_df.columns)}
    # print('row_names', row_names)
    # print('col_names', col_names)

    # set up scenarios
    scenario_folder = args['scenario_folder']
    scenario_files = glob.glob(os.path.join(scenario_folder, '*.tif'))
    scenario_names = [os.path.splitext(os.path.basename(sf))[0] for sf in scenario_files]

    for sname, sfile in zip(scenario_names, scenario_files):
        print(sname)
        lulc_input_path = sfile
        carbon_output_path = os.path.join(output_folder, f'{sname}_carbon.tif')
        print('carbon_output_path', carbon_output_path)
        lulc_array, lulc_array_nodata = read_to_array(sfile)
        lulc_array = lulc_array.astype(np.float32)
        lulc_array[lulc_array == lulc_array_nodata] = np.nan
        print('lulc_array', lulc_array.shape)
        carbon_zones_array, cz_nodata = read_to_array(carbon_zones_path)
        carbon_zones_array = carbon_zones_array.astype(np.float32)
        carbon_zones_array[carbon_zones_array == cz_nodata] = np.nan

        output_carbon = write_carbon_table_to_array(
            lulc_array,
            carbon_zones_array,
            lookup_table,
            row_names,
            col_names,
        )

        print(output_carbon.shape)

        pygeo.new_raster_from_base(carbon_zones_path, carbon_output_path, gdal.GDT_Float32, [-9999])
        outds = gdal.OpenEx(carbon_output_path, 1)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(output_carbon)
        outband.FlushCache()
        outband = None
        del outds


def read_to_array(tifpath):
    ds = gdal.OpenEx(tifpath)
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    a = band.ReadAsArray()
    return a, nodata


if __name__ == '__main__':
    args = {
        'country_folder': "/Users/hawt0010/Projects/WBNCI/scratch/packages/Belize",
        'output_folder': "/Users/hawt0010/Projects/WBNCI/scratch/packages/Belize/ModelResults",
        'scenario_folder': "/Users/hawt0010/Projects/WBNCI/scratch/packages/Belize/Projected/Scenarios"
    }
    execute(args)
