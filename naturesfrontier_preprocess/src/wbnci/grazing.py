import os
from osgeo.gdal import GDT_Float32
import numpy as np
import pygeoprocessing.geoprocessing as pygeo
from wbnci.utils import read_lulc_table, basename_no_ext

KM2_TO_HA = 100

def execute(args):
    """
    `args` should be a dictionary with keys:
    + `target_folder`
    + `lu_raster`
    + `lu_codes_table`
    + `current_grazing_value_raster`
    + `potential_grazing_value_raster`
    + `potential_grazing_methane_raster`
    + `pixel_area`
    
    ! I don't have a way of distinguishing current and potential based on the LU codes
    ! I'm assuming that everything goes to potential
    """
    
    calculate_grazing_value(args)
    calculate_grazing_methane(args)


def calculate_grazing_value(args):
    lu_codes = read_lulc_table(args['lu_codes_table'])
    grazing_codes = lu_codes['grazing_codes']
    
    src_bands = [
        (args['lu_raster'], 1),
        (args['current_grazing_value_raster'], 1),
        (args['potential_grazing_value_raster'], 1),
        (args['pixel_area'], 1)
    ]
    scenario_name = basename_no_ext(args['lu_raster'])
    target_file = os.path.join(args['target_folder'], f'{scenario_name}_grazing_value.tif')
    
    def local_op(lu, cgv, pgv, pa):
        result = np.zeros(lu.shape)
        grazing_pix = np.all([
            np.isin(lu, grazing_codes),
            pgv > 0], axis=0)
        result[grazing_pix] = pgv[grazing_pix] * pa[grazing_pix] * KM2_TO_HA
        
        return result
    
    pygeo.raster_calculator(
        src_bands,
        local_op,
        target_file,
        GDT_Float32,
        0.0,
        raster_driver_creation_tuple=('GTIFF', ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
    )


def calculate_grazing_methane(args):
    lu_codes = read_lulc_table(args['lu_codes_table'])
    grazing_codes = lu_codes['grazing_codes']
    
    src_bands = [
        (args['lu_raster'], 1),
        (args['current_grazing_methane_raster'], 1),
        (args['potential_grazing_value_raster'], 1),
        (args['potential_grazing_methane_raster'], 1),
        (args['pixel_area'], 1)
    ]
    scenario_name = basename_no_ext(args['lu_raster'])
    target_file = os.path.join(args['target_folder'], f'{scenario_name}_grazing_methane.tif')
    
    def local_op(lu, cgm, pgv, pgm, pa):
        result = np.zeros(lu.shape)
        grazing_pix = np.all([
            np.isin(lu, grazing_codes),
            pgv > 0], axis=0)
        result[grazing_pix] = pgm[grazing_pix] * pa[grazing_pix] * KM2_TO_HA
        
        return result
    
    pygeo.raster_calculator(
        src_bands,
        local_op,
        target_file,
        GDT_Float32,
        0.0,
        raster_driver_creation_tuple=('GTIFF', ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
    )
