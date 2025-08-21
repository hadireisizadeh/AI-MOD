import os
from osgeo.gdal import GDT_Float32
import numpy as np
import pygeoprocessing.geoprocessing as pygeo
from wbnci.utils import read_lulc_table, basename_no_ext

def execute(args):
    """
    `args` should be a dictionary with keys:
    + `lu_raster`
    + `target_folder`
    + `lu_codes_table`
    + `forestry_value_raster`
    + `pixel_area`
    
    ! I don't have a way of distinguishing current and potential based on the LU codes
    """
    
    
    lu_codes = read_lulc_table(args['lu_codes_table'])
    forestry_codes = lu_codes['forestry_codes']
    
    src_bands = [
        (args['lu_raster'], 1),
        (args['forestry_value_raster'], 1)
    ]
    scenario_name = basename_no_ext(args['lu_raster'])
    target_file = os.path.join(args['target_folder'], f'{scenario_name}_forestry_value.tif')
    
    forestry_nodata = pygeo.get_raster_info(args['forestry_value_raster'])['nodata']
    
    def local_op(lu, fv):
        result = np.zeros(lu.shape)
        forestry_pix = np.all(
            [np.isin(lu, forestry_codes),
             fv != forestry_nodata], axis=0)
        result[forestry_pix] = fv[forestry_pix]
        
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