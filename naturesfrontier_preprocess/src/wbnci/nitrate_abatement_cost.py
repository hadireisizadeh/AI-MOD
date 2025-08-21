import glob
import os
import pandas as pd
import numpy as np
import osgeo.gdal as gdal
from osgeo.gdal import GDT_Float32
import pygeoprocessing.geoprocessing as pygeo
from wbnci.utils import read_lulc_table, basename_no_ext


ag_lucodes = [10, 11, 12, 15, 16, 20, 25, 26, 30, 40]
AG_BASELINE_LIST = [10, 11, 12, 20, 30, 40]
AG_INTENSIFIED_LIST = [15, 16, 25, 26]
natural_lucodes = [50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110, 120, 121,
                   122, 130, 150, 151, 152, 153, 180]


def execute(args):
    data_dir = args['country_folder']
    output_folder = args['output_folder']
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    base_ag_file = os.path.join(data_dir, 'Projected', 
                                  'abatement_costs_ag_expansion.tif')
    intense_ag_file = os.path.join(data_dir, 'Projected', 
                                    'abatement_costs_ag_intensification.tif')
    natural_cover_file = os.path.join(data_dir, 'Projected',
                                      'abatement_costs_restoration.tif')
    
    # Scenario LULC layers
    scenario_files = glob.glob(os.path.join(args['scenario_folder'], '*.tif'))

    for sf in scenario_files:
        score_scenario(sf, base_ag_file, intense_ag_file, natural_cover_file,
                       output_folder)


def score_scenario(scenario_file, base_ag_file, intense_ag_file, natural_cover_file,
                   output_folder):
    """
    Ag value is scored by looking up the appropriate value in the pre-computed
    production value rasters. If the LU code is in `AG_BASELINE_LIST`, then the
    value is taken from `base_prod_file`, if the code is == `AG_INTENSIFIED_RAINFED`,
    the value is taken from `intense_rf_file`, and if the code is == 
    `AG_INTENSIFIED_IRRIGATED` the value is taken from `intense_irr_file`. Otherwise
    the value given is 0.
    """

    nodata = -99999.0
    skey = os.path.splitext(os.path.basename(scenario_file))[0]
    print(f'scoring {skey} for nitrates')
    target_raster_path = os.path.join(output_folder, f'{skey}_nitrate.tif')
    
    def local_op(sf, base, intense, nat):
        result = np.zeros(sf.shape)
        result[np.isin(sf, AG_BASELINE_LIST)] = base[np.isin(sf, AG_BASELINE_LIST)]
        result[np.isin(sf, AG_INTENSIFIED_LIST)] = intense[np.isin(sf, AG_INTENSIFIED_LIST)]
        result[np.isin(sf, natural_lucodes)] = nat[np.isin(sf, natural_lucodes)]
        return result

    pygeo.raster_calculator(
        [(scenario_file, 1), (base_ag_file, 1), (intense_ag_file, 1), (natural_cover_file, 1)],
        local_op,
        target_raster_path,
        GDT_Float32,
        nodata
    )
