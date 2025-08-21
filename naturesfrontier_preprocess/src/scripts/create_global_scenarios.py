import os
import sys
import glob
from itertools import chain
import pandas as pd
import geopandas as gpd
import pygeoprocessing.geoprocessing as pygeo
import taskgraph

from wbnci.scenario_creation import (
    sustainable_current_intensity, restoration, fixedarea_intensified_rainfed,
    fixedarea_intensified_irrigated, fixedarea_bmps_rainfed,
    fixedarea_bmps_irrigated, extensification_current_practices,
    extensification_intensified_rainfed, extensification_intensified_irrigated,
    extensification_bmps_rainfed, extensification_bmps_irrigated,
    grazing_expansion, forestry_expansion
    )
from wbnci.preprocessing import (create_regular_sdu_grid, aggregate_marginal_values,
                                 build_sdu_score_table, join_table_to_grid, read_country_list)

global_data_root = "/Users/peterhawthorne/Projects/WBNCI/data/"
scenario_data_src_folder = "/Users/peterhawthorne/Projects/WBNCI/data/scenario_construction"
crop_value_src_folder = "/Users/peterhawthorne/Projects/WBNCI/data/agriculture/RevKAligned"
biodiv_src_folder = "/Users/peterhawthorne/Projects/WBNCI/data/biodiversity/GlobalLayers"
transition_cost_src_folder = "/Users/peterhawthorne/Projects/WBNCI/data/transition_costs"


src_files = {
    "reference_lulc": "/Users/peterhawthorne/Projects/WBNCI/data/current_lulc/modifiedESA_feb2.tif",
    "global_lulc": "/Users/peterhawthorne/Projects/WBNCI/data/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif",
    "potential_vegetation": 
        "/Users/peterhawthorne/Projects/WBNCI/data/scenario_construction/potential_vegetation/potential_vegetation.tif",
    "protected_areas":
        "/Users/peterhawthorne/Projects/WBNCI/data/protected_areas/wdpa_merged.tif",
    "sustainable_irrigation":
        os.path.join(scenario_data_src_folder, "ag_irrigation_potential", "sustainable_irrigation_mask.tif"),
    "grazing_suitability": "/Users/peterhawthorne/Projects/WBNCI/data/grazing/potential_meat_revenue_per_ha_d66e8f2b4cc07c5bd3ac354a50be2ee2.tif"
}

def main():
    scenario_map_folder = "/Users/peterhawthorne/Projects/WBNCI/data/global_scenario_maps"
    if not os.path.isdir(scenario_map_folder):
        os.mkdir(scenario_map_folder)

    input_files = align_rasters(src_files)
    
    succession_file = "/Users/peterhawthorne/Projects/WBNCI/data/potential_vegetation/succession_matrix.csv"
    input_files["sustainable_current"] = os.path.join(scenario_map_folder, "sustainable_current.tif")
    input_files["restoration"] = os.path.join(scenario_map_folder, "restoration.tif")
    input_files["base_lulc"] = input_files["global_lulc"]

    restoration(input_files, scenario_map_folder, succession_file)
    sustainable_current_intensity(input_files, scenario_map_folder)
    grazing_expansion(input_files, scenario_map_folder)


def align_rasters(raster_dict):
    workspace = "/Users/peterhawthorne/Projects/WBNCI/data/global_scenario_maps/inputs"
    if not os.path.isdir(workspace):
        os.mkdir(workspace)
    
    raster_names = ["global_lulc", "potential_vegetation", "protected_areas", 
                    "sustainable_irrigation", "grazing_suitability"]
    base_raster_path_list = [
        raster_dict[k] for k in raster_names
    ]
    target_raster_path_list = [
        os.path.join(workspace, f'{raster}.tif') for raster in raster_names
    ]
    resample_method_list = ['near' for _ in raster_names]
    reference_info = pygeo.get_raster_info(raster_dict['reference_lulc'])
    
    
    # pygeo.align_and_resize_raster_stack(
    #     base_raster_path_list, 
    #     target_raster_path_list,
    #     resample_method_list,
    #     reference_info['pixel_size'],
    #     reference_info['bounding_box']
    # )
    
    results = {
        raster: os.path.join(workspace, f'{raster}.tif') for raster in raster_names
    }
    return results


if __name__ == "__main__":
    main()