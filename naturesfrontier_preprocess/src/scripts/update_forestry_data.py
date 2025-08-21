import os
import sys
import shutil
import glob
import yaml
from itertools import chain, count
import pandas as pd
import geopandas as gpd
import pygeoprocessing.geoprocessing as pygeo
import taskgraph
from osgeo.gdal import SetCacheMax

from wbnci.preprocessing import (create_regular_sdu_grid, aggregate_marginal_values,
                                 build_sdu_score_table, join_table_to_grid, read_country_list)
import wbnci.forestry


def main(config_file):
    
    with open(config_file, 'r') as f:
        args = yaml.safe_load(f)
    
    output_dir = args["workspace"]
    if os.path.isfile(args["country_list"]):
        countries = read_country_list(args["country_list"])
    else:
        countries = [c.strip() for c in args["country_list"].split(",")]
    
    if "n_workers" in args:
        n_workers = int(args["n_workers"])
    else:
        n_workers = 8
    
    SetCacheMax(2**27)  # this keeps GDAL from caching too much, which may be causing OOM errors on MSI
    
    tg = taskgraph.TaskGraph(output_dir, n_workers=n_workers)

    for country in countries:
        print(country)
        country_root = os.path.join(output_dir, country)
        if not os.path.isdir(country_root):
            os.makedirs(country_root)
        shutil.copyfile(config_file, os.path.join(country_root, os.path.basename(config_file)))
        slice_input_tasks = tg_slice_inputs(tg, country_root, args)
        create_scenario_evaluation_tasks = tg_evaluate_scenarios(
            tg, country_root, args, slice_input_tasks)
        create_results_aggregation_tasks = tg_aggregate_results(
            tg, country_root, create_scenario_evaluation_tasks)

    tg.close()
    tg.join()



def tg_slice_inputs(tg, country_root, args, make_aoi_task):
    c = os.path.basename(country_root)
    base_raster = args["base"]["current_lulc"]
    national_boundary_file = os.path.join(country_root, "national_boundary", "national_boundary.shp")
    target_folder = os.path.join(country_root, "InputRasters")
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    
    tasks = {}
    
    for category in ["forestry"]:
        # target_path_list = [os.path.join(target_folder, f"{k}.tif") for k in args[category]]
        target_path_list = [os.path.join(target_folder, f) for f in args[category].values() if os.path.splitext(f)[1]==".tif"]
        tasks[category] = tg.add_task(
            func=slice_inputs,
            args=[base_raster, args[category], national_boundary_file, target_folder],
            task_name=f"slice_inputs_{category}_{c}",
            target_path_list=target_path_list,
            dependent_task_list=[make_aoi_task]
        )

    return tasks
    
    
def slice_inputs(base_raster, source_raster_dict, national_boundary_file, target_folder):
    """
    Assumes that `source_raster_dict` will be keyed as new_name: orig_path
    """
    base_lulc_info = pygeo.get_raster_info(base_raster)
    
    src_paths = []
    dst_paths = []
    for k, v in source_raster_dict.items():
        if os.path.splitext(v)[1] == ".tif":
            src_paths.append(v)
            dst_paths.append(os.path.join(target_folder, f'{k}.tif'))
    
    pygeo.align_and_resize_raster_stack(
        base_raster_path_list=src_paths,
        target_raster_path_list=dst_paths,
        resample_method_list=['near' for _ in src_paths],
        target_pixel_size=base_lulc_info['pixel_size'],
        bounding_box_mode="intersection",
        base_vector_path_list=[national_boundary_file]
    )


def tg_evaluate_scenarios(tg, country_root, args, data_slicing_tasks, scenario_tasks,
                          biodiversity_preprocessing_task):
    """
    Add a task for each model x scenario pair. Dependent on appropriate data slicing and making
    scenarios.
    
    tg_evaluate_scenarios(
            tg, country_root, slice_input_tasks, create_scenario_tasks)
    """
    
    lu_table_file = args["lu_table_file"]
    
    input_folder = os.path.join(country_root, "InputRasters")
    model_results_folder = os.path.join(country_root, "ModelResults")
    if not os.path.isdir(model_results_folder):
        os.makedirs(model_results_folder)

    tasks = {}

    for (scenario_name, scenario_task) in scenario_tasks.items():
        scenario_file = os.path.join(country_root, "ScenarioMaps", f"{scenario_name}.tif")

        forestry_args = {
            "lu_raster": scenario_file, 
            "target_folder": model_results_folder,
            "lu_codes_table": lu_table_file,
            "forestry_value_raster": os.path.join(input_folder, "forestry_value.tif"),
            "pixel_area": os.path.join(input_folder, "pixel_area.tif")
            
        }

        target_file = os.path.join(model_results_folder, f"{scenario_name}_forestry_value.tif")
        tasks[(scenario_name, "forestry")] = tg.add_task(
            func=wbnci.forestry.execute,
            args=[forestry_args],
            task_name=f"{scenario_name}_forestry",
            hash_algorithm="md5",
            target_path_list=[target_file],
            dependent_task_list=[data_slicing_tasks["forestry"], scenario_task]
        )
    
    return tasks


def tg_aggregate_results(tg, country_root, sdu_task, scenario_tasks, evaluation_tasks):
    """
    Create the Value Tables
    """
    
    output_folder = os.path.join(country_root, "ValueTables")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    model_results_folder = os.path.join(country_root, "ModelResults")
    sdu_shpfile = os.path.join(country_root, "sdu_map", "sdu.shp")
    mask_raster = os.path.join(country_root, "ScenarioMaps", "sustainable_current.tif")

    transition_names = ["restoration", "sustainable_current",
                        "extensification_bmps_irrigated", "extensification_bmps_rainfed",
                        "extensification_current_practices", "extensification_intensified_irrigated",
                        "extensification_intensified_rainfed", "fixedarea_bmps_irrigated",
                        "fixedarea_bmps_rainfed", "fixedarea_intensified_irrigated",
                        "fixedarea_intensified_rainfed", "forestry_expansion", "grazing_expansion",
                        "all_urban", "all_econ"]
    services = ["forestry_value"]
    eval_task_names = ["forestry"]
    
    tasks = {}
    
    for t in transition_names:
        target_file = os.path.join(output_folder, f"{t}.csv")
        task_deps = [evaluation_tasks[(t, et)] for et in eval_task_names] + [sdu_task, scenario_tasks["sustainable_current"]]
        tasks[t] = tg.add_task(
            func=_update_value_table,
            args=[t, services, model_results_folder, sdu_shpfile,
                  mask_raster, output_folder, transition_names],
            task_name=f"aggregate_to_sdus_{t}",
            hash_algorithm="md5",
            target_path_list=[target_file],
            dependent_task_list=task_deps
        )


def _update_value_table(transition, services, model_results_folder, sdu_shpfile, 
                        mask_raster, output_folder, transition_names):
    raster_lookup = {
        s: os.path.join(model_results_folder, f'{transition}_{s}.tif') for s in services
    }
    sdu_scores = aggregate_marginal_values(sdu_shpfile, 'SDUID', mask_raster, raster_lookup)
    orig_table_file = os.path.join(output_folder, f'{transition}.csv')
    update_table_file = os.path.join(output_folder, f'{transition}_update.csv')
    new_table_file = os.path.join(output_folder, f'{transition}_new.csv')
    build_sdu_score_table('SDUID', transition_names, transition, sdu_scores, None, update_table_file)
    df = pd.read_csv(orig_table_file)
    updatedf = pd.read_csv(update_table_file)
    for s in services:
        df[s] = updatedf[s]
    df.to_csv(new_table_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Usage: python preprocess.py [config file path]")
    
    if not os.path.isfile(sys.argv[1]):
        raise Exception(f"Error: config file {sys.argv[1]} does not exist")
    
    main(sys.argv[1])
