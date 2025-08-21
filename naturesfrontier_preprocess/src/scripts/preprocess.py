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

from wbnci.cropland_suitability_masks import (
    suit_map_current_practices,
    suit_map_intensified_rainfed,
    suit_map_intensified_irrigated)
from wbnci.scenario_creation import (
    sustainable_current_intensity, restoration, fixedarea_intensified_rainfed,
    fixedarea_intensified_irrigated, fixedarea_bmps_rainfed,
    fixedarea_bmps_irrigated, extensification_current_practices,
    extensification_intensified_rainfed, extensification_intensified_irrigated,
    extensification_bmps_rainfed, extensification_bmps_irrigated,
    grazing_expansion, forestry_expansion, all_urban, all_econ)
from wbnci.preprocessing import (create_regular_sdu_grid, aggregate_marginal_values,
                                 build_sdu_score_table, join_table_to_grid, read_country_list)
import wbnci.grazing
import wbnci.forestry
import wbnci.carbon
import wbnci.carbon_new
import wbnci.cropland
import wbnci.biodiversity
import wbnci.transition_cost
import wbnci.nitrate_cancer_cases
import wbnci.other_wq


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
        make_aoi_task = tg_make_aoi(tg, country_root, args)
        slice_input_tasks = tg_slice_inputs(tg, country_root, args, make_aoi_task)
        mask_base_raster_task = tg_mask_base_lulc(
            tg, country_root, slice_input_tasks["base"])
        crop_suit_mask_task = tg_create_cropland_suitability_masks(
            tg, country_root, mask_base_raster_task, slice_input_tasks)
        create_sdu_task = tg_create_sdus(tg, country_root, args, mask_base_raster_task)
        create_scenario_tasks = tg_create_scenarios(
            tg, country_root, args, mask_base_raster_task, slice_input_tasks, crop_suit_mask_task)
        biodiversity_preprocessing_task = tg_biodiversity_preprocessing(
            tg, country_root, args, slice_input_tasks, create_scenario_tasks)
        create_scenario_evaluation_tasks = tg_evaluate_scenarios(
            tg, country_root, args, slice_input_tasks, create_scenario_tasks, biodiversity_preprocessing_task)
        create_results_aggregation_tasks = tg_aggregate_results(
            tg, country_root, create_sdu_task, create_scenario_tasks, create_scenario_evaluation_tasks)

    tg.close()
    tg.join()



def tg_make_aoi(tg, country_root, args):
    c = os.path.basename(country_root)
    output_file = os.path.join(country_root, "national_boundary", "national_boundary.shp")
    make_aoi_task = tg.add_task(
        func=make_aoi,
        args=[country_root, args["national_boundaries"]],
        task_name=f"make_aoi_{c}",
        target_path_list=[output_file]
    )
    return make_aoi_task


def make_aoi(country_root, global_boundary_file):
    # get country boundry shapefile
    boundaries = gpd.read_file(global_boundary_file)
    c = os.path.basename(country_root)
    cdf = boundaries[boundaries['nev_name'] == c]
    national_boundary_file = os.path.join(country_root, "national_boundary", "national_boundary.shp")
    if not os.path.isdir(os.path.dirname(national_boundary_file)):
        os.makedirs(os.path.dirname(national_boundary_file))
    cdf.to_file(national_boundary_file)
    return None


def tg_slice_inputs(tg, country_root, args, make_aoi_task):
    c = os.path.basename(country_root)
    base_raster = args["base"]["current_lulc"]
    national_boundary_file = os.path.join(country_root, "national_boundary", "national_boundary.shp")
    target_folder = os.path.join(country_root, "InputRasters")
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    
    tasks = {}
    
    for category in ["base", "scenario_creation", "cropland", "carbon", "forestry", "grazing"]:
        # For these cases we slice each of the files listed in the config dict
        target_path_list = [os.path.join(target_folder, f) for f in args[category].values() if os.path.splitext(f)[1]==".tif"]
        tasks[category] = tg.add_task(
            func=slice_inputs,
            args=[base_raster, args[category], national_boundary_file, target_folder],
            task_name=f"slice_inputs_{category}_{c}",
            target_path_list=target_path_list,
            dependent_task_list=[make_aoi_task]
        )
    
    # handle cases where we want to slice all files in a folder (assumed to be indicated by
    # `data_root` in the config file
    # In these cases, we send the sliced rasters to their own subfolder
    def slice_all_in_folder(category):
        src_files = glob.glob(os.path.join(args[category]["data_root"], "*.tif"))
        target_path_list = [os.path.join(target_folder, category, os.path.basename(sf)) for sf in src_files]
        cat_target_folder = os.path.join(target_folder, category)
        if not os.path.isdir(cat_target_folder):
            os.mkdir(cat_target_folder)
        src_raster_dict = {}
        for sf in src_files:
            k = os.path.splitext(os.path.basename(sf))[0]
            src_raster_dict[k] = sf
        t = tg.add_task(
            func=slice_inputs,
            args=[base_raster, src_raster_dict, national_boundary_file, cat_target_folder],
            task_name=f"slice_inputs_{category}_{c}",
            target_path_list=target_path_list,
            dependent_task_list=[make_aoi_task]
        )
        return t

    special_case_categories = [
        "biodiversity", "transition_cost", "noxn_in_drinking_water"]
    for c in special_case_categories:
        tasks[c] = slice_all_in_folder(c)
    

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
    

def tg_mask_base_lulc(tg, country_root, slice_base_raster_task):
    c = os.path.basename(country_root)
    national_boundary_file = os.path.join(country_root, "national_boundary", "national_boundary.shp")
    base_file = os.path.join(country_root, "InputRasters", "current_lulc.tif")
    target_file = os.path.join(country_root, "InputRasters", "current_lulc_masked.tif")
    mask_base_raster_task = tg.add_task(
        func=pygeo.mask_raster,
        args=[(base_file, 1), national_boundary_file, target_file],
        task_name=f"mask_base_lulc_{c}",
        target_path_list=[target_file],
        dependent_task_list=[slice_base_raster_task]
    )
    
    return mask_base_raster_task


def mask_base_raster(country_root, mask, src, dst):
    pygeo.mask_raster(
        base_raster_path_band=(src, 1),
        mask_vector_path=mask,
        target_mask_raster_path=dst
    )


def tg_create_cropland_suitability_masks(tg, country_root, mask_base_raster_task, slice_input_tasks):
    c = os.path.basename(country_root)
    suit_maps = ["current_practices", "intensified_rainfed", "intensified_irrigated"]
    target_files = [os.path.join(country_root, "InputRasters", "cropland_suitability_maps", 
                                 f"{sm}.tif") for sm in suit_maps]
    crop_suit_maps_task = tg.add_task(
        func=create_cropland_suitability_masks,
        args=[country_root],
        task_name=f"create_cropland_suitability_masks_{c}",
        target_path_list=target_files,
        dependent_task_list=[mask_base_raster_task, slice_input_tasks['cropland'], 
                             slice_input_tasks['scenario_creation'], slice_input_tasks['base']]
    )
    return crop_suit_maps_task


def create_cropland_suitability_masks(country_root):
    """
    Creates the cropland suitability masks which are used by the scenario creation functions
    """
    input_rasters = sorted(glob.glob(os.path.join(country_root, "InputRasters", "*.tif")))
    files = {}
    for f in input_rasters:
        key = os.path.splitext(os.path.basename(f))[0]
        files[key] = f
    target_folder = os.path.join(country_root, "InputRasters", "cropland_suitability_maps")
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    
    suit_map_current_practices(target_folder, **files)
    suit_map_intensified_rainfed(target_folder, **files)
    suit_map_intensified_irrigated(target_folder, **files)


def tg_create_sdus(tg, country_root, args, mask_base_raster_task):
    c = os.path.basename(country_root)
    output_file = os.path.join(country_root, "sdu_map", "sdu.shp")
    make_sdu_task = tg.add_task(
        func=create_sdus,
        args=[country_root, args],
        task_name=f"create_sdus_{c}",
        target_path_list=[output_file],
        dependent_task_list=[mask_base_raster_task]
    )
    return make_sdu_task


def create_sdus(country_root, args):
    target_folder = os.path.join(country_root, "sdu_map")
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    target_file = os.path.join(target_folder, "sdu.shp")
    
    if not "sdu_area" in args:
        raise ValueError("config file must contain value for 'sdu_area'")

    create_regular_sdu_grid(
        os.path.join(country_root, "InputRasters", "current_lulc_masked.tif"),
        "hexagon",
        args["sdu_area"],
        target_file,
        "SDUID",
        True
    )


def tg_create_scenarios(tg, country_root, args, mask_base_raster_task, slicing_tasks, crop_suit_mask_task):
    """
    Create the scenario rasters
    
    Returns dict:
        {scenario_name: task}, where scenario_name is the name of the scenario creation function (i.e. no ".tif")
    
    """
    c = os.path.basename(country_root)
    scenario_map_folder = os.path.join(country_root, "ScenarioMaps")
    if not os.path.isdir(scenario_map_folder):
        os.makedirs(scenario_map_folder)
    input_map_folder = os.path.join(country_root, "InputRasters")
    succession_file = args["succession_file"]
    
    scenario_files = _make_scenario_file_dict(country_root, args)
    scenario_files["base_lulc"] = os.path.join(input_map_folder, "current_lulc_masked.tif")
    scenario_files["sustainable_current"] = os.path.join(scenario_map_folder, "sustainable_current.tif")
    scenario_files["restoration"] = os.path.join(scenario_map_folder, "restoration.tif")
    scenario_files["grazing_suitability"] = os.path.join(input_map_folder, "grazing_potential_value.tif")
    scenario_files["forestry_suitability"] = os.path.join(input_map_folder, "forestry_value.tif")

    cropland_suitability_folder = os.path.join(input_map_folder, "cropland_suitability_maps")
    scenario_files["cropland_current_practices_suitability"] = os.path.join(
        cropland_suitability_folder, "current_practices.tif")
    scenario_files["cropland_intensified_rainfed_suitability"] = os.path.join(
        cropland_suitability_folder, "intensified_rainfed.tif")
    scenario_files["cropland_intensified_irrigated_suitability"] = os.path.join(
        cropland_suitability_folder, "intensified_irrigated.tif")

    scenario_tasks = {}

    # restoration
    scenario_tasks["restoration"] = tg.add_task(
        func=restoration,
        args=[scenario_map_folder, succession_file],
        kwargs=scenario_files,
        task_name=f"restoration_scenario_{c}",
        hash_algorithm="md5",
        target_path_list=[os.path.join(scenario_map_folder, "restoration.tif")],
        dependent_task_list=[mask_base_raster_task, slicing_tasks["scenario_creation"]]
    )
    
    # sustainable current
    scenario_tasks["sustainable_current"] = tg.add_task(
        func=sustainable_current_intensity,
        args=[scenario_map_folder],
        kwargs=scenario_files,
        task_name=f"sustainable_current_scenario_{c}",
        hash_algorithm="md5",
        target_path_list=[os.path.join(scenario_map_folder, "sustainable_current.tif")],
        dependent_task_list=[scenario_tasks["restoration"]]
    )
    
    # all the others
    scenario_functions = [fixedarea_intensified_rainfed,
        fixedarea_intensified_irrigated, fixedarea_bmps_rainfed,
        fixedarea_bmps_irrigated, extensification_current_practices,
        extensification_intensified_rainfed, extensification_intensified_irrigated,
        extensification_bmps_rainfed, extensification_bmps_irrigated,
        grazing_expansion, forestry_expansion, all_urban, all_econ]

    for scen_func in scenario_functions:
        scen_name = scen_func.__name__
        scenario_tasks[scen_name] = tg.add_task(
            func=scen_func,
            args=[scenario_map_folder],
            kwargs=scenario_files,
            hash_algorithm="md5",
            target_path_list=[os.path.join(scenario_map_folder, f"{scen_name}.tif")],
            dependent_task_list=[scenario_tasks["restoration"], scenario_tasks["sustainable_current"],
                                 slicing_tasks["cropland"], slicing_tasks["grazing"], slicing_tasks["forestry"],
                                 crop_suit_mask_task, slicing_tasks["scenario_creation"]]
        )

    return scenario_tasks
        

def _make_scenario_file_dict(country_root, args):
    result = {}
    for k in args["scenario_creation"].keys():
        result[k] = os.path.join(country_root, "InputRasters", f"{k}.tif")
    for k in args["cropland"].keys():
        result[k] = os.path.join(country_root, "InputRasters", f"{k}.tif")
    for k in args["forestry"].keys():
        result[k] = os.path.join(country_root, "InputRasters", f"{k}.tif")
    for k in args["grazing"].keys():
        result[k] = os.path.join(country_root, "InputRasters", f"{k}.tif")

    return result


def biodiversity_preprocessing(country_root, args):
    # biodiversity preprocessing
    lu_table_file = args["lu_table_file"]
    
    input_folder = os.path.join(country_root, "InputRasters")
    model_results_folder = os.path.join(country_root, "ModelResults")
    if not os.path.isdir(model_results_folder):
        os.makedirs(model_results_folder)

    scenario_file = os.path.join(country_root, "ScenarioMaps", f"restoration.tif")
    minValues = {
        'Richness': 0.,
        'RedList': 0.,
        'Endemics': 0.,
        'KBAs': 0.,
        'Ecoregion': 0.
    }
                
    maxValues = {
        'Richness': 0.,
        'RedList': 0.,
        'Endemics': 0.,
        'KBAs': 1.,
        'Ecoregion': 1/0.053406819701195
    }            

    biodiv_args = {
        "preProcessing": True,
        "raster_input_folder": os.path.join(input_folder, "biodiversity"),
        "lu_raster": scenario_file, 
        "restoration_raster": os.path.join(
            country_root, "ScenarioMaps", "restoration.tif"),
        "target_folder": model_results_folder,
        "lu_codes_table": lu_table_file,
        "predicts_table_1_path": args["biodiversity"]["predicts_table_1"],
        "predicts_table_2_path": args["biodiversity"]["predicts_table_2"],
        "plantation_raster": args["biodiversity"]["plantation_raster"],
        "minValues": minValues,
        "maxValues": maxValues
    }
    
    newMaxValues = wbnci.biodiversity.execute(biodiv_args)
    output_file = os.path.join(country_root, "InputRasters", "biodiversity",
                               "max_values.csv")
    
    with open(output_file, "w") as f:
        for k, v in newMaxValues.items():
            f.write(f"{k}, {v}\n")
    
    
def tg_biodiversity_preprocessing(tg, country_root, args, data_slicing_tasks, scenario_tasks):
    output_file = os.path.join(country_root, "InputRasters", "biodiversity",
                               "max_values.csv")

    biodiversity_preprocessing_task = tg.add_task(
        func=biodiversity_preprocessing,
        args=[country_root, args],
        task_name=f"biodiversity_preprocessing",
        hash_algorithm="md5",
        target_path_list=[output_file],
        dependent_task_list=[data_slicing_tasks["biodiversity"], scenario_tasks["restoration"]]
    )

    return biodiversity_preprocessing_task


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

    # biodiversity
    for (scenario_name, scenario_task) in scenario_tasks.items():
        scenario_file = os.path.join(country_root, "ScenarioMaps", f"{scenario_name}.tif")
        minValues = {
            'Richness': 0.,
            'RedList': 0.,
            'Endemics': 0.,
            'KBAs': 0.,
            'Ecoregion': 0.
        }

        biodiv_args = {
            "preProcessing": False,
            "raster_input_folder": os.path.join(input_folder, "biodiversity"),
            "lu_raster": scenario_file,
            "restoration_raster": os.path.join(
                country_root, "ScenarioMaps", "restoration.tif"),
            "target_folder": model_results_folder,
            "lu_codes_table": lu_table_file,
            "predicts_table_1_path": args["biodiversity"]["predicts_table_1"],
            "predicts_table_2_path": args["biodiversity"]["predicts_table_2"],
            "plantation_raster": args["biodiversity"]["plantation_raster"],
            "minValues": minValues,
            "maxValuesFile": os.path.join(country_root, "InputRasters", "biodiversity",
                               "max_values.csv")
        }

        target_file = os.path.join(model_results_folder, f"{scenario_name}_biodiversity.tif")
        tasks[(scenario_name, "biodiversity")] = tg.add_task(
            func=wbnci.biodiversity.execute,
            args=[biodiv_args],
            task_name=f"{scenario_name}_biodiversity",
            hash_algorithm="md5",
            target_path_list=[target_file],
            dependent_task_list=[data_slicing_tasks["biodiversity"], scenario_task,
                                 biodiversity_preprocessing_task]
        )
        
    # carbon
    for (scenario_name, scenario_task) in scenario_tasks.items():
        scenario_file = os.path.join(country_root, "ScenarioMaps", f"{scenario_name}.tif")

        carbon_args = {
            "raster_input_folder": input_folder,
            "lu_raster": scenario_file, 
            "potential_vegetation": os.path.join(input_folder, "potential_vegetation.tif"),
            "target_folder": model_results_folder,
            "lu_codes_table": lu_table_file,
            "carbon_zone_file": os.path.join(input_folder, "carbon_zone_file.tif"),
            "carbon_table_file": args["carbon"]["carbon_table_file"],
            "pixel_area": os.path.join(input_folder, "pixel_area.tif")
        }

        target_file = os.path.join(model_results_folder, f"{scenario_name}_carbon.tif")
        tasks[(scenario_name, "carbon")] = tg.add_task(
            func=wbnci.carbon_new.execute,
            args=[carbon_args],
            task_name=f"{scenario_name}_carbon",
            hash_algorithm="md5",
            target_path_list=[target_file],
            dependent_task_list=[data_slicing_tasks["carbon"], scenario_task]
        )
    
    # cropland
    for (scenario_name, scenario_task) in scenario_tasks.items():
        scenario_file = os.path.join(country_root, "ScenarioMaps", f"{scenario_name}.tif")

        cropland_args = {
            "lu_raster": scenario_file, 
            "target_folder": model_results_folder,
            "lu_codes_table": lu_table_file,
            "cropland_current_value": os.path.join(input_folder, "crop_value_current.tif"),
            "cropland_irrigated_value": os.path.join(input_folder, "crop_value_intensified_irrigated.tif"),
            "cropland_rainfed_value": os.path.join(input_folder, "crop_value_intensified_rainfed.tif"),
            "palm_oil_value": os.path.join(input_folder, "palm_oil_value.tif"),
            "pixel_area": os.path.join(input_folder, "pixel_area.tif"),
            "set_zero_floor": scenario_name == "sustainable_current"
        }

        target_file = os.path.join(model_results_folder, f"{scenario_name}_cropland_value.tif")
        tasks[(scenario_name, "cropland")] = tg.add_task(
            func=wbnci.cropland.execute,
            args=[cropland_args],
            task_name=f"{scenario_name}_cropland",
            hash_algorithm="md5",
            target_path_list=[target_file],
            dependent_task_list=[data_slicing_tasks["cropland"], scenario_task]
        )

    # forestry
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

    # grazing
    for (scenario_name, scenario_task) in scenario_tasks.items():
        scenario_file = os.path.join(country_root, "ScenarioMaps", f"{scenario_name}.tif")

        grazing_args = {
            "lu_raster": scenario_file, 
            "target_folder": model_results_folder,
            "lu_codes_table": lu_table_file,
            "current_grazing_value_raster": os.path.join(input_folder, "grazing_current_value.tif"),
            "potential_grazing_value_raster": os.path.join(input_folder, "grazing_potential_value.tif"),
            "current_grazing_methane_raster": os.path.join(input_folder, "grazing_current_methane.tif"),
            "potential_grazing_methane_raster": os.path.join(input_folder, "grazing_potential_methane.tif"),
            "pixel_area": os.path.join(input_folder, "pixel_area.tif")
            
        }

        target_file = os.path.join(model_results_folder, f"{scenario_name}_grazing_value.tif")
        tasks[(scenario_name, "grazing")] = tg.add_task(
            func=wbnci.grazing.execute,
            args=[grazing_args],
            task_name=f"{scenario_name}_grazing",
            hash_algorithm="md5",
            target_path_list=[target_file],
            dependent_task_list=[data_slicing_tasks["grazing"], scenario_task]
        )

    # nitrate
    # for (scenario_name, scenario_task) in scenario_tasks.items():
    #     scenario_file = os.path.join(country_root, "ScenarioMaps", f"{scenario_name}.tif")

    #     nitrate_cancer_cases_args = {
    #         "lu_raster": scenario_file, 
    #         "target_folder": model_results_folder,
    #         "nitrate_cancer_cases_input_folder": os.path.join(input_folder, "nitrate_cancer_cases")
    #     }

    #     target_file = os.path.join(model_results_folder, f"{scenario_name}_nitrate_cancer_cases.tif")
    #     tasks[(scenario_name, "nitrate_cancer_cases")] = tg.add_task(
    #         func=wbnci.nitrate_cancer_cases.execute,
    #         args=[nitrate_cancer_cases_args],
    #         task_name=f"{scenario_name}_nitrate_cancer_cases",
    #         hash_algorithm="md5",
    #         target_path_list=[target_file],
    #         dependent_task_list=[data_slicing_tasks["nitrate_cancer_cases"], scenario_task]
    #     )
    
    # other water quality values
    for (scenario_name, scenario_task) in scenario_tasks.items():
        scenario_file = os.path.join(country_root, "ScenarioMaps", f"{scenario_name}.tif")
        # wq_vals = ["ground_noxn", "surface_noxn", "noxn_in_drinking_water"]
        wq_vals = ["noxn_in_drinking_water"]

        other_wq_args = {
            "lu_raster": scenario_file, 
            "target_folder": model_results_folder,
            "ground_noxn_input_folder": os.path.join(input_folder, "ground_noxn"),
            "surface_noxn_input_folder": os.path.join(input_folder, "surface_noxn"),
            "noxn_in_drinking_water_input_folder": os.path.join(input_folder, "noxn_in_drinking_water")
        }

        target_files = [os.path.join(model_results_folder, f"{scenario_name}_{wq_val}.tif") for 
            wq_val in wq_vals]
        dependent_tasks = [data_slicing_tasks[wq_val] for wq_val in wq_vals] + [scenario_task]

        tasks[(scenario_name, "other_water_quality")] = tg.add_task(
            func=wbnci.other_wq.execute,
            args=[other_wq_args],
            task_name=f"{scenario_name}_other_water_quality",
            hash_algorithm="md5",
            target_path_list=target_files,
            dependent_task_list=dependent_tasks
        )

    
    # transition cost
    for (scenario_name, scenario_task) in scenario_tasks.items():
        scenario_file = os.path.join(country_root, "ScenarioMaps", f"{scenario_name}.tif")

        transition_cost_args = {
            "lu_raster": scenario_file, 
            "target_folder": model_results_folder,
            "transition_cost_input_folder": os.path.join(input_folder, "transition_cost")
        }

        target_file = os.path.join(model_results_folder, f"{scenario_name}_transition_cost.tif")
        tasks[(scenario_name, "transition_cost")] = tg.add_task(
            func=wbnci.transition_cost.execute,
            args=[transition_cost_args],
            task_name=f"{scenario_name}_transition_cost",
            hash_algorithm="md5",
            target_path_list=[target_file],
            dependent_task_list=[data_slicing_tasks["transition_cost"], scenario_task]
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
    country_name = os.path.basename(country_root)
    sdu_shpfile = os.path.join(country_root, "sdu_map", "sdu.shp")
    mask_raster = os.path.join(country_root, "ScenarioMaps", "sustainable_current.tif")

    transition_names = ["restoration", "sustainable_current",
                        "extensification_bmps_irrigated", "extensification_bmps_rainfed",
                        "extensification_current_practices", "extensification_intensified_irrigated",
                        "extensification_intensified_rainfed", "fixedarea_bmps_irrigated",
                        "fixedarea_bmps_rainfed", "fixedarea_intensified_irrigated",
                        "fixedarea_intensified_rainfed", "forestry_expansion", "grazing_expansion",
                        "all_urban", "all_econ"]
    services = ["biodiversity", "carbon", "forestry_value", "grazing_value", "grazing_methane", 
                "cropland_value", "noxn_in_drinking_water", "transition_cost"]
    # services = ["biodiversity", "carbon", "forestry_value", "grazing_value", "grazing_methane", 
    #             "cropland_value", "nitrate_cancer_cases", "ground_noxn", 
    #             "noxn_in_drinking_water", "surface_noxn", "transition_cost"]
    eval_task_names = ["biodiversity", "carbon", "cropland", "forestry", "grazing", 
                       "transition_cost", "other_water_quality"]
    
    tasks = {}
    
    for t in transition_names:
        target_file = os.path.join(output_folder, f"{t}.csv")
        task_deps = [evaluation_tasks[(t, et)] for et in eval_task_names] + [sdu_task, scenario_tasks["sustainable_current"]]
        tasks[t] = tg.add_task(
            func=_score_table_for_scenario,
            args=[t, services, model_results_folder, sdu_shpfile,
                  mask_raster, output_folder, transition_names],
            task_name=f"{country_name}_aggregate_to_sdus_{t}",
            hash_algorithm="md5",
            target_path_list=[target_file],
            dependent_task_list=task_deps
        )


def _score_table_for_scenario(transition, services, model_results_folder, sdu_shpfile, 
                              mask_raster, output_folder, transition_names):
    raster_lookup = {
        s: os.path.join(model_results_folder, f'{transition}_{s}.tif') for s in services
    }
    sdu_scores = aggregate_marginal_values(sdu_shpfile, 'SDUID', mask_raster, raster_lookup)
    table_file = os.path.join(output_folder, f'{transition}.csv')
    build_sdu_score_table('SDUID', transition_names, transition, sdu_scores, None, table_file)
    df = pd.read_csv(table_file)
    keep_cols = chain(['SDUID'], services)
    df = df[keep_cols]
    df['production_value'] = df['forestry_value'] + df['grazing_value'] + df['cropland_value']
    df['net_econ_value'] = df['production_value'] - df['transition_cost']
    # GHGs calcs: convert the carbon storage to CO2 equivalents, scale the methane from kg to tons 
    # and convert from annual production to 20 year equivalent stock
    df['net_ghg_co2e'] = 44/12 * df['carbon'] - (0.001 * 20) * df['grazing_methane']
    df.to_csv(table_file, index=False)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Usage: python preprocess.py [config file path]")
    
    if not os.path.isfile(sys.argv[1]):
        raise Exception(f"Error: config file {sys.argv[1]} does not exist")
    
    main(sys.argv[1])
