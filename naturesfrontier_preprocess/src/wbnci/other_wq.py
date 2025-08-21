import os
import shutil
from osgeo.gdal import GDT_Float32
import pygeoprocessing.geoprocessing as pygeo


def execute(args):
    """
    This is a dummy function that just grabs/renames the right raster
    from the inputs folder.
    
    args must include
    + lu_raster
    + target_folder
    + "ground_noxn_input_folder": os.path.join(input_folder, "ground_noxn"),
    + "surface_noxn_input_folder": os.path.join(input_folder, "surface_noxn"),
    + "noxn_in_drinking_water_input_folder": os.path.join(input_folder, "noxn_in_drinking_water")
    """

    # wq_vals = ["ground_noxn", "surface_noxn", "noxn_in_drinking_water"]
    wq_vals = ["noxn_in_drinking_water"]

    scenario_raster_path = args["lu_raster"]
    scenario_name = os.path.splitext(os.path.basename(scenario_raster_path))[0]
    
    output_folder = args['target_folder']
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    for wq_val in wq_vals:
        src_file = os.path.join(args[f'{wq_val}_input_folder'],
                                f"{scenario_name}_{wq_val}.tif")

                                # f"{wq_val}_{scenario_name}.tif")
        dst_file = os.path.join(output_folder, 
                                f"{scenario_name}_{wq_val}.tif")
        if os.path.isfile(src_file):
            shutil.copyfile(src_file, dst_file)
        else:
            pygeo.new_raster_from_base(
                scenario_raster_path, dst_file, GDT_Float32, [-1], [0]
            )


