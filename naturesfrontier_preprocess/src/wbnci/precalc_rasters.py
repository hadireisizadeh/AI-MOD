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
    + source_dir
    """
    scenario_raster_path = args["lu_raster"]
    scenario_name = os.path.splitext(os.path.basename(scenario_raster_path))[0]
    
    output_folder = args['target_folder']
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    src_file = os.path.join(args['source_dir'],
                            f"cancer_cases_{scenario_name}.tif")
    dst_file = os.path.join(output_folder, 
                            f"{scenario_name}_nitrate_cancer_cases.tif")
    if os.path.isfile(src_file):
        shutil.copyfile(src_file, dst_file)
    else:
        pygeo.new_raster_from_base(
            scenario_raster_path, dst_file, GDT_Float32, [-1], [0]
        )


