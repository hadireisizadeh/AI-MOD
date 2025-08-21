from __future__ import annotations

import os
import shutil
import json
import numpy as np
import pandas as pd
import osgeo.gdal as gdal

def read_to_array(tifpath, dtype=None):
    ds = gdal.OpenEx(tifpath)
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    a = band.ReadAsArray(buf_type=dtype)
    return a, nodata


def remove_packages_from_run(config_file):
    with open(config_file) as f:
        args = json.load(f)
    
    workspace = os.path.join(args['working_dir'], 'packages')
    df = pd.read_csv(args['country_list_file'])
    for country in df['nev_name']:
        cdir = os.path.join(workspace, country)
        print(cdir)
        if os.path.isdir(cdir):
            shutil.rmtree(cdir)


def remove_optimization_results_from_run(config_file):
    with open(config_file) as f:
        args = json.load(f)
    
    workspace = os.path.join(args['working_dir'], 'packages')
    country_list = read_country_list(args['country_list_file'])
    for country in country_list:
        optim_dir = os.path.join(workspace, country, "OptimizationResults")
        if os.path.isdir(optim_dir):
            shutil.rmtree(optim_dir)
        

def basename_no_ext(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def read_country_list(argvalue):
    """
    Add some flexibility to the config file by allowing a list of country names
    in addition to the path to a file with the list.
    """
    if isinstance(argvalue, list):
        return argvalue
    else:
        if not os.path.isfile(argvalue):
            raise ValueError("args['country_list_file'] is not a list of country names or path to an existing file")
        else:
            df = pd.read_csv(argvalue)
            return list(df['nev_name'])


def read_lulc_table(filename):
    """
    Reads `filename` and returns a dict of np.arrays with keys:
        + `all_codes`
        + `base_codes`
        + `natural_codes`
        + `fixed_codes`
        + `cropland_codes`
        + `grazing_codes`
        + `forestry_codes`
        + `irrigated_cropland_codes`
        + `rainfed_cropland_codes`
        + `forestry_eligible_codes`
    
    Expects `filename` to have columns `Code`, `Fixed`, `Natural`, `Cropland`, `Grazing`, `Forestry`, and `Irrigated`
    """
    df = pd.read_csv(filename)
    
    def get_codes(df, col):
        return np.array(df[df[col] == "Yes"]["Code"])
    
    result = {
        'all_codes': np.array(df['Code']),
        'base_codes': np.array(df['BaseCode']),
        'fixed_codes': get_codes(df, "Fixed"),
        'natural_codes': get_codes(df, "Natural"),
        'cropland_codes': get_codes(df, "Cropland"),
        'grazing_codes': get_codes(df, "Grazing"),
        'forestry_codes': get_codes(df, "Forestry"),
        'forestry_eligible_codes': np.unique(df[df['Forestry'] == 'Yes']['BaseCode']),
        'irrigated_cropland_codes': get_codes(df, "Irrigated"),
        'rainfed_cropland_codes': np.array( [c for c in get_codes(df, "Cropland") 
                                           if c not in get_codes(df, "Irrigated")] )
    }
    
    df['multi'] = np.any([
        np.all([df['Cropland'] == "Yes", df['Grazing'] == "Yes"], axis=0),
        np.all([df['Cropland'] == "Yes", df['Forestry'] == "Yes"], axis=0),
        np.all([df['Grazing'] == "Yes", df['Forestry'] == "Yes"], axis=0)], axis=0)

    result['multiuse_codes'] = np.array(df['multi'])
    
    return result



def add_color_table(raster_file: str, color_dict: dict[int, tuple[int, int, int]]):
    """
    Adds a ColorTable, `color_dict`, to geotiff `raster_file`. 
    
    `color_dict` should contain `lucode: RGB` pairs. 
    
    """
    ds = gdal.OpenEx(raster_file, 1)
    band = ds.GetRasterBand(1)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    colors = gdal.ColorTable()
    for k, rgb in color_dict.items():
        colors.SetColorEntry(k, rgb)

    band.SetRasterColorTable(colors)
    band = None
    ds = None

    del band, ds


def tif_to_png(srcfile: str, dstfile: str) -> None:
    """
    Takes a geotiff, `srcfile`, and converts it to a png image file, `dstfile`.
    
    `srcfile` should have an included ColorTable. 
    """
    driver = gdal.GetDriverByName("PNG")
    src = gdal.Open(srcfile)
    ds = driver.CreateCopy(dstfile, src)
    ds = None
    src = None


def get_headers(table_file: str, drop_cols: list[str] = [], sep: str = ",") -> list[str]:
    """
    Read first line of `table_file` as csv (default) and optionally, remove any column names
    included in `drop_cols`.
    """
    with open(table_file) as f:
        header = next(f)
    cols = [c.strip() for c in header.split(sep)]
    for c in drop_cols:
        if c in cols:
            cols.remove(c)
    return cols
