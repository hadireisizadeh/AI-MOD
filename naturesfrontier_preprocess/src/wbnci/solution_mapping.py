from __future__ import annotations

import os
import pathlib
import sys
import json
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import osgeo.gdal as gdal
import h5py
import pygeoprocessing.geoprocessing as geo
import taskgraph
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import TypeVar

from .lucodes import (class_to_rgb, class_to_name, class_to_hex, 
                      activity_to_rgb, activity_code_to_hex, activity_code_to_name)
from .utils import (read_lulc_table, add_color_table, tif_to_png)

PathLike = TypeVar("PathLike", str, pathlib.Path, None)


def raster_from_solution_archive(solution_file, solution_number, sdu_list, sdu_raster, 
                                 scenario_folder, optimization_scenarios, target_file,  
                                 sense="maximization"):
    """
    Creates a land use raster with embedded ColorTable from the HDF5 optimization
    solutions archive. 
    """
    results = h5py.File(solution_file, 'r')
    sol = results['solutions'][sense][:, solution_number]
    
    scenario_rasters = [os.path.join(scenario_folder, f"{s}.tif") for s in optimization_scenarios]
    scenario_raster_band_list = [(sf, 1) for sf in scenario_rasters]
        
    assignments = {-1: -1, 255: 255}
    for sdu, i in zip(sdu_list, sol):
        assignments[int(sdu)] = int(i) - 1
    
    def pixel_op(sdu, *tr):
        tchoice = np.vectorize(assignments.__getitem__)(sdu)
        result = np.empty(sdu.shape, dtype=np.int32)
        result[:] = 0
        for i, t in enumerate(tr):
            tmask = tchoice == i
            result[tmask] = t[tmask]
        return result

    geo.raster_calculator(
        [(sdu_raster, 1)] + scenario_raster_band_list,
        pixel_op,
        target_file,
        gdal.GDT_Byte,
        0,
        raster_driver_creation_tuple=('GTIFF', ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))

    )

    add_color_table(target_file, class_to_rgb)


def lulc_tif_to_png(srcfile, dstfile=None, needs_color_table=False):
    """
    Converts a raster (with color table) to a png file.

    """
    if dstfile is None:
        dstfile = os.path.splitext(srcfile)[0] + '.png'
    if needs_color_table:
        add_color_table(srcfile, class_to_rgb)

    tif_to_png(srcfile, dstfile)


def rasterize_sdu_shapefile(sdu_shp_file: PathLike, reference_raster: PathLike) -> None:
    """
    Burn "SDUID" value from `sdu_shp_file` into a new raster. The created raster
    has the same name as `sdu_shp_file` but with a ".tif" extension.
    """
    ssf_str = str(sdu_shp_file)         # need to convert to strings for pygeoprocessing
    rrf_str = str(reference_raster)
    
    target_file = os.path.splitext(ssf_str)[0] + '.tif'
    geo.new_raster_from_base(
        rrf_str,
        target_file,
        gdal.GDT_Int32,
        [-1]
    )

    geo.rasterize(
        ssf_str,
        target_file,
        option_list=["ATTRIBUTE=SDUID", "COMPRESS=PACKBITS"]
    )


def make_legend(target_file):
    proxy_artists = [
        mpatches.Patch(color=class_to_hex[c], label=class_to_name[c])
        for c in class_to_hex.keys()
    ]
    plt.legend(handles=proxy_artists)
    plt.axis('off')
    plt.savefig(target_file, bbox_inches='tight')


# def changed_area_shapefile(solution_file, sdu_file, target_file):
#     df = pd.read_csv(solution_file)
#     gdf = gpd.read_file(sdu_file)

#     df.set_index('SDUID', inplace=True)
#     df['action'] = [row.idxmax() for _, row in df.iterrows()]
#     df.reset_index(inplace=True)

#     gdf = gdf.merge(df, on='SDUID')
    
    
def lulc_to_activity(lulc_raster_file, lu_table_file, activity_raster_file, make_png=True):
    """
    Creates a raster and accompanying png image recoding the LULCs to 
    major use/activity category
    """
    lus = read_lulc_table(lu_table_file)
    
    def local_op(lu):
        result = np.zeros(lu.shape, dtype='B')
        result[np.isin(lu, lus['natural_codes'])] = 1
        result[np.isin(lu, lus['cropland_codes'])] = 2
        result[np.isin(lu, lus['grazing_codes'])] = 3
        result[np.isin(lu, lus['forestry_codes'])] = 4
        result[lu == 190] = 6
        result[lu == 210] = 7
        result[lu == 220] = 8
        return result
    
    geo.raster_calculator(
        [(lulc_raster_file, 1)],
        local_op,
        activity_raster_file,
        gdal.GDT_Byte,
        0
    )
    
    add_color_table(activity_raster_file, activity_to_rgb)

    if make_png:
        png_file = os.path.splitext(activity_raster_file)[0] + '.png'
        tif_to_png(activity_raster_file, png_file)

    
def make_activity_legend(target_file):
    proxy_artists = [
        mpatches.Patch(
            color=activity_code_to_hex[c],
            label=activity_code_to_name[c])
        for c in activity_code_to_hex.keys()
    ]
    plt.legend(handles=proxy_artists)
    plt.axis('off')
    plt.savefig(target_file, bbox_inches='tight', dpi=300)


#!--- OLD FUNCTIONS ---#
def make_raster(solution_file, sdu_raster, scenario_folder, target_file):
    print(f'Rasterizing {os.path.basename(solution_file)}')

    df = pd.read_csv(solution_file)
    df.set_index('SDUID', inplace=True)

    transitions = list(df.columns)
    tidx = {t: i for i, t in enumerate(transitions)}
    transition_raster_band_list = [(os.path.join(scenario_folder, f'{t}.tif'), 1) for t in transitions]

    assignments = {-1: -1}
    for sduid, row in df.iterrows():
        assignments[sduid] = tidx[row.idxmax()]

    def pixel_op(sdu, *tr):
        tchoice = np.vectorize(assignments.__getitem__)(sdu)
        result = np.empty(sdu.shape, dtype=np.int32)
        result[:] = -1
        for i, t in enumerate(tr):
            tmask = tchoice == i
            result[tmask] = t[tmask]
        return result

    geo.raster_calculator(
        [(sdu_raster, 1)] + transition_raster_band_list,
        pixel_op,
        target_file,
        gdal.GDT_Byte,
        255
    )

    add_color_table(target_file, class_to_rgb)
    
    
def mapped_points_callout_frontier(country_dir, mapped_points, target_file, 
                        axis_names=['agriculture_adj', 'biodiversity_adj', 'carbon_adj']):
    country_name = os.path.basename(country_dir)
    cname = country_name.replace(' ', '_')
    results_table = os.path.join(country_dir, 'OptimizationResults',
                                 f'optim_results_{cname}.csv')
    
    df = pd.read_csv(results_table)

    dbase = df[df['point'] == 'baseline']
    dfront = df[df['point'] != 'baseline']
    dmapped = dfront[dfront['point'].isin([str(p) for p in mapped_points])]
    dunmapped = dfront[~dfront['point'].isin([str(p) for p in mapped_points])]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # make scatter plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    dbase.plot(axis_names[0], axis_names[1], kind='scatter', ax=ax[0], c=colors[1])
    dunmapped.plot(axis_names[0], axis_names[1], kind='scatter', ax=ax[0], c=colors[0])
    dmapped.plot(axis_names[0], axis_names[1], kind='scatter', ax=ax[0], c=colors[3])
    ax[0].text(dbase[axis_names[0]] * .96, dbase[axis_names[1]], 'baseline',
               ha='right', va='center')
    for i, (_, row) in enumerate(dmapped.iterrows()):
        p = row['point']
        ag = row[axis_names[0]]
        bio = row[axis_names[1]]
        if i%2==0:
            bio -= 0.03
            va = 'top'
        else:
            bio += 0.03
            va = 'bottom'
        ax[0].text(ag, bio, p, ha='center', va=va)
    ax[0].set_title('Biodiversity vs Ag Value')
    ax[0].set_xlabel('Net production ($ crop value - N cost)')
    ax[0].set_ylabel('Biodiversity (e.g. number of species)')
    dbase.plot(axis_names[0], axis_names[2], kind='scatter', ax=ax[1], c=colors[1])
    dunmapped.plot(axis_names[0], axis_names[2], kind='scatter', ax=ax[1], c=colors[0])
    dmapped.plot(axis_names[0], axis_names[2], kind='scatter', ax=ax[1], c=colors[3])
    ax[1].text(dbase[axis_names[0]] * .96, dbase[axis_names[2]], 'baseline',
               ha='right', va='center')
    for i, (_, row) in enumerate(dmapped.iterrows()):
        p = row['point']
        ag = row[axis_names[0]]
        ca = row[axis_names[2]]
        if i%2==0:
            ca += 0.03
            va = 'bottom'
        else:
            ca -= 0.03
            va = 'top'
        ax[1].text(ag, ca, p, ha='center', va=va)
    ax[1].set_title('Carbon vs Ag Value')
    ax[1].set_xlabel('Net production ($ crop value - N cost)')
    ax[1].set_ylabel('Carbon ($ social value of carbon storage)')

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    plt.savefig(target_file, bbox_inches='tight', dpi=300)
    plt.close()
