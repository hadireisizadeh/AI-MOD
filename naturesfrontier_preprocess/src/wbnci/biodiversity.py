import glob
import os
import shutil
import pandas as pd
import numpy as np
import osgeo.gdal as gdal
from osgeo.gdal import GDT_Float32
import pygeoprocessing.geoprocessing as pygeo

globalRichness = {
    'Amphibians': 6631,
    'Birds': 10424,
    'Mammals': 5709,
    'Reptiles': 6416
}






taxa_types = ['Amphibians', 'Birds', 'Mammals', 'Reptiles']



def execute(args):
    """
    args dict must include:
    * lu_raster
    * raster_input_folder
    * predicts_table_1_path
    * predicts_table_2_path
    * plantation_raster
    * output_folder
    * preprocessing t/f
    * minValues dict
    * maxValues dict
    """
    
    scenario_raster_path = args["lu_raster"]
    scenario_name = os.path.splitext(os.path.basename(scenario_raster_path))[0]
    
    restoration_raster_path = os.path.join(os.path.dirname(scenario_raster_path), 'restoration.tif')
    
    output_folder = args['target_folder']

    workspace = os.path.join(output_folder, f"{scenario_name}_workspace")
    if not os.path.isdir(workspace):
        os.makedirs(workspace)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    raster_dir = args["raster_input_folder"]
    
    # Reference LULC layers
    lulc_minus10 = os.path.join(raster_dir, 'band14.tif')
    lulc_base = os.path.join(raster_dir, 'band1.tif')

    # Biodiversity Metric Layers
    species_pool_files = {taxa: os.path.join(raster_dir, f'{taxa}.tif')
                          for taxa in taxa_types}
                          
    redList_pool_files = {taxa: os.path.join(raster_dir, f'{taxa}RL.tif')
                          for taxa in taxa_types}	
    
    endemics_pool_files = {taxa: os.path.join(raster_dir, f'{taxa}Endemic.tif')
                          for taxa in taxa_types}	
    kba_files = os.path.join(raster_dir, 'KBAs.tif')
    ecoregion_files = os.path.join(raster_dir, 'ecoMaps.tif')
    forest_intactness_files = os.path.join(raster_dir, "flii.tif")
    # plantation_files = os.path.join(raster_dir, "plantation_map_may27.tif")
    plantation_files = os.path.join(raster_dir, os.path.basename(args["plantation_raster"]))

    # Load parameter tables
    etp_df = pd.read_csv(args["predicts_table_1_path"])
    etp_df['pcode'] = etp_df['code0'] + etp_df['code1']
    ptsr_df = pd.read_csv(args["predicts_table_2_path"])

    if "maxValuesFile" in args:
        args['maxValues'] = {}
        with open(args['maxValuesFile']) as f:
            for line in f:
                k, v = line.strip().split(",")
                args['maxValues'][k] = float(v)
    
    if args['preProcessing'] == True:
        maxValues = score_scenario(
                        scenario_raster_path, lulc_minus10, lulc_base, restoration_raster_path,
                        species_pool_files, redList_pool_files, endemics_pool_files,
                        kba_files, ecoregion_files, forest_intactness_files, plantation_files,
                        etp_df, ptsr_df, 
                        workspace, output_folder, args)
        
        return maxValues 
    else:
        score_scenario(
                        scenario_raster_path, lulc_minus10, lulc_base, restoration_raster_path,
                        species_pool_files, redList_pool_files, endemics_pool_files,
                        kba_files, ecoregion_files, forest_intactness_files, plantation_files,
                        etp_df, ptsr_df, 
                        workspace, output_folder, args)      
    

def score_scenario(scenario_file, past_a_raster, past_b_raster, restoration_raster,
                   species_pool_files, redList_pool_files, endemics_pool_files,
                   kba_files, ecoregion_files, forest_intactness_files, plantation_files,
                   etp_df, ptsr_df,
                   workspace, output_folder, args):
    """
    Scores the land cover scenario `scenario_file`. Rasters `past_a_raster` and
    `past_b_raster` represent earlier land cover states and are used to calculate
    the age of the current habitat. `species_pool_files` should be a dict with taxa
    names as keys and corresponding potential richness rasters as values.
    `esa_to_predicts` and `predicts_to_SR` are DataFrames.
    :param scenario_file:
    :param past_a_raster:
    :param past_b_raster:
    :restoration_raster: 
    :param species_pool_files:
    :param esa_to_predicts:
    :param predicts_to_SR:
    :param workspace:
    :param output_folder:
    :return:
    """

    if not os.path.isdir(workspace):
        os.makedirs(workspace)
    skey = os.path.splitext(os.path.basename(scenario_file))[0]

    nodata = -9999.0
    scen_nodata = pygeo.get_raster_info(scenario_file)['nodata'][0]
    past_a_nodata = pygeo.get_raster_info(past_a_raster)['nodata'][0]
    past_b_nodata = pygeo.get_raster_info(past_b_raster)['nodata'][0]

    # convert dicts to np arrays for fast lookup
    max_esa = max(etp_df['Code'])
    etp = np.zeros(max_esa+1, dtype=np.int32)
    for e, p in zip(etp_df['Code'], etp_df['pcode']):
        etp[e] = p

    max_predicts = max(ptsr_df['Value'])
    ptsr = np.zeros(max_predicts+1, dtype=np.float32)
    for p, sr in zip(ptsr_df['Value'], ptsr_df['Mean_sr']):
        ptsr[p] = sr

    
    
    #Adding in land degredation stuff 
    climaxClasses = [50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170]
    
    def create_degraded(scen, restored):
        
        check_degraded = np.where(scen == restored, 0, 1)
        check_scen_climax = np.where(np.isin(scen, climaxClasses) == True, 0, 1)
        check_restored_climax = np.where(np.isin(restored, climaxClasses) == True, 1, 0)
                                    
        degraded =       check_degraded * check_scen_climax * check_restored_climax         
        return degraded
    
    scen_degraded_file = os.path.join(workspace, f'{skey}_degraded.tif')
    pygeo.raster_calculator(
        [(scenario_file, 1), (restoration_raster, 1)],
        create_degraded,
        scen_degraded_file,
        GDT_Float32,
        nodata
    )	 
    
    
    
    def processPlantation(lulc, plantation_base):
        result = np.where(np.isin(lulc, [35, 39, 45, 49, 
                55,
                65, 66, 67,
                75, 76, 77,
                85, 86, 87,
                95, 99,
                105, 109,
                115, 119,
                165, 
                175]) == True, 1, 0)
        
        result = result * plantation_base
        return result
                
    clipped_plantation = os.path.join(workspace, f'{skey}_plantation.tif')
    pygeo.raster_calculator(
        [(scenario_file, 1), (plantation_files, 1)],
        processPlantation,
        clipped_plantation,
        GDT_Float32,
        nodata
    )
    
    # create LU mean SR raster
    def esa_to_sr_op(scen, past_a, past_b, degraded, plantation):
        """Calculates species richness mean based on current and past LU."""
        result = np.empty(scen.shape, dtype=np.float32)
        luage = np.zeros(scen.shape, dtype=np.int32)
        result[:] = nodata
        valid_mask = (
            (scen != scen_nodata) &
            (scen > 0) & (scen < 210) &
            (past_a != past_a_nodata) &
            (past_b != past_b_nodata))



        valid_lulc_codes = [10, 11, 12, 15, 16, 20, 25, 26, 29,
                                        30, 34, 35, 39,
                                        40, 44, 45, 49,
                                        55,
                                        65, 66, 67,
                                        75, 76, 77,
                                        85, 86, 87,
                                        95,
                                        104, 105, 109,
                                        114, 115, 119,
                                        124, 125, 126,
                                        134,
                                        144,
                                        154, 155, 156, 157,
                                        165, 
                                        175, 
                                        184,
                                        190,
                                        204, 205, 206]
                                        
                                        
        luage +=  np.where(np.isin(past_a, valid_lulc_codes) == True, 0, 1)
        luage +=  np.where(np.isin(past_b, valid_lulc_codes) == True, 0, 1)
        

            
                              
        scen = scen.astype('int') #I had to add this 
        result[valid_mask] = ptsr[etp[scen[valid_mask]] + luage[valid_mask]]
        
        
        result0 = np.where(degraded == 1, 79.8, result)  #value for degraded ysv 
        result = np.minimum(result0, result)
        
        #Plantation 
        result = np.where(plantation == 1, 62.9, result)
        
        return result

    sr_file = os.path.join(workspace, f'{skey}_lu_srfactor.tif')

    pygeo.raster_calculator(
        [(scenario_file, 1), (past_a_raster, 1), (past_b_raster, 1), (scen_degraded_file, 1), (clipped_plantation,1)],
        esa_to_sr_op,
        sr_file,
        GDT_Float32,
        nodata
    )

    # create realized SR raster
    gr = [globalRichness[t] for t in taxa_types]

    def sr_op(sr, *taxa):
        """Calculates summed weighted richness by taxa."""
        result = np.empty(sr.shape, dtype=np.float32)
        result[:] = nodata
        valid_mask = (sr != nodata)
        result[valid_mask] = 0
        for t, g in zip(taxa, gr):
            result[valid_mask] += sr[valid_mask] * t[valid_mask] / (100 * g)
        
        
        result[result <0] = 0
        return result

    

    scen_SRresult_file = os.path.join(workspace, f'{skey}_biodiversitySR.tif')
    pygeo.raster_calculator(
        [(sr_file, 1)] + [(species_pool_files[t], 1) for t in taxa_types],
        sr_op,
        scen_SRresult_file,
        GDT_Float32,
        nodata
    )

    

    def binaryLULC(lulc, degraded, plantation):
        result = np.where(np.isin(lulc, [10, 11, 12, 15, 16, 20, 25, 26, 29,
                                        30, 34, 35, 39,
                                        40, 44, 45, 49,
                                        55,
                                        65, 66, 67,
                                        75, 76, 77,
                                        85, 86, 87,
                                        95,
                                        104, 105, 109,
                                        114, 115, 119,
                                        124, 125, 126,
                                        134,
                                        144,
                                        154, 155, 156, 157,
                                        165, 
                                        175, 
                                        184,
                                        190,
                                        204, 205, 206]) == True, 0, 1)
                                        

                                
                                 
                                
        result = np.where(np.isin(lulc, [34,39,44,49,                                               
                                        104,109,
                                        114,119,
                                        124, 125, 126,
                                        134,
                                        144,
                                        154, 155, 156, 157,
                                        184,
                                        204, 205, 206]) == True, 0.5, result)
                                        
                                        
        # inValid_mask = (lulc == past_a_nodata)
        inValid_mask = (lulc == scen_nodata)
        
        

        result = np.where(degraded == 1, 0, result)
        
        
        
        result = np.where(np.isin(lulc, [35, 45, 
                        55,
                        65, 66, 67,
                        75, 76, 77,
                        85, 86, 87,
                        95,
                        105,
                        115, 
                        165, 
                        175]) == True, 0.5, result)
                        
        result[plantation == 1] = 0
        
        result[inValid_mask] = nodata
        return result
    
    scen_resultBinary_file = os.path.join(workspace, f'{skey}_binaryLULC.tif')
    pygeo.raster_calculator(
        [(scenario_file, 1), (scen_degraded_file, 1), (clipped_plantation, 1)],
        binaryLULC,
        scen_resultBinary_file,
        GDT_Float32,
        nodata
    )	
    

    

    def redList(binary, *taxa):
        result = np.empty(binary.shape, dtype=np.float32)
        result[:] = nodata
        valid_mask = (binary != nodata)
        
        result[valid_mask] = 0
        for t, g in zip(taxa, gr):
            t[t < 0] = 0
            result[valid_mask] += (binary[valid_mask] * t[valid_mask]) 
        result[result <0] = 0
        return result	
    
    scen_RLresult_file = os.path.join(workspace, f'{skey}_biodiversityRedList.tif')
    pygeo.raster_calculator(
        [(scen_resultBinary_file, 1)] + [(redList_pool_files[t], 1) for t in taxa_types],
        redList,
        scen_RLresult_file,
        GDT_Float32,
        nodata
    )		
    
    def kbaCalc(binary, kbas):
        result = np.empty(binary.shape, dtype=np.float32)
        result[:] = nodata
        valid_mask = (binary != nodata)
        result[valid_mask] = 0
        
        kbas[kbas <0] = 0
        
        result[valid_mask] =  binary[valid_mask] * kbas[valid_mask]
        result[result <0] = 0
        return result
        
    scen_resultKBA_file = os.path.join(workspace, f'{skey}_kbaScore.tif')
    pygeo.raster_calculator(
        [(scen_resultBinary_file, 1), (kba_files,1)],
        kbaCalc,
        scen_resultKBA_file,
        GDT_Float32,
        nodata
        )
 
 
    def endemics(binary, *taxa):
        result = np.empty(binary.shape, dtype=np.float32)
        result[:] = nodata
        valid_mask = (binary != nodata)
        result[valid_mask] = 0
        for t, g in zip(taxa, gr):
            t[t<0] = 0
            result[valid_mask] += binary[valid_mask] * t[valid_mask] / (100 * g)
        result[result < 0] = 0
        return result


    scen_Endemicresult_file = os.path.join(workspace, f'{skey}_biodiversityEndemic.tif')
    pygeo.raster_calculator(
        [(scen_resultBinary_file, 1)] + [(endemics_pool_files[t], 1) for t in taxa_types],
        endemics,
        scen_Endemicresult_file,
        GDT_Float32,
        nodata
    )	
    
    er_nodata = pygeo.get_raster_info(ecoregion_files)['nodata'][0]
    def ecoCalc(binary, ecoregion):
        result = np.empty(binary.shape, dtype=np.float32)
        result[:] = nodata
        valid_mask = np.all([binary != nodata, ecoregion != er_nodata], axis=0)
        result[valid_mask] = 0
        
        ecoregion[ecoregion <= 0] = 1
        
        result[valid_mask] = binary[valid_mask] * (1 / ecoregion[valid_mask])
        # result[result < 0] = 0
        return result
        
    scen_resultEco_file = os.path.join(workspace, f'{skey}_ecoScore.tif')
    pygeo.raster_calculator(
        [(scen_resultBinary_file, 1), (ecoregion_files,1)],
        ecoCalc,
        scen_resultEco_file,
        GDT_Float32,
        nodata
    )
    
    def forest_intactness_calc(binary, forest_intactness):
        result = np.empty(binary.shape, dtype=np.float32)
        result[:] = nodata
        valid_mask = np.all([binary != nodata], axis=0)
        #Don't let foerstry pixels count for intactness at all
        binary[binary == 0.5] = 0
        result[valid_mask] = forest_intactness[valid_mask]  * binary[valid_mask] / 10_000
        return result
    
    scen_result_ForInt_file = os.path.join(workspace, f'{skey}_ForestIntactnessScore.tif')
    pygeo.raster_calculator(
        [(scen_resultBinary_file, 1), (forest_intactness_files, 1)],
        forest_intactness_calc,
        scen_result_ForInt_file,
        GDT_Float32,
        nodata
    )
    
    # def weightTargets(fName):
    # 	aggregate_vector_path = os.path.join(country_dir, 'Projected', f'national_boundary.shp')
    # 	result = pygeo.zonal_statistics((fName,1), aggregate_vector_path)	
    # 	return result[0]['min'], result[0]['max']

    def weightTargets(filename):
        ds = gdal.OpenEx(filename)
        band = ds.GetRasterBand(1)
        band.ComputeStatistics(0)
        minval = band.GetMinimum()
        maxval = band.GetMaximum()
        band = None
        ds = None
        return minval, maxval
    
    if args['preProcessing'] == True:
        maxValues = args['maxValues']
        maxValues['Richness']	=	weightTargets(scen_SRresult_file)[1]
        maxValues['RedList']		=	weightTargets(scen_RLresult_file)[1]
        maxValues['Endemics']	=	weightTargets(scen_Endemicresult_file)[1]
        
        return maxValues
        
        
    
    def weightResults(binary, Richness, RedList, KBA, Endemic, Ecoregion, ForestIntactness):
        """
        This function combines the scores for each submetric into the final NCI biodiversity score
        by assigning the maximum value across submetrics to each pixel. All of the submetrics except
        for ForestIntactness are normalized to be 0-1 across the within-country range. 
        """
        

        result = np.empty(Richness.shape, dtype=np.float32)
        result[:] = nodata
        result[Richness > 0] = 0
        Richness[Richness < 0] = 0
        RedList[RedList < 0] = 0
        KBA[KBA < 0] = 0
        Endemic[Endemic < 0] = 0
        Ecoregion[Ecoregion < 0] = 0
          
        Richness = np.nan_to_num((Richness - minValues['Richness']) / (maxValues['Richness'] - minValues['Richness']), nan=nodata)
        RedList = np.nan_to_num((RedList - minValues['RedList']) / (maxValues['RedList'] - minValues['RedList']), nan=nodata)
        Endemic = np.nan_to_num((Endemic - minValues['Endemics']) / (maxValues['Endemics'] - minValues['Endemics']), nan=nodata)
        KBA = np.nan_to_num(0.5 * KBA, nan=nodata)
        Ecoregion = np.nan_to_num((Ecoregion - minValues['Ecoregion']) / (maxValues['Ecoregion'] - minValues['Ecoregion']), nan=nodata)

        mats = [Richness, RedList, KBA, Endemic, Ecoregion, ForestIntactness]

        result = np.maximum.reduce(mats)
        
        inValid_mask = (binary < 0)
        result[inValid_mask] = nodata
        return result
    
    
    minValues = args['minValues']
    maxValues = args['maxValues']  
    
    scen_result_file = os.path.join(output_folder, f'{skey}_biodiversity.tif')
    pygeo.raster_calculator(
        [(scen_resultBinary_file, 1), (scen_SRresult_file, 1), (scen_RLresult_file, 1),
         (scen_resultKBA_file, 1), (scen_Endemicresult_file, 1), (scen_resultEco_file, 1),
         (scen_result_ForInt_file, 1)],
        weightResults,
        scen_result_file,
        GDT_Float32,
        nodata
    )
    
    def getOutcome(binary, Richness, RedList, KBA, Endemic, Ecoregion, ForestIntactness):
        """
        This function makes a map where each pixel gets a value depending on which layer it is coming from
        1: species richness
        2: IUCN listing
        3: Endemic
        4: KBAs
        5: Ecoregions
        6: Intactness
        """
        result = np.empty(Richness.shape, dtype=np.float32)
        result[:] = nodata
        result[Richness > 0] = 0
        Richness[Richness < 0] = 0
        RedList[RedList < 0] = 0
        KBA[KBA < 0] = 0
        Endemic[Endemic < 0] = 0
        Ecoregion[Ecoregion < 0] = 0
          
        Richness = np.nan_to_num((Richness - minValues['Richness']) / (maxValues['Richness'] - minValues['Richness']), nan=nodata)
        RedList = np.nan_to_num((RedList - minValues['RedList']) / (maxValues['RedList'] - minValues['RedList']), nan=nodata)
        Endemic = np.nan_to_num((Endemic - minValues['Endemics']) / (maxValues['Endemics'] - minValues['Endemics']), nan=nodata)
        KBA = np.nan_to_num(0.5 * KBA, nan=nodata)
        Ecoregion = np.nan_to_num((Ecoregion - minValues['Ecoregion']) / (maxValues['Ecoregion'] - minValues['Ecoregion']), nan=nodata)

        mats = [Richness, RedList,  Endemic, KBA, Ecoregion, ForestIntactness]

        result = np.maximum.reduce(mats)
        
        inValid_mask = (binary < 0)
        result[inValid_mask] = nodata
        
        
        out_rast = np.empty(Richness.shape, dtype=np.float32)
        j = 1
        for i in mats:
            out_rast = np.where(result == i, j, out_rast) 
            j += 1 
        out_rast[inValid_mask] = nodata    
        return out_rast 

    which_bio_file = os.path.join(workspace, f'{skey}_whichMetric.tif')
    pygeo.raster_calculator(
        [(scen_resultBinary_file, 1), (scen_SRresult_file, 1), (scen_RLresult_file, 1),
         (scen_resultKBA_file, 1), (scen_Endemicresult_file, 1), (scen_resultEco_file, 1),
         (scen_result_ForInt_file, 1)],
        getOutcome,
        which_bio_file,
        GDT_Float32,
        nodata
    )
    """End Jeff's additions"""


if __name__ == '__main__':
    




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

    print(maxValues)
    args = {

        'lu_raster':'C:\\Users\\jeffr\\Desktop\\Haiti\\ScenarioMaps\\restoration.tif',
        'raster_input_folder': 'C:\\Users\\jeffr\\Desktop\\Haiti\\InputRasters',
        'predicts_table_1_path': 'C:\\Users\\jeffr\\Desktop\\predicts_May2021.csv',
        'predicts_table_2_path': 'C:\\Users\\jeffr\\Desktop\\predicts2_May2021.csv',
        'target_folder': 'C:\\Users\\jeffr\\Desktop\\Haiti\\modelTest',
        'preProcessing': True,
        'minValues': minValues,
        'maxValues': maxValues
        
        
    }
    
    maxValues = execute(args)

    print(maxValues)
    
    args = {

        'lu_raster':'C:\\Users\\jeffr\\Desktop\\Haiti\\ScenarioMaps\\restoration.tif',
        'raster_input_folder': 'C:\\Users\\jeffr\\Desktop\\Haiti\\InputRasters',
        'predicts_table_1_path': 'C:\\Users\\jeffr\\Desktop\\predicts_May2021.csv',
        'predicts_table_2_path': 'C:\\Users\\jeffr\\Desktop\\predicts2_May2021.csv',
        'target_folder': 'C:\\Users\\jeffr\\Desktop\\Haiti\\modelTest',
        'preProcessing': False,
        'minValues': minValues,
        'maxValues': maxValues
        
        
    }
    execute(args)

    print(maxValues)
    
    args = {

        'lu_raster':'C:\\Users\\jeffr\\Desktop\\Haiti\\ScenarioMaps\\sustainable_current.tif',
        'raster_input_folder': 'C:\\Users\\jeffr\\Desktop\\Haiti\\InputRasters',
        'predicts_table_1_path': 'C:\\Users\\jeffr\\Desktop\\predicts_May2021.csv',
        'predicts_table_2_path': 'C:\\Users\\jeffr\\Desktop\\predicts2_May2021.csv',
        'target_folder': 'C:\\Users\\jeffr\\Desktop\\Haiti\\modelTest',
        'preProcessing': False,
        'minValues': minValues,
        'maxValues': maxValues
        
        
    }
    execute(args)

    print(maxValues)
    
    args = {

        'lu_raster':'C:\\Users\\jeffr\\Desktop\\Haiti\\ScenarioMaps\\grazing_expansion.tif',
        'raster_input_folder': 'C:\\Users\\jeffr\\Desktop\\Haiti\\InputRasters',
        'predicts_table_1_path': 'C:\\Users\\jeffr\\Desktop\\predicts_May2021.csv',
        'predicts_table_2_path': 'C:\\Users\\jeffr\\Desktop\\predicts2_May2021.csv',
        'target_folder': 'C:\\Users\\jeffr\\Desktop\\Haiti\\modelTest',
        'preProcessing': False,
        'minValues': minValues,
        'maxValues': maxValues
        
        
    }
    execute(args)

    print(maxValues)
