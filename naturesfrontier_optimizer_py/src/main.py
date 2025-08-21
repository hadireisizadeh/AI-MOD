import pandas as pd 
from pathlib import Path
import h5py
import numpy as np
from typing import Dict, List, Optional
from .data_handler import DataHandler
from .optimizer import random_sample_frontier

def do_nci(workspace: str,
           country: str,
           objectives: List[str],
           sign_obj,
           method: str,      
           T: float,
           pts: int,
           results_folder: str,
           suffix: str = "",
           objectives_to_minimize: Optional[List[int]] = None,
           **kwargs) -> None:
    """
    Main function to run Natural Capital Index optimization.
    """
    if objectives_to_minimize is None:
        objectives_to_minimize = []

    # Define file lists and value columns
    table_folder = Path(workspace) / country / "ValueTables"
    file_list = [
        "extensification_bmps_irrigated.csv",
        "extensification_bmps_rainfed.csv",
        "extensification_current_practices.csv",
        "extensification_intensified_irrigated.csv",
        "extensification_intensified_rainfed.csv",
        "fixedarea_bmps_irrigated.csv",
        "fixedarea_bmps_rainfed.csv",
        "fixedarea_intensified_irrigated.csv",
        "fixedarea_intensified_rainfed.csv",
        "forestry_expansion.csv",
        "grazing_expansion.csv",
        "restoration.csv",
        "sustainable_current.csv"
    ]

    value_columns = [
        "net_econ_value",
        "biodiversity",
        "net_ghg_co2e",
        "carbon",
        "cropland_value",
        "forestry_value",
        "grazing_value",
        "grazing_methane",
        "production_value",
        "transition_cost",
        "noxn_in_drinking_water"
    ]

    # Load data
    data = DataHandler.load_objective_tables(str(table_folder), file_list, value_columns)
    
    # Setup output directory and files
    output_folder = Path(workspace) / country / (results_folder + suffix) 
    output_folder.mkdir(parents=True, exist_ok=True)
    sol_file = output_folder / "solutions.h5"

    # Create HDF5 file
    with h5py.File(sol_file, 'w') as f:
        f.create_group('solutions')

    # Run maximization optimization
    max_results = random_sample_frontier(
    data, objectives, sign_obj, pts,
    method=method, T=T, include_endpoints=True,
    reportvars=value_columns, minimize=objectives_to_minimize  # (ignored by current impl; see #3)
)
    
    #max_results = random_sample_frontier(
    #data, objectives, sign_obj, maxpts, 
    #sense='maximize', method=method, T=T, include_endpoints=True, reportvars=value_columns, minimize=objectives_to_minimize
#)


    # Save maximization results
    save_results(output_folder / "summary_table_maximize.csv",
                max_results.scores,
                value_columns)
    
    with h5py.File(sol_file, 'r+') as f:
        g = f['solutions']
        g.create_dataset('maximization',
                        data=max_results.solutions,
                        chunks=True,
                        compression='gzip',
                        compression_opts=6)

def save_results(filepath: Path,
                scores: np.ndarray,
                columns: List[str]) -> None:
    """Save results to CSV file."""
    header = ['ID'] + columns
    ids = np.arange(1, len(scores) + 1)
    data = np.column_stack([ids, scores])
    
    np.savetxt(filepath,
               data,
               delimiter=',',
               header=','.join(header),
               comments='')