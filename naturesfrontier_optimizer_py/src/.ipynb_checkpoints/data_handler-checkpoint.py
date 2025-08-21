import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import csv

class DataHandler:
    @staticmethod
    def read_country_list_file(list_file: str) -> List[str]:
        with open(list_file, 'r') as f:
            reader = csv.reader(f)
            return [row[0] for row in reader]

    @staticmethod
    def load_objective_tables(data_folder: str,
                            scenario_files: List[str],
                            objective_columns: List[str]) -> Dict[str, np.ndarray]:
        data = DataHandler.load_data(data_folder, file_list=scenario_files)
        values = {}
        for ob in objective_columns:
            values[ob] = DataHandler.extract_matrix(data, ob)
        return values

    @staticmethod
    def load_data(data_dir: str,
                  file_list: Optional[List[str]] = None,
                  col_list: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Loads CSV files from specified directory.
        """
        data_path = Path(data_dir)
        if file_list is None:
            file_list = [f.name for f in data_path.glob("*.csv")]

        data = {}
        for f in file_list:
            name = Path(f).stem
            full_path = data_path / f
            df = pd.read_csv(full_path)
            
            if col_list is not None:
                df = df[col_list]
            
            data[name] = df

        return data

    @staticmethod
    def extract_matrix(df_dict: Dict[str, pd.DataFrame],
                      column: str,
                      key_order: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract objective values into matrix form.
        """
        if key_order is None:
            key_order = sorted(df_dict.keys())

        first_df = df_dict[key_order[0]]
        result = np.zeros((len(first_df), len(key_order)))

        for i, k in enumerate(key_order):
            result[:, i] = df_dict[k][column].values.astype(float)

        return result

    @staticmethod
    def extract_vector(df_dict: Dict[str, pd.DataFrame],
                      column: str) -> np.ndarray:
        """
        Extract a single column as vector.
        """
        k = next(iter(df_dict))
        return df_dict[k][column].values.astype(float)