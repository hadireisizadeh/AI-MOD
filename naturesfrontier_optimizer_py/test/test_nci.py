import pytest
import numpy as np
from pathlib import Path
from src.nci import DataHandler, random_sample_frontier, get_baseline_scores

@pytest.fixture
def test_data_path():
    return Path(__file__).parent / "testdata"

def test_data_loading(test_data_path):
    file_list = [
        "ag_to_natural.csv",
        "baseline.csv",
        "expanded_ag.csv"
    ]
    objectives = ["agriculture", "biodiversity", "carbon"]
    
    data = DataHandler.load_objective_tables(
        str(test_data_path), 
        file_list, 
        objectives
    )
    
    assert "agriculture" in data
    assert np.array_equal(data["agriculture"][0], np.array([0.0, 10.0, 10.0]))

def test_baseline_scores(test_data_path):
    file_list = [
        "ag_to_natural.csv",
        "baseline.csv",
        "expanded_ag.csv"
    ]
    objectives = ["agriculture", "biodiversity", "carbon"]
    
    data = DataHandler.load_objective_tables(
        str(test_data_path), 
        file_list, 
        objectives
    )
    
    baseline = get_baseline_scores(data, objectives, 1)
    assert np.array_equal(baseline, np.array([32, 17, 21]))