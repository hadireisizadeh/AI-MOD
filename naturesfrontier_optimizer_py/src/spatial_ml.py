from typing import List
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import geopandas as gpd
import pandas as pd
import numpy as np

class SpatialOptimizer(torch.nn.Module):
    def __init__(self, 
                 n_features: int,
                 n_scenarios: int = 13,  # Default from original scenarios
                 hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(n_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, n_scenarios)
        
    def forward(self, x, edge_index, edge_weight=None):
        # First graph convolution
        h = torch.relu(self.conv1(x, edge_index, edge_weight))
        # Second graph convolution
        h = torch.relu(self.conv2(h, edge_index, edge_weight))
        # Output layer
        return self.conv3(h, edge_index, edge_weight)

def prepare_spatial_data(sdu_shapefile: str, value_table_path: str):
    """Prepare spatial graph data for the optimizer."""
    # Load SDU shapefile
    gdf = gpd.read_file(sdu_shapefile)
    # Load value table
    df = pd.read_csv(value_table_path)
    
    # Create edge index based on spatial adjacency
    edge_list = []
    edge_weights = []
    
    # Find adjacent SDUs
    for idx, sdu in gdf.iterrows():
        # Get neighbors that share boundaries
        neighbors = gdf[gdf.geometry.touches(sdu.geometry)]
        for nidx, neighbor in neighbors.iterrows():
            edge_list.append([idx, nidx])
            # Weight based on shared boundary length
            edge_weights.append(
                sdu.geometry.intersection(neighbor.geometry).length
            )
    
    # Convert to torch tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    # Create feature matrix
    features = ['net_econ_value', 'biodiversity', 'net_ghg_co2e']
    x = torch.tensor(df[features].values, dtype=torch.float)
    
    return Data(x=x, 
                edge_index=edge_index, 
                edge_weight=edge_weight)

def spatial_optimize(data_graph: Data,
                    model: SpatialOptimizer,
                    objectives: List[str],
                    weights: np.ndarray):
    """Run optimization with spatial considerations."""
    
    model.eval()
    with torch.no_grad():
        # Get scenario scores considering spatial relationships
        scenario_scores = model(data_graph.x, 
                              data_graph.edge_index,
                              data_graph.edge_weight)
        
        # Calculate weighted objective values
        weighted_scores = torch.zeros_like(scenario_scores)
        for i, obj in enumerate(objectives):
            weighted_scores[:, i] = scenario_scores[:, i] * weights[i]
        
        # Get best scenario for each SDU
        best_scenarios = torch.argmax(weighted_scores, dim=1)
        
        return best_scenarios.numpy()
    
# Initialize paths
sdu_file = "E:/research/naturesfrontier/optimizer_running/workspace/Suriname/sdu_map/sdu.shp"
value_table = "E:/research/naturesfrontier/optimizer_running/workspace/Suriname/ValueTables/sustainable_current.csv"

# Prepare data
data_graph = prepare_spatial_data(sdu_file, value_table)

# Initialize model
n_features = data_graph.x.shape[1]
model = SpatialOptimizer(n_features=n_features)

# Define objectives and weights
objectives = ['net_econ_value', 'biodiversity', 'net_ghg_co2e']
weights = np.array([0.33, 0.33, 0.33])  # Equal weights example

# Run optimization
solutions = spatial_optimize(data_graph, model, objectives, weights)