import torch
from torch_geometric.utils import dense_to_sparse
import numpy as np

def compute_distance(node_i, node_j):
    """
    Berechnet die euklidische Distanz zwischen zwei Knoten.
    """
    return torch.sqrt(torch.sum((node_i - node_j) ** 2))

def compute_bearing(node_i, node_j):
    """
    Berechnet den Bearing (Winkel) von Knoten i zu Knoten j.
    """
    delta_x = node_j[0] - node_i[0]
    delta_y = node_j[1] - node_i[1]
    angle = torch.atan2(delta_y, delta_x)
    return angle

def compute_relative_orientation(node_i, node_j, theta_i, theta_j):
    """
    Berechnet die relative Orientierung von Knoten i zu Knoten j.
    """
    delta_x = node_j[0] - node_i[0]
    delta_y = node_j[1] - node_i[1]
    relative_angle = torch.atan2(delta_y, delta_x) - theta_j
    return relative_angle

def compute_distance_matrix(positions):
    """
    Compute the pairwise distance matrix for all nodes.
    """
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # Shape: [num_nodes, num_nodes, 2]
    distances = torch.norm(diff, dim=2)  # Shape: [num_nodes, num_nodes]
    return distances

def build_edge_index_batch(positions, communication_range, observation_range, attack_range):
    """
    Optimized version that computes edge indices and features in a batched way.
    """
    num_nodes = positions.shape[0]
    
    # Compute distance matrix between all pairs of nodes
    distances = compute_distance_matrix(positions)  # Shape: [num_nodes, num_nodes]
    
    # Get the maximum range
    max_range = max(communication_range, observation_range, attack_range)
    
    # Create masks for nodes within range
    mask = distances <= max_range
    mask.fill_diagonal_(False)  # Exclude self-loops

    # Indices of valid edges (where the mask is True)
    edge_indices = mask.nonzero(as_tuple=True)
    
    # Calculate edge features (distance, bearing, relative orientation) for valid edges
    edge_features = []
    for i, j in zip(*edge_indices):
        distance = distances[i, j].item()
        bearing = compute_bearing(positions[i], positions[j])
        relative_orientation = compute_relative_orientation(positions[i], positions[j], theta_i=0, theta_j=0)  # Placeholder
        
        edge_features.append([distance, bearing.item(), relative_orientation.item()])
    
    # Convert the edge index and edge attributes to tensors
    edge_index = torch.stack(edge_indices, dim=1)  # Shape: [2, num_edges]
    edge_attr = torch.tensor(edge_features, dtype=torch.float)  # Shape: [num_edges, 3]
    
    return edge_index, edge_attr


def build_edge_index(positions, communication_range, observation_range, attack_range):
    """
    Erstellt die Edge-Index-Matrix und die Edge-Features basierend auf den Distanzen und Bereichen.
    """
    num_nodes = positions.shape[0]
    edge_indices = []
    edge_features = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            distance = compute_distance(positions[i], positions[j])
            
            # Prüfe, ob die Knoten in den Bereichen liegen
            if distance <= max(communication_range, observation_range, attack_range):
                # Berechne Edge Features
                bearing = compute_bearing(positions[i], positions[j])
                relative_orientation = compute_relative_orientation(
                    positions[i], positions[j], theta_i=0, theta_j=0  # Für jetzt: keine Orientierung
                )
                edge_indices.append([i, j])
                edge_features.append([distance.item(), bearing.item(), relative_orientation.item()])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return edge_index, edge_attr

