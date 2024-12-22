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

# Beispielnutzung:
positions = torch.tensor([[0, 0], [1, 1], [2, 0], [3, 3]], dtype=torch.float)  # Positionen der Knoten
communication_range = 2.0
observation_range = 2.0
attack_range = 1.5

edge_index, edge_attr = build_edge_index(positions, communication_range, observation_range, attack_range)

print("Edge Index:")
print(edge_index)
print("Edge Features:")
print(edge_attr)
