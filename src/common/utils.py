import json
import numpy as np
from pathlib import Path


def load_tsp_data(num_cities):
    """
    Load TSP data from a JSON file based on the number of cities.

    Args:
        num_cities (int): The number of cities for which to load the TSP data.

    Returns:
        distance_matrix (np.ndarray): The distance matrix between cities.
        coordinates (np.ndarray): The coordinates of the cities.
        best_cost (float): The known optimal cost for the TSP instance.

    """

    root_dir = Path(__file__).resolve().parents[2]

    file_path = root_dir / "data" / f"tsp_n{num_cities}.json"

    if not file_path.exists():
        raise FileNotFoundError(
            f"TSP data file for {num_cities} cities not found at {file_path}"
        )

    with open(file_path, "r") as f:
        tsp_data = json.load(f)

    distance_matrix = np.array(tsp_data["input"]["distance_matrix"])
    coordinates = np.array(tsp_data["input"]["coordinates"])
    best_cost = tsp_data["ground_truth"]["min_cost"]

    print(f"Loaded TSP data for {num_cities} cities from {file_path}")
    return distance_matrix, coordinates, best_cost
