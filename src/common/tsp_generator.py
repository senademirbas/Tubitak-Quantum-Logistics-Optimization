import json
import sys
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.common.brute_force_solver import BruteForceSolver


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder to handle NumPy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class TSPGenerator:
    """Class to generate and save TSP instances with random coordinates."""

    def __init__(self, num_cities, seed=42):
        self.num_cities = num_cities
        self.seed = seed
        np.random.seed(self.seed)
        self.coordinates = []
        self.distance_matrix = []

    def generate_data(self):
        """Generates random coordinates between 0-100 and computes the distance matrix."""

        self.coordinates = np.random.randint(0, 101, size=(self.num_cities, 2))
        return self.coordinates

    def calculate_distance_matrix(self):
        """Calculates the Euclidean distance matrix from the coordinates."""

        if len(self.coordinates) == 0:
            self.generate_data()

        matrix = np.zeros((self.num_cities, self.num_cities))

        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    # euclidian distance = sqrt((x2-x1)^2 + (y2-y1)^2)
                    dist = np.linalg.norm(self.coordinates[i] - self.coordinates[j])
                    matrix[i][j] = dist
                else:
                    matrix[i][j] = 0.0

        self.distance_matrix = matrix
        return matrix

    def save_data(self):
        """Saves the coordinates and distance matrix to CSV files."""

        if len(self.distance_matrix) == 0:
            self.calculate_distance_matrix()

        print(
            f" [N={self.num_cities}] : Calculating optimal solution using Brute Force..."
        )
        solver = BruteForceSolver(self.distance_matrix)
        solution = solver.solve()

        data_payload = {
            "metadata": {
                "num_cities": self.num_cities,
                "seed": self.seed,
                "description": "TSP Instance with Ground Truth",
            },
            "input": {
                "coordinates": self.coordinates,
                "distance_matrix": self.distance_matrix,
            },
            "ground_truth": {
                "optimal_path": solution["path"],
                "min_cost": solution["cost"],
            },
        }

        output_dir = project_root / "data"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = output_dir / f"tsp_n{self.num_cities}.json"

        with open(filename, "w") as f:
            json.dump(data_payload, f, indent=4, cls=NumpyEncoder)

        print(f" [N={self.num_cities}]  : {filename}")
        print(f"   -> : {solution['cost']:.4f}")


if __name__ == "__main__":
    """Example usage of TSPGenerator."""

    scenarios = [5, 6, 7]

    print("Generating TSP instances...")
    for n in scenarios:
        """Generate TSP instance with n cities."""
        tsp_gen = TSPGenerator(num_cities=n, seed=2026)
        tsp_gen.generate_data()
        tsp_gen.calculate_distance_matrix()
        tsp_gen.save_data()

    print(f"TSP instance with {scenarios} cities generated and saved.")
