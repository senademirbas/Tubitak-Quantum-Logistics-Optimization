import itertools


class BruteForceSolver:
    """Brute Force Solver for the Traveling Salesman Problem (TSP)."""

    def __init__(self, distance_matrix):
        """Initializes the solver with the given distance matrix."""

        self.matrix = distance_matrix
        self.num_cities = len(distance_matrix)

    def solve(self):
        """Solves the TSP using brute force by evaluating all possible routes.
        Returns:
            dict: A dictionary containing the best path and its cost.
        """

        cities = list(range(self.num_cities))

        permutations = itertools.permutations(cities[1:])

        best_distance = float("inf")
        best_path = []

        for p in permutations:
            current_path = [0] + list(p) + [0]
            current_distance = 0

            for i in range(len(current_path) - 1):
                u, v = current_path[i], current_path[i + 1]
                current_distance += self.matrix[u][v]

            if current_distance < best_distance:
                best_distance = current_distance
                best_path = current_path

        return {"path": best_path, "cost": best_distance}
