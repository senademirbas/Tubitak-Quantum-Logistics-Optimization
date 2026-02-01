import random
import sys
import time
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.common.utils import load_tsp_data


class GeneticAlgorithmTSP:
    """Class to solve TSP using a Genetic Algorithm.
    Args:
        num_cities (int): Number of cities in the TSP instance.
        pop_size (int): Size of the population.
        mutation_rate (float): Probability of mutation.
        generations (int): Number of generations to run the algorithm.

    Returns:
        None

    Methods:
        run(): Executes the genetic algorithm and prints the best solution found.

    """

    def __init__(self, num_cities, pop_size=100, mutation_rate=0.01, generations=500):
        self.num_cities = num_cities
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.distance_matrix, _, self.optimal_cost = load_tsp_data(num_cities)

    def calculate_distance(self, route):
        """Calculates the total distance of the given route.
        Args:
            route (list): A list representing the order of cities in the route.

        Returns:
            float: The total distance of the route.
        """

        distance = 0

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            distance += self.distance_matrix[u][v]

        distance += self.distance_matrix[route[-1]][route[0]]  # Return to start
        return distance

    def create_population(self):
        """Creates the initial population of random routes.

        Args:
            None
        Returns:
            list: A list of routes representing the population."""

        population = []
        base_route = list(range(self.num_cities))
        for _ in range(self.pop_size):
            route = random.sample(base_route, self.num_cities)
            population.append(route)
        return population

    def selection(self, population, distances):
        """Selects a parent route using tournament selection.
        Args:
            population (list): The current population of routes.
            distances (list): The distances corresponding to each route in the population.

        Returns:
            list: A selected parent route.
        """
        tournament_size = 5
        candidates_indices = random.sample(range(len(population)), tournament_size)
        best_idx = candidates_indices[0]
        for idx in candidates_indices:
            if distances[idx] < distances[best_idx]:
                best_idx = idx
        return population[best_idx]

    def crossover(self, parent1, parent2):
        """Performs ordered crossover between two parent routes.
        Args:
            parent1 (list): The first parent route.
            parent2 (list): The second parent route.

        Returns:
            list: The child route resulting from the crossover.
        """

        size = self.num_cities
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[start:end] = parent1[start:end]

        pointer = 0
        for city in parent2:
            if city not in child:
                while pointer < size and child[pointer] != -1:
                    pointer += 1
                if pointer < size:
                    child[pointer] = city
        return child

    def mutate(self, route):
        """Mutates a route by swapping two cities with a certain probability.
        Args:
            route (list): The route to be mutated.

        Returns:
            list: The mutated route.
        """

        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.num_cities), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def run(self):
        """Runs the genetic algorithm to solve the TSP.
        Args:
            None

        Returns:
            None
        """

        print(f"Starting Genetic Algorithm for TSP (N={self.num_cities})")

        start_time = time.time()
        population = self.create_population()
        global_best_distance = float("inf")

        for gen in range(self.generations):
            distances = [self.calculate_distance(ind) for ind in population]

            min_dist = min(distances)
            if min_dist < global_best_distance:
                global_best_distance = min_dist

            new_population = []

            bes_idx = distances.index(min_dist)
            new_population.append(population[bes_idx])

            while len(new_population) < self.pop_size:
                parent1 = self.selection(population, distances)
                parent2 = self.selection(population, distances)

                child = self.crossover(parent1, parent2)
                child = self.mutate(child)

                new_population.append(child)

            population = new_population

        end_time = time.time()

        print(f"Genetic Algorithm completed in {end_time - start_time:.4f} seconds")
        print(f"Best distance found: {global_best_distance:.4f}")
        print(f"Known optimal distance: {self.optimal_cost:.4f}")

        gap = ((global_best_distance - self.optimal_cost) / self.optimal_cost) * 100
        print(f"   -> Optimality gap: %{gap:.2f}%")


if __name__ == "__main__":
    """Example usage of the GeneticAlgorithmTSP class."""

    ga = GeneticAlgorithmTSP(num_cities=7, generations=100)
    ga.run()
