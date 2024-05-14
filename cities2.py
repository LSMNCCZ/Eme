import random
import numpy as np
import matplotlib.pyplot as plt

# Define the distance function
def calculate_distance(city1, city2):
    return np.sqrt((city1.coordinate.x - city2.coordinate.x)**2 + (city1.coordinate.y - city2.coordinate.y)**2)

# Initialize cities
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"

class City:
    def __init__(self, name, coordinate, representation_number):
        self.name = name
        self.coordinate = Point(*coordinate)
        self.representation_number = representation_number

    def __repr__(self):
        return f"City(name={self.name}, coordinate={self.coordinate}, representation_number={self.representation_number})"

cities = [
    City("Muntinlupa", (1, 1), 1),
    City("San Juan", (5, 8), 2),
    City("Paranaque", (2, 3), 3),
    City("Makati", (4, 6), 4),
    City("Taguig", (5, 3), 5),
    City("Las Pinas", (1, 2), 6),
    City("Pasig", (6, 7), 7),
    City("Manila", (4, 4), 8),
    City("Pasay", (3, 3), 9),
    City("Valenzuela", (7, 9), 10),
    City("Mandaluyong", (5, 7), 11),
    City("Marikina", (7, 8), 12),
    City("Malabon", (8, 9), 13),
    City("Caloocan", (8, 8), 14),
    City("Quezon", (7, 7), 15)
]

# Create a distance matrix
num_cities = len(cities)
distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            distance_matrix[i][j] = calculate_distance(cities[i], cities[j])

# Genetic Algorithm parameters
population_size = 100
generations = 30

# Initialize population
def initialize_population():
    population = []
    for _ in range(population_size):
        individual = list(range(1, num_cities + 1))
        random.shuffle(individual)
        population.append(individual)
    return population

# Evaluate fitness
def evaluate_fitness(individual):
    total_distance = 0
    for i in range(len(individual) - 1):
        from_city = individual[i] - 1
        to_city = individual[i + 1] - 1
        total_distance += distance_matrix[from_city][to_city]
    # Add distance to return to the start point
    total_distance += distance_matrix[individual[-1] - 1][individual[0] - 1]
    return total_distance

# Select parents
def select_parents(population, fitness):
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    parents = random.choices(population, weights=probabilities, k=2)
    return parents

# PMX crossover
def pmx_crossover(parent1, parent2, crossover_rate):
    if random.random() > crossover_rate:
        return parent1[:], parent2[:]

    size = len(parent1)
    p1, p2 = [0]*size, [0]*size

    for k in range(size):
        p1[parent1[k]-1] = k
        p2[parent2[k]-1] = k

    cx_point1 = random.randint(0, size - 1)
    cx_point2 = random.randint(0, size - 1)
    if cx_point2 >= cx_point1:
        cx_point2 += 1
    else:
        cx_point1, cx_point2 = cx_point2, cx_point1

    for k in range(cx_point1, cx_point2):
        temp1 = parent1[k]
        temp2 = parent2[k]
        parent1[k], parent1[p1[temp2-1]] = temp2, temp1
        parent2[k], parent2[p2[temp1-1]] = temp1, temp2
        p1[temp1-1], p1[temp2-1] = p1[temp2-1], p1[temp1-1]
        p2[temp1-1], p2[temp2-1] = p2[temp2-1], p2[temp1-1]

    return parent1, parent2

# Mutation
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# Genetic Algorithm main loop
def genetic_algorithm(crossover_rate, mutation_rate):
    population = initialize_population()
    best_individual = min(population, key=evaluate_fitness)
    best_fitness = evaluate_fitness(best_individual)
    
    for generation in range(generations):
        fitness = [1 / evaluate_fitness(ind) for ind in population]
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness)
            offspring1, offspring2 = pmx_crossover(parent1[:], parent2[:], crossover_rate)
            new_population.append(mutate(offspring1, mutation_rate))
            new_population.append(mutate(offspring2, mutation_rate))
        
        population = new_population
        current_best = min(population, key=evaluate_fitness)
        current_fitness = evaluate_fitness(current_best)
        
        if current_fitness < best_fitness:
            best_individual, best_fitness = current_best, current_fitness
        
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Route = {best_individual}, Total Distance = {best_fitness}")
    
    return best_individual, best_fitness

# Experiment with specific rates
crossover_rate = 6 / 14
mutation_rate = 6 / 14

best_individual, best_fitness = genetic_algorithm(crossover_rate, mutation_rate)

# Print the best route and its distance
print(f"\nBest route after {generations} generations: {best_individual}")
print(f"Best route distance: {best_fitness}")

# Plot the best route
x_coords = [cities[city-1].coordinate.x for city in best_individual]
y_coords = [cities[city-1].coordinate.y for city in best_individual]

plt.figure(figsize=(10, 8))
plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], marker='o')

# Plot the cities
for i, city in enumerate(best_individual):
    if i == 0:
        plt.plot(cities[city-1].coordinate.x, cities[city-1].coordinate.y, marker='o', markersize=10, color='red')
    else:
        plt.plot(cities[city-1].coordinate.x, cities[city-1].coordinate.y, marker='o', markersize=6, color='blue')
    plt.text(cities[city-1].coordinate.x, cities[city-1].coordinate.y, cities[city-1].name, fontsize=9, ha='right')

plt.title('Best TSP Route')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()

