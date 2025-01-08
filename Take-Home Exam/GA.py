import math
import numpy as np
from matplotlib import pyplot as plt
import copy
import gif
from IPython.display import Image
import random
from tqdm import tqdm
import time

import concurrent.futures
from functools import partial # for parallel processing

### 
import numpy as np
import time
import random
import concurrent.futures
from functools import partial
from tqdm import tqdm

def calculate_fitness(chromosome, fitness_function):
    return fitness_function(chromosome)

class GeneticAlgorithm:
    def __init__(
        self,
        population_size,
        gene_type,
        gene_interval,
        chromosome_length,
        crossover_rate,
        mutation_rate,
        elitism_count,
        fitness_function,
        tournament_size=4,
        n_crossover_points=1
    ):
        self.population_size = population_size
        self.gene_type = gene_type
        self.gene_interval = gene_interval
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.fitness_function = fitness_function
        self.tournament_size = tournament_size
        self.n_crossover_points = n_crossover_points

        # Instead of list of lists, use a NumPy array
        if gene_type == "int":
            self.population = np.random.randint(
                gene_interval[0],
                gene_interval[1] + 1,
                size=(population_size, chromosome_length)
            )
        elif gene_type == "float":
            self.population = np.random.uniform(
                gene_interval[0],
                gene_interval[1],
                size=(population_size, chromosome_length)
            )
        else:
            raise ValueError("Invalid gene type")

        self.fitness = np.zeros(population_size, dtype=float)
        self.best_individual = None
        self.generation = 0

        self.calculate_population_fitness()
        self.best_individual = self.population[np.argmax(self.fitness)]
        self.generation = 0
        self.best_fitness_per_generation=[]

    def calculate_population_fitness(self):
        # For CPU-bound tasks, try a ProcessPoolExecutor
        # If fitness_function is vectorized, prefer that approach directly.
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # partial object: pass the user-provided fitness function
            fitness_function_partial = partial(calculate_fitness, 
                                               fitness_function=self.fitness_function)
            # Map over rows in self.population
            results = executor.map(fitness_function_partial, self.population)
        self.fitness = np.fromiter(results, dtype=float, count=self.population_size)

    def crossover(self, parent1, parent2):
        """
        n-point crossover
        parent1, parent2: 1D numpy arrays
        """
        if random.random() < self.crossover_rate:
            crossover_points = sorted(random.sample(range(1, self.chromosome_length), self.n_crossover_points))
            child1 = np.copy(parent1)
            child2 = np.copy(parent2)
            for i in range(len(crossover_points)):
                start = crossover_points[i]
                end = crossover_points[i+1] if (i+1 < len(crossover_points)) else None
                if i % 2 == 0:
                    child1[start:end] = parent2[start:end]
                    child2[start:end] = parent1[start:end]
            return child1, child2
        return np.copy(parent1), np.copy(parent2)

    def mutate(self, chromosome):
        """
        Vectorized approach: for each gene, flip with probability self.mutation_rate
        """
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                if self.gene_type == "int":
                    chromosome[i] = random.randint(self.gene_interval[0], self.gene_interval[1])
                else:  # float
                    chromosome[i] = random.uniform(self.gene_interval[0], self.gene_interval[1])
        return chromosome

    def evolve(self):
        new_population = []
        # Elitism: pick top individuals
        # Use argsort to get sorted indices by descending fitness
        sorted_indices = np.argsort(-self.fitness)  # negative for descending
        elites = self.population[sorted_indices[:self.elitism_count]]
        new_population.extend(elites)

        # Keep creating children until we have enough for the new population
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            if len(new_population) < self.population_size:
                new_population.append(child1)
            child2 = self.mutate(child2)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        # Convert new_population back to a numpy array
        self.population = np.array(new_population)
        self.calculate_population_fitness()
        self.best_individual = self.population[np.argmax(self.fitness)]
        self.best_fitness_per_generation.append(np.max(self.fitness))

        self.generation += 1

    def select_parent(self):
        """
        Tournament selection: random sample, choose best
        """
    
        # Randomly pick indices (faster to operate on fitness than re-calling the function)
        idx = np.random.randint(0, self.population_size, size=self.tournament_size)
        # The best index in the tournament
        best_t_idx = idx[np.argmax(self.fitness[idx])]
        return self.population[best_t_idx].copy()

    def run(self, max_generation=1000, max_time_sec=np.inf):
        start_time = time.time()
        for i in tqdm(range(max_generation), desc="Generations", unit="generation"):
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time_sec:
                print("Time limit reached. Stopping the process.")
                break
            if np.max(self.fitness) == 10.0:
                print("Found a perfect solution. Stopping the process.")
                break
            self.evolve()

# plan is an array of 40 floating point numbers
def sim(plan):
    """
    Simulates the motion of a multi-jointed body based on a given plan.

    Parameters:
    plan (list of float): A list of control parameters for the simulation. Each element should be between -1 and 1.

    Returns:
    tuple:
        - data (list of list of list of float): A list of positions of the body parts at each time step. Each position is a list of two lists, representing the x and y coordinates of the body parts.
        - final_position (float): The final x-coordinate of the head (body part 5).

    The simulation involves:
    - Adjusting the control parameters to be within the range [-1, 1].
    - Initializing physical properties such as mass, edge lengths, spring constants, and damping factors.
    - Iteratively updating the positions and velocities of the body parts based on forces, torques, and constraints.
    - Handling contact with the ground and applying friction.
    - Recording the positions of the body parts at each time step.
    - Returning the recorded data and the final x-coordinate of the head.
    """
    for i in range(0, len(plan)):
        if plan[i] > 1:
            plan[i] = 1.0
        elif plan[i] < -1:
            plan[i] = -1.0

    dt = 0.1 # time step
    friction = 1.0 # friction coefficient
    gravity = 0.1 # gravity constant
    mass = [30, 10, 5, 10, 5, 10] # mass of body parts
    edgel = [0.5, 0.5, 0.5, 0.5, 0.9] # edge lengths
    edgesp = [160.0, 180.0, 160.0, 180.0, 160.0]    # spring constants
    edgef = [8.0, 8.0, 8.0, 8.0, 8.0]   # damping factors
    anglessp = [20.0, 20.0, 10.0, 10.0] # angular spring constants
    anglesf = [8.0, 8.0, 4.0, 4.0]  # angular damping factors

    edge = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5)] # edges
    angles = [(4, 0), (4, 2), (0, 1), (2, 3)] # angles

    # vel and pos of the body parts, 0 is hip, 5 is head, others are joints
    v = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    p = [[0, 0, -0.25, 0.25, 0.25, 0.15], [1, 0.5, 0, 0.5, 0, 1.9]]

    spin = 0.0
    maxspin = 0.0
    lastang = 0.0

    data = []

    for j in range(20): # 20 time steps
        for k in range(10): # 10 substeps
            lamb = 0.05 + 0.1 * k 
            t0 = 0.5
            if j > 0:
                t0 = plan[2 * j - 2]
            t0 *= 1 - lamb
            t0 += plan[2 * j] * lamb

            t1 = 0.0
            if j > 0:
                t1 = plan[2 * j - 1]
            t1 *= 1 - lamb
            t1 += plan[2 * j + 1] * lamb

            contact = [False, False, False, False, False, False]
            for z in range(6):
                if p[1][z] <= 0:
                    contact[z] = True
                    spin = 0
                    p[1][z] = 0

            anglesl = [-(2.8 + t0), -(2.8 - t0), -(1 - t1) * 0.9, -(1 + t1) * 0.9] 

            disp = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            dist = [0, 0, 0, 0, 0]
            dispn = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            for z in range(5):
                disp[0][z] = p[0][edge[z][1]] - p[0][edge[z][0]]
                disp[1][z] = p[1][edge[z][1]] - p[1][edge[z][0]]
                dist[z] = (
                    math.sqrt(disp[0][z] * disp[0][z] + disp[1][z] * disp[1][z]) + 0.01
                )
                inv = 1.0 / dist[z]
                dispn[0][z] = disp[0][z] * inv
                dispn[1][z] = disp[1][z] * inv 

            dispv = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            distv = [0, 0, 0, 0, 0]
            for z in range(5):
                dispv[0][z] = v[0][edge[z][1]] - v[0][edge[z][0]]
                dispv[1][z] = v[1][edge[z][1]] - v[1][edge[z][0]]
                distv[z] = 2 * (disp[0][z] * dispv[0][z] + disp[1][z] * dispv[1][z])

            forceedge = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            for z in range(5):
                c = (edgel[z] - dist[z]) * edgesp[z] - distv[z] * edgef[z]
                forceedge[0][z] = c * dispn[0][z]
                forceedge[1][z] = c * dispn[1][z]

            edgeang = [0, 0, 0, 0, 0]
            edgeangv = [0, 0, 0, 0, 0]
            for z in range(5):
                edgeang[z] = math.atan2(disp[1][z], disp[0][z])
                edgeangv[z] = (dispv[0][z] * disp[1][z] - dispv[1][z] * disp[0][z]) / (
                    dist[z] * dist[z]
                )

            inc = edgeang[4] - lastang
            if inc < -math.pi:
                inc += 2.0 * math.pi
            elif inc > math.pi:
                inc -= 2.0 * math.pi
            spin += inc
            spinc = spin - 0.005 * (k + 10 * j)
            if spinc > maxspin:
                maxspin = spinc
                lastang = edgeang[4]

            angv = [0, 0, 0, 0]
            for z in range(4): 
                angv[z] = edgeangv[angles[z][1]] - edgeangv[angles[z][0]]

            angf = [0, 0, 0, 0]
            for z in range(4):
                ang = edgeang[angles[z][1]] - edgeang[angles[z][0]] - anglesl[z]
                if ang > math.pi:
                    ang -= 2 * math.pi
                elif ang < -math.pi:
                    ang += 2 * math.pi
                m0 = dist[angles[z][0]] / edgel[angles[z][0]]
                m1 = dist[angles[z][1]] / edgel[angles[z][1]]
                angf[z] = ang * anglessp[z] - angv[z] * anglesf[z] * min(m0, m1)

            edgetorque = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            for z in range(5):
                inv = 1.0 / (dist[z] * dist[z])
                edgetorque[0][z] = -disp[1][z] * inv
                edgetorque[1][z] = disp[0][z] * inv

            for z in range(4):
                i0 = angles[z][0]
                i1 = angles[z][1]
                forceedge[0][i0] += angf[z] * edgetorque[0][i0]
                forceedge[1][i0] += angf[z] * edgetorque[1][i0]
                forceedge[0][i1] -= angf[z] * edgetorque[0][i1]
                forceedge[1][i1] -= angf[z] * edgetorque[1][i1]

            f = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
            for z in range(5):
                i0 = edge[z][0]
                i1 = edge[z][1]
                f[0][i0] -= forceedge[0][z]
                f[1][i0] -= forceedge[1][z]
                f[0][i1] += forceedge[0][z]
                f[1][i1] += forceedge[1][z]

            for z in range(6):
                f[1][z] -= gravity * mass[z]
                invm = 1.0 / mass[z]
                v[0][z] += f[0][z] * dt * invm
                v[1][z] += f[1][z] * dt * invm

                if contact[z]:
                    fric = 0.0
                    if v[1][z] < 0.0:
                        fric = -v[1][z]
                        v[1][z] = 0.0

                    s = np.sign(v[0][z])
                    if v[0][z] * s < fric * friction:
                        v[0][z] = 0
                    else:
                        v[0][z] -= fric * friction * s
                p[0][z] += v[0][z] * dt
                p[1][z] += v[1][z] * dt

            data.append(copy.deepcopy(p))

            if contact[0] or contact[5]:
                return data, p[0][5]
    return data, p[0][5]

def sim2(plan):
    _, total_distance = sim(plan)
    return total_distance

if __name__ == "__main__":
    ga = GeneticAlgorithm(
        population_size=1000,
        gene_type="float",
        gene_interval=(-1, 1),
        chromosome_length=40,
        crossover_rate=0.9,
        n_crossover_points=3,
        mutation_rate=0.04,
        elitism_count=2,
        tournament_size=8,
        fitness_function=sim2
    )

    ga.run(9)
    print(f"Best Individual: {ga.best_individual}")
    print(f"Best Fitness: {ga.fitness_function(ga.best_individual)}")

    plt.plot(ga.best_fitness_per_generation)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness through Generations")
    plt.show()