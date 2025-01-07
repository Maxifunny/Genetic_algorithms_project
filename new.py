import random
import numpy as np
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(self, N=10, M=100, G=250, E=5, pc=0.8, pm=0.05, x0=0, seed=None):

        """
        Inicjalizacja parametrów algorytmu genetycznego.
        N - liczba kroków (długość wektora sterowania)
        M - liczebność populacji
        G - liczba generacji
        E - liczba elitarnych osobników
        pc - prawdopodobieństwo krzyżowania
        pm - prawdopodobieństwo mutacji
        x0 - początkowa wartość stanu
        seed - ziarno dla generatora losowego
        """
        self.N = N
        self.M = M
        self.G = G
        self.E = E
        self.pc = pc
        self.pm = pm
        self.x0 = x0
        self.lower_bound = -200
        self.upper_bound = 200
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def initialize_population(self):
        return [np.random.uniform(self.lower_bound, self.upper_bound, self.N).tolist() for _ in range(self.M)]

    def fitness_function(self, individual):
        x = self.x0
        cost = 0
        for u in individual:
            cost += x ** 2 + u ** 2
            x = x + u
        cost += x ** 2
        return cost

    def calculate_fitness(self, population, C_max=None):
        costs = [self.fitness_function(individual) for individual in population]

        if C_max is None:
            C_max = max(costs) * 1.5

        fitness = [C_max - cost for cost in costs]
        return fitness, C_max

    def select_parents(self, population, fitness):
        tournament_size = 3
        parents = random.sample(list(zip(population, fitness)), tournament_size)
        parents.sort(key=lambda x: x[1], reverse=True)
        return parents[0][0], parents[1][0]

    def crossover(self, parent1, parent2):
        point = random.randint(1, self.N - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.pm:
                individual[i] += random.uniform(-10, 10)
                individual[i] = max(self.lower_bound, min(self.upper_bound, individual[i]))
        return individual

    def run(self):
        population = self.initialize_population()
        fitness, C_max = self.calculate_fitness(population)

        for generation in range(self.G):
            fitness, _ = self.calculate_fitness(population, C_max)

            best_fitness = max(fitness)
            avg_fitness = sum(fitness) / len(fitness)
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            # Sortowanie populacji według przystosowania
            sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)]

            # Wybranie elitarnych osobników
            elite_population = sorted_population[:self.E]

            # Tworzenie nowej populacji na podstawie pozostałych osobników
            remaining_population = sorted_population[self.E:]
            new_population = []

            while len(new_population) < (self.M - self.E):
                parent1, parent2 = self.select_parents(remaining_population, fitness)
                if random.random() < self.pc:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                new_population.append(self.mutate(child1))
                if len(new_population) < (self.M - self.E):
                    new_population.append(self.mutate(child2))

            # Połączenie elity i nowo wygenerowanej populacji
            population = elite_population + new_population

        fitness, _ = self.calculate_fitness(population, C_max)
        best_individual = population[np.argmax(fitness)]
        best_cost = self.fitness_function(best_individual)  # Obliczenie kosztu dla najlepszego rozwiązania
        print("\nNajlepsze znalezione rozwiązanie (N = {}):".format(self.N), best_individual)
        print("Wartość funkcji przystosowania:", max(fitness))
        #print("Wartość funkcji kosztu (minimalizowana):", best_cost)  # Dodano wypisywanie kosztu

        self.plot_results()

        return best_individual

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, label="Najlepsze przystosowanie", color='green')
        plt.plot(self.avg_fitness_history, label="Średnie przystosowanie", color='blue')
        plt.xlabel("Generacja")
        plt.ylabel("Wartość funkcji przystosowania")
        plt.title(f"Funkcja przystosowania algorytmu genetycznego (N={self.N})")
        plt.legend()
        plt.grid()
        plt.savefig(f"results{self.N}.png")
        plt.show()


if __name__ == "__main__":
    vector_lengths = [10, 15, 25]
    for vec_len in vector_lengths:
        print(f"\nRunning Genetic Algorithm with vector length: {vec_len}")
        ga = GeneticAlgorithm(N=vec_len, M=100, G=100, E=5, pc=0.5, pm=0.1, x0=0, seed=42)
        ga.run()
