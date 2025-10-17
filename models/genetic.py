import random
import string
from tqdm.auto import tqdm

import metrics
from data_wrappers.code import CodeSearchNetDataset


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.POPULATION_SIZE = population_size
        self.MUTATION_RATE = mutation_rate
        self.PARENTS_CHILDREN = 4

        self.symbols = list(string.ascii_lowercase) + [';', ',', '.', '/']
        self.population = self.initialize_population()

        print('Loading dataset...')
        self.dataset = CodeSearchNetDataset(language="python", split="train")[:500]
        print('Dataset loaded.')

    def initialize_population(self):
        # Initialize a random population of keyboard layouts
        population = []
        for _ in range(self.POPULATION_SIZE):
            random.shuffle(self.symbols)
            population.append([self.symbols[:10], self.symbols[10:20], self.symbols[20:]])
        return population

    def select_parents(self):
        # Select two parents from the population
        get_score = lambda x: metrics.evaluate_keyboard(x, self.dataset)['Composite Score']
        population = sorted(self.population, key=get_score, reverse=False)

        return population[:self.POPULATION_SIZE // 2]

    def crossover(self, parent1, parent2):
        def PMX(p1, p2):
            size = len(p1)
            a = random.randint(0, size - 1)
            b = random.randint(0, size - 1)
            a, b = (a, b) if a < b else (b, a)

            genes = [None] * size
            genes[a:b] = p1[a:b]

            p2_filtered = [item for item in p2 if item not in genes]
            current_pos = 0
            for i in range(size):
                if genes[i] is None:
                    genes[i] = p2_filtered[current_pos]
                    current_pos += 1
            return genes

        # Perform crossover between two parents to produce an offspring
        genes1 = parent1[0] + parent1[1] + parent1[2]
        genes2 = parent2[0] + parent2[1] + parent2[2]
        children = []
        for _ in range(self.PARENTS_CHILDREN):
            child_genes = PMX(genes1, genes2)
            children.append([child_genes[:10], child_genes[10:20], child_genes[20:]])

        return children

    def mutate(self, individual):
        # Mutate an individual based on mutation rate
        if individual and random.random() < self.MUTATION_RATE:
            section = random.randint(0, 2)
            idx1, idx2 = random.sample(range(len(individual[section])), 2)
            individual[section][idx1], individual[section][idx2] = individual[section][idx2], individual[section][idx1]
        return individual

    def evolve(self):
        # Evolve the population to the next generation
        new_population = []
        while len(new_population) < self.POPULATION_SIZE:
            parents = self.select_parents()
            random.shuffle(parents)
            for i in range(0, len(parents) - 1, 2):
                children = self.crossover(parents[i], parents[i + 1])
                for child in children:
                    offspring = self.mutate(child)
                    new_population.append(offspring)
        self.population = new_population


if __name__ == '__main__':
    ga = GeneticAlgorithm(population_size=20, mutation_rate=0.2)
    generations = 100

    for generation in tqdm(range(generations), desc="Generations"):
        ga.evolve()
        best_individual = max(ga.population, key=lambda x: metrics.evaluate_keyboard(x, ga.dataset)['Composite Score'])
        best_score = metrics.evaluate_keyboard(best_individual, ga.dataset)
        print(f"Generation {generation + 1}: Best Score: {best_score['Composite Score']:.4f}, Layout: {best_individual}")

# Generation 65: Best Score: 0.2407,
# Layout: [['j', '.', 'v', ',', 'p', 'w', ';', 'f', 'z', 'q'],
#          ['k', 'h', 'e', 's', 'i', 't', 'n', 'a', 'o', 'd'],
#          ['/', 'l', 'g', 'c', 'r', 'x', 'm', 'u', 'y', 'b']]