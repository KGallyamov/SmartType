import random
import string
import metrics
from data_wrappers.code import CodeSearchNetDataset


class AntColonyOptimization:
    def __init__(self, num_ants=60, alpha=1.2, beta=2.5, evaporation_rate=0.35, iterations=120):
        self.NUM_ANTS = num_ants
        self.ALPHA = alpha
        self.BETA = beta
        self.EVAPORATION_RATE = evaporation_rate
        self.ITERATIONS = iterations

        self.symbols = list(string.ascii_lowercase) + [';', ',', '.', '/']
        self.size = len(self.symbols)
        self.pheromones = {s: {t: 1.0 for t in self.symbols if t != s} for s in self.symbols}

        print("Loading dataset...")
        self.dataset = CodeSearchNetDataset(language="python", split="train")[:500]
        print("Dataset loaded.")

    def heuristic(self, prev, candidate):
        return 1.0

    def adaptive_exploration(self, iteration):
        return max(0.2, 0.8 * (1 - iteration / self.ITERATIONS))

    def choose_next_symbol(self, prev_symbol, available_symbols, exploration_rate):
        if random.random() < exploration_rate:
            return random.choice(available_symbols)

        pheromone_list = [self.pheromones[prev_symbol][s] ** self.ALPHA for s in available_symbols]
        heuristic_list = [self.heuristic(prev_symbol, s) ** self.BETA for s in available_symbols]

        probabilities = [p * h for p, h in zip(pheromone_list, heuristic_list)]
        total = sum(probabilities)

        if total == 0:
            return random.choice(available_symbols)

        probabilities = [p / total for p in probabilities]
        return random.choices(available_symbols, weights=probabilities, k=1)[0]

    def generate_layout(self, iteration):
        layout = [[], [], []]
        symbols_left = self.symbols.copy()
        exploration_rate = self.adaptive_exploration(iteration)

        prev = random.choice(symbols_left)
        symbols_left.remove(prev)
        layout[0].append(prev)

        for row in range(3):
            while len(layout[row]) < 10 and symbols_left:
                nxt = self.choose_next_symbol(prev, symbols_left, exploration_rate)
                layout[row].append(nxt)
                symbols_left.remove(nxt)
                prev = nxt
            if row < 2 and symbols_left:
                prev = random.choice(symbols_left)
                symbols_left.remove(prev)

        return layout

    def mutate_layout(self, layout):
        row = random.randint(0, 2)
        i, j = random.sample(range(len(layout[row])), 2)
        layout[row][i], layout[row][j] = layout[row][j], layout[row][i]
        return layout

    def evaporate_pheromones(self):
        for s in self.pheromones:
            for t in self.pheromones[s]:
                self.pheromones[s][t] *= (1 - self.EVAPORATION_RATE)

    def reinforce(self, layout, score, weight=1.0):
        flat = layout[0] + layout[1] + layout[2]
        reward = weight * (1.0 / (score + 1e-6))
        for i in range(len(flat) - 1):
            s, t = flat[i], flat[i + 1]
            self.pheromones[s][t] += reward

    def run(self):
        best_layout = None
        best_score = float('inf')
        prev_best = float('inf')

        for iteration in range(self.ITERATIONS):
            layouts_scores = []

            for _ in range(self.NUM_ANTS):
                layout = self.generate_layout(iteration)

                if random.random() < 0.3:
                    layout = self.mutate_layout(layout)

                score = metrics.evaluate_keyboard(layout, self.dataset)['Composite Score']
                layouts_scores.append((layout, score))

                if score < best_score:
                    best_score = score
                    best_layout = layout

            layouts_scores.sort(key=lambda x: x[1])
            elite = layouts_scores[:max(3, self.NUM_ANTS // 4)]

            self.evaporate_pheromones()

            for layout, score in elite:
                self.reinforce(layout, score)

            if best_layout:
                self.reinforce(best_layout, best_score, weight=3.0)

            if iteration % 10 == 0 and iteration > 0:
                if abs(prev_best - best_score) < 1e-4:
                    print("[!] Stagnation detected → pheromone soft reset")
                    for s in self.pheromones:
                        for t in self.pheromones[s]:
                            self.pheromones[s][t] *= 0.4

            prev_best = best_score
            print(f"Iteration {iteration+1}: Best Score: {best_score:.4f}")

        return best_layout, best_score


if __name__ == "__main__":
    aco = AntColonyOptimization(
        num_ants=10,
        alpha=1.2,
        beta=2.5,
        evaporation_rate=0.35,
        iterations=20
    )

    the_best_layout, the_best_score = aco.run()

    print("\nBest layout found by ACO:")
    print(the_best_layout)
    print(f"Score: {the_best_score:.4f}")

    # Iteration 1: Best Score: 0.2914
    # Iteration 2: Best Score: 0.2857
    # Iteration 3: Best Score: 0.2857
    # Iteration 4: Best Score: 0.2857
    # Iteration 5: Best Score: 0.2857
    # Iteration 6: Best Score: 0.2831
    # Iteration 7: Best Score: 0.2831
    # Iteration 8: Best Score: 0.2831
    # Iteration 9: Best Score: 0.2831
    # Iteration 10: Best Score: 0.2831
    # Iteration 11: Best Score: 0.2831
    # Iteration 12: Best Score: 0.2831
    # Iteration 13: Best Score: 0.2831
    # Iteration 14: Best Score: 0.2831
    # Iteration 15: Best Score: 0.2718
    # Iteration 16: Best Score: 0.2718
    # Iteration 17: Best Score: 0.2718
    # Iteration 18: Best Score: 0.2718
    # Iteration 19: Best Score: 0.2718
    # Iteration 20: Best Score: 0.2718
    #
    # Best layout found by ACO:
    # [['/', 'y', 'z', 'b', '.', 't', 'h', ',', 'p', ';'],
    # ['v', 'x', 'e', 'd', 'l', 'f', 'u', 'r', 'o', 'a'],
    # ['m', 'w', 'k', 's', 'n', 'j', 'q', 'c']]
    # Score: 0.2718