"""
Genetic Algorithm Optimizer - ewolucyjny dobór hyperparametrów.

Funkcjonalności:
- Genetic algorithm dla hyperparameter tuning
- Mutation, crossover, selection
- Adaptive mutation rate
- Elite preservation
"""

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import cross_val_score

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class Individual:
    """
    Osobnik w populacji (zestaw hyperparametrów).
    """

    def __init__(self, params: Dict, fitness: Optional[float] = None):
        """
        Args:
            params: Słownik z parametrami
            fitness: Wartość fitness (wyższa = lepsza)
        """
        self.params = params
        self.fitness = fitness

    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f if self.fitness else 'None'}, params={self.params})"


class GeneticOptimizer:
    """
    Genetic Algorithm dla hyperparameter optimization.
    
    Algorytm:
    1. Inicjalizacja losowej populacji
    2. Ewaluacja fitness (CV score)
    3. Selekcja najlepszych (tournament/roulette)
    4. Crossover (rekombinacja)
    5. Mutacja
    6. Powtarzanie kroków 2-5
    """

    def __init__(
        self,
        model_class: Any,
        param_space: Dict,
        problem_type: str,
        population_size: int = 20,
        n_generations: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        elite_size: int = 2,
        cv_folds: int = 5,
        scoring: str = 'accuracy',
        random_state: int = 42
    ):
        """
        Inicjalizacja genetic optimizer.

        Args:
            model_class: Klasa modelu
            param_space: Przestrzeń parametrów (dict z ranges/choices)
            problem_type: Typ problemu
            population_size: Rozmiar populacji
            n_generations: Liczba generacji
            mutation_rate: Prawdopodobieństwo mutacji
            crossover_rate: Prawdopodobieństwo crossover
            elite_size: Liczba elit (zachowywanych bez zmian)
            cv_folds: Liczba foldów CV
            scoring: Metryka
            random_state: Random state
        """
        self.model_class = model_class
        self.param_space = param_space
        self.problem_type = problem_type
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state

        # Best results
        self.best_individual = None
        self.population_history = []

        # Set seeds
        random.seed(random_state)
        np.random.seed(random_state)

        logger.info(f"Genetic Optimizer zainicjalizowany dla {model_class.__name__}")

    def _random_params(self) -> Dict:
        """
        Generuje losowe parametry z param_space.

        Returns:
            Dict: Losowe parametry
        """
        params = {}

        for param_name, param_def in self.param_space.items():
            if isinstance(param_def, dict):
                # Range or choices
                if 'range' in param_def:
                    # Continuous range
                    low, high = param_def['range']
                    if param_def.get('log', False):
                        # Log scale
                        params[param_name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                    else:
                        params[param_name] = np.random.uniform(low, high)

                    # Cast to int if needed
                    if param_def.get('type') == 'int':
                        params[param_name] = int(params[param_name])

                elif 'choices' in param_def:
                    # Categorical
                    params[param_name] = random.choice(param_def['choices'])

            elif isinstance(param_def, (list, tuple)):
                # Simple list of choices
                params[param_name] = random.choice(param_def)

        return params

    def _initialize_population(self) -> List[Individual]:
        """
        Inicjalizuje losową populację.

        Returns:
            List[Individual]: Populacja
        """
        population = []

        for _ in range(self.population_size):
            params = self._random_params()
            individual = Individual(params)
            population.append(individual)

        logger.info(f"Zainicjalizowano populację {self.population_size} osobników")

        return population

    def _evaluate_fitness(
        self,
        individual: Individual,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Ewaluuje fitness osobnika (CV score).

        Args:
            individual: Osobnik
            X: Features
            y: Target

        Returns:
            float: Fitness score
        """
        params = individual.params.copy()
        params['random_state'] = self.random_state

        try:
            model = self.model_class(**params)

            scores = cross_val_score(
                model, X, y,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=1
            )

            fitness = np.mean(scores)

        except Exception as e:
            logger.warning(f"Błąd ewaluacji: {e}")
            fitness = -np.inf  # Kara za niepoprawne parametry

        return fitness

    def _tournament_selection(
        self,
        population: List[Individual],
        tournament_size: int = 3
    ) -> Individual:
        """
        Tournament selection.

        Args:
            population: Populacja
            tournament_size: Rozmiar turnieju

        Returns:
            Individual: Wybrany osobnik
        """
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda ind: ind.fitness)
        return winner

    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Crossover (rekombinacja) dwóch rodziców.

        Args:
            parent1: Rodzic 1
            parent2: Rodzic 2

        Returns:
            Tuple[Individual, Individual]: Dwoje dzieci
        """
        if random.random() > self.crossover_rate:
            # Bez crossover - zwróć kopie rodziców
            return Individual(parent1.params.copy()), Individual(parent2.params.copy())

        # Uniform crossover
        child1_params = {}
        child2_params = {}

        for param_name in parent1.params.keys():
            if random.random() < 0.5:
                child1_params[param_name] = parent1.params[param_name]
                child2_params[param_name] = parent2.params[param_name]
            else:
                child1_params[param_name] = parent2.params[param_name]
                child2_params[param_name] = parent1.params[param_name]

        return Individual(child1_params), Individual(child2_params)

    def _mutate(self, individual: Individual) -> Individual:
        """
        Mutuje osobnika.

        Args:
            individual: Osobnik

        Returns:
            Individual: Zmutowany osobnik
        """
        mutated_params = individual.params.copy()

        for param_name, param_value in mutated_params.items():
            if random.random() < self.mutation_rate:
                # Mutuj ten parametr
                param_def = self.param_space[param_name]

                if isinstance(param_def, dict):
                    if 'range' in param_def:
                        # Continuous mutation
                        low, high = param_def['range']

                        # Gaussian mutation
                        current_value = param_value
                        mutation_std = (high - low) * 0.1  # 10% range as std

                        if param_def.get('log', False):
                            # Log scale
                            log_value = np.log(current_value)
                            log_value += np.random.normal(0, mutation_std / current_value)
                            new_value = np.exp(log_value)
                        else:
                            new_value = current_value + np.random.normal(0, mutation_std)

                        # Clip to range
                        new_value = np.clip(new_value, low, high)

                        # Cast to int if needed
                        if param_def.get('type') == 'int':
                            new_value = int(new_value)

                        mutated_params[param_name] = new_value

                    elif 'choices' in param_def:
                        # Categorical mutation - wybierz losową wartość
                        mutated_params[param_name] = random.choice(param_def['choices'])

                elif isinstance(param_def, (list, tuple)):
                    # Simple choices
                    mutated_params[param_name] = random.choice(param_def)

        return Individual(mutated_params)

    @handle_errors(show_in_ui=False)
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Uruchamia genetic algorithm optimization.

        Args:
            X: Features
            y: Target
            verbose: Czy wyświetlać progress

        Returns:
            Dict: Wyniki optymalizacji
        """
        logger.info(f"Rozpoczęcie Genetic Algorithm (pop={self.population_size}, gen={self.n_generations})")

        # Initialize population
        population = self._initialize_population()

        # Evolution loop
        for generation in range(self.n_generations):
            # Evaluate fitness
            for individual in population:
                if individual.fitness is None:
                    individual.fitness = self._evaluate_fitness(individual, X, y)

            # Sort by fitness
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            # Track best
            best_individual = population[0]

            if self.best_individual is None or best_individual.fitness > self.best_individual.fitness:
                self.best_individual = Individual(best_individual.params.copy(), best_individual.fitness)

            # Log
            if verbose:
                avg_fitness = np.mean([ind.fitness for ind in population])
                logger.info(
                    f"Generation {generation+1}/{self.n_generations} - "
                    f"Best: {best_individual.fitness:.4f}, Avg: {avg_fitness:.4f}"
                )

            # Save history
            self.population_history.append({
                'generation': generation + 1,
                'best_fitness': best_individual.fitness,
                'avg_fitness': np.mean([ind.fitness for ind in population]),
                'best_params': best_individual.params.copy()
            })

            # Create next generation
            next_population = []

            # Elitism - keep best individuals
            next_population.extend([
                Individual(ind.params.copy(), ind.fitness)
                for ind in population[:self.elite_size]
            ])

            # Fill rest with offspring
            while len(next_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                next_population.extend([child1, child2])

            # Trim to population size
            population = next_population[:self.population_size]

        # Final evaluation
        for individual in population:
            if individual.fitness is None:
                individual.fitness = self._evaluate_fitness(individual, X, y)

        population.sort(key=lambda ind: ind.fitness, reverse=True)

        if population[0].fitness > self.best_individual.fitness:
            self.best_individual = Individual(population[0].params.copy(), population[0].fitness)

        logger.info(f"Genetic Algorithm zakończony - Best fitness: {self.best_individual.fitness:.4f}")

        # Train final model
        best_params = self.best_individual.params.copy()
        best_params['random_state'] = self.random_state

        best_model = self.model_class(**best_params)
        best_model.fit(X, y)

        return {
            'best_params': self.best_individual.params,
            'best_score': self.best_individual.fitness,
            'best_model': best_model,
            'population_history': self.population_history,
            'final_population': population
        }

    def get_best_model(self) -> Any:
        """Zwraca najlepszy model."""
        if self.best_individual is None:
            raise ValueError("Optymalizacja nie została uruchomiona")

        params = self.best_individual.params.copy()
        params['random_state'] = self.random_state

        model = self.model_class(**params)

        return model

    def plot_evolution(self):
        """Generuje wykres ewolucji fitness."""
        if not self.population_history:
            raise ValueError("Optymalizacja nie została uruchomiona")

        import matplotlib.pyplot as plt

        generations = [h['generation'] for h in self.population_history]
        best_fitness = [h['best_fitness'] for h in self.population_history]
        avg_fitness = [h['avg_fitness'] for h in self.population_history]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Genetic Algorithm Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gcf()