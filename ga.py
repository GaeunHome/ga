import numpy as np
import random
from typing import List, Tuple, Optional

class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int,               # Population size
        generations: int,            # Number of generations for the algorithm
        mutation_rate: float,        # Gene mutation rate
        crossover_rate: float,       # Gene crossover rate
        tournament_size: int,        # Tournament size for selection
        elitism: bool,               # Whether to apply elitism strategy
        random_seed: Optional[int],  # Random seed for reproducibility
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _init_population(self, M: int, N: int) -> List[List[int]]:
        """
        Initialize the population and generate random individuals, ensuring that every student is assigned at least one task.
        :param M: Number of students
        :param N: Number of tasks
        :return: Initialized population
        """
        # TODO: Initialize individuals based on the number of students M and number of tasks N
        population = []
        for _ in range(self.pop_size):
            individual = self._create_valid_individual(M, N)
            population.append(individual)
        return population

    def _create_valid_individual(self, M: int, N: int) -> List[int]:
        """Create a valid individual where each student is assigned at least one task."""
        individual = [-1] * N
        tasks = list(range(N))
        random.shuffle(tasks)
        # Assign at least one task to each student
        for student in range(M):
            individual[tasks[student]] = student
        # Assign remaining tasks randomly
        for task in tasks[M:]:
            individual[task] = random.randint(0, M - 1)
        return individual

    def _fitness(self, individual: List[int], student_times: np.ndarray) -> float:
        """
        Fitness function: calculate the fitness value of an individual.
        :param individual: Individual
        :param student_times: Time required for each student to complete each task
        :return: Fitness value
        """
        # TODO: Design a fitness function to compute the fitness value of the allocation plan
        total_time = sum(student_times[student, task] for task, student in enumerate(individual))
        return 1 / total_time  # Minimize total time, so fitness is inverse

    def _selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """
        Use tournament selection to choose parents for crossover.
        :param population: Current population
        :param fitness_scores: Fitness scores for each individual
        :return: Selected parent
        """
        # TODO: Use tournament selection to choose parents based on fitness scores
        selected = random.choices(
            population=population,
            weights=fitness_scores,
            k=self.tournament_size
        )
        return max(selected, key=lambda ind: self._fitness(ind, student_times))

    def _crossover(self, parent1: List[int], parent2: List[int], M: int) -> Tuple[List[int], List[int]]:
        """
        Crossover: generate two offspring from two parents.
        :param parent1: Parent 1
        :param parent2: Parent 2
        :param M: Number of students
        :return: Generated offspring
        """
        # TODO: Complete the crossover operation to generate two offspring

        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            # Ensure validity of offspring
            child1 = self._ensure_validity(child1, M)
            child2 = self._ensure_validity(child2, M)
            return child1, child2
        return parent1, parent2

    def _mutate(self, individual: List[int], M: int) -> List[int]:
        """
        Mutation operation: randomly change some genes (task assignments) of the individual.
        :param individual: Individual
        :param M: Number of students
        :return: Mutated individual
        """
        # TODO: Implement the mutation operation to randomly modify genes
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.randint(0, M - 1)
        # Ensure validity after mutation
        return self._ensure_validity(individual, M)

    def _ensure_validity(self, individual: List[int], M: int) -> List[int]:
        """Ensure that each student is assigned at least one task."""
        # Count the number of tasks assigned to each student
        task_count = [0] * M
        for student in individual:
            task_count[student] += 1

        # Find students with no tasks assigned
        missing_students = [student for student in range(M) if task_count[student] == 0]

        # If there are missing students, reassign tasks
        if missing_students:
            for student in missing_students:
                # Find a student with more than one task
                for i, count in enumerate(task_count):
                    if count > 1:
                        # Reassign one of their tasks to the missing student
                        for j, assigned_student in enumerate(individual):
                            if assigned_student == i:
                                individual[j] = student
                                task_count[i] -= 1
                                task_count[student] += 1
                                break
                        break

        return individual

    def __call__(self, M: int, N: int, student_times: np.ndarray) -> Tuple[List[int], int]:
        """
        Execute the genetic algorithm and return the optimal solution (allocation plan) and its total time cost.
        :param M: Number of students
        :param N: Number of tasks
        :param student_times: Time required for each student to complete each task
        :return: Optimal allocation plan and total time cost
        """
        # TODO: Complete the genetic algorithm process, including initialization, selection, crossover, mutation, and elitism strategy
        population = self._init_population(M, N)
        best_individual = None
        best_fitness = float('-inf')

        for _ in range(self.generations):
            fitness_scores = [self._fitness(ind, student_times) for ind in population]
            new_population = []

            if self.elitism:
                best_idx = np.argmax(fitness_scores)
                best_individual = population[best_idx]
                best_fitness = fitness_scores[best_idx]
                new_population.append(best_individual)

            while len(new_population) < self.pop_size:
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)
                child1, child2 = self._crossover(parent1, parent2, M)
                new_population.extend([self._mutate(child1, M), self._mutate(child2, M)])

            population = new_population

        best_individual = max(population, key=lambda ind: self._fitness(ind, student_times))
        total_time = int(1 / self._fitness(best_individual, student_times))
        return best_individual, total_time

if __name__ == "__main__":
    def write_output_to_file(problem_num: int, total_time: int, filename: str = "results.txt") -> None:
        """Write results to a file and check if the format is correct."""
        print(f"Problem {problem_num}: total_time = {total_time}")

        if not isinstance(total_time, int):
            raise ValueError(f"Invalid format for problem {problem_num}. Total time should be an integer.")

        with open(filename, 'a') as file:
            file.write(f"total_time = {total_time}\n")

    # Updated problem
    # Define multiple test problems
    M1, N1 = 2, 3
    cost1 = [
        [3, 2, 4],
        [4, 3, 2]
    ]

    M2, N2 = 4, 4
    cost2 = [
        [5, 6, 7, 4],
        [4, 5, 6, 3],
        [6, 4, 5, 2],
        [3, 2, 4, 5]]
    

    M3, N3 = 8, 9
    cost3 = [
        [90, 100, 60, 5, 50, 1, 100, 80, 70],
        [100, 5, 90, 100, 50, 70, 60, 90, 100],
        [50, 1, 100, 70, 90, 60, 80, 100, 4],
        [60, 100, 1, 80, 70, 90, 100, 50, 100],
        [70, 90, 50, 100, 100, 4, 1, 60, 80],
        [100, 60, 100, 90, 80, 5, 70, 100, 50],
        [100, 4, 80, 100, 90, 70, 50, 1, 60],
        [1, 90, 100, 50, 60, 80, 100, 70, 5]
    ]

    M4, N4 = 3, 3
    cost4 = [
        [2, 5, 6],  # Head Chef Jack
        [4, 3, 5],  # Sous Chef Annie
        [5, 6, 2]   # Pastry Chef Paul
    ]

    # Q5
    M5, N5 = 4, 4
    cost5 = [
        [4, 5, 1, 6],  # Professor Wang
        [9, 1, 2, 6],  # Dr. Lee
        [6, 9, 3, 5],  # Expert Zhang
        [2, 4, 5, 2]   # Consultant Zhao
    ]

    M6, N6 = 4, 4
    cost6 = [[5, 4, 6, 7],   # Van
             [8, 3, 4, 6],   # Marie
             [6, 7, 3, 8],   # Luc
             [7, 8, 9, 2]]   # Ada
    
    M7, N7 = 4, 4
    cost7 = [[4, 7, 8, 9],   # Screenwriter A
             [6, 3, 6, 7],   # Screenwriter B
             [8, 6, 2, 6],   # Screenwriter C
             [7, 8, 7, 3]]   # Screenwriter D
    
    M8, N8 = 5, 5
    cost8 = [[8, 8, 24, 24, 24],      # Speaker Smith
            [6, 18, 18, 6, 18],       # Speaker Chen
            [30, 10, 30, 10, 30],     # Speaker Mulier
            [21, 21, 21, 7, 7],       # Speaker Singh
            [27, 27, 9, 27, 9]]       # Speaker Nakamura
    
    M9, N9 = 5, 5
    cost9 = [[10, 10, np.inf, np.inf, np.inf],  # Alice
            [12, np.inf, np.inf, 12, 12],       # Bob
            [np.inf, 15, 15, np.inf, np.inf],   # Charlie
            [11, np.inf, 11, np.inf, np.inf],   # Diana
            [np.inf, 14, np.inf, 14, 14]]       # Evelyn
    
    M10, N10 = 9, 10
    cost10 = [[1, 90, 100, 50, 70, 20, 100, 60, 80, 90],
              [100, 10, 1, 100, 60, 80, 70, 100, 50, 90],
              [90, 50, 70, 1, 100, 100, 60, 90, 80, 100],
              [70, 100, 90, 5, 10, 60, 100, 80, 90, 50],
              [50, 100, 100, 90, 20, 4, 80, 70, 60, 100],
              [100, 5, 80, 70, 90, 100, 4, 50, 1, 60],
              [90, 60, 50, 4, 100, 90, 100, 5, 10, 80],
              [100, 70, 90, 100, 4, 60, 1, 90, 100, 5],
              [80, 100, 5, 60, 50, 90, 70, 100, 4, 1]]

    M11, N11= 5, 5
    cost11 = [
        [8,8,4,1,2],
        [5,2,6,6,2],
        [7,10,2,1,5],
        [1,3,2,7,7],
        [8,11,9,8,9]]

    M12, N12 = 2, 3
    cost12 = [
        [9, 10, 11], # Tom
        [1, 2, 3]]    # Jerry

    problems = [(M1, N1, np.array(cost1)),
                (M2, N2, np.array(cost2)),
                (M3, N3, np.array(cost3)),
                (M4, N4, np.array(cost4)),
                (M5, N5, np.array(cost5)),
                (M6, N6, np.array(cost6)),
                (M7, N7, np.array(cost7)),
                (M8, N8, np.array(cost8)),
                (M9, N9, np.array(cost9)),
                (M10, N10, np.array(cost10)),
                (M11, N11, np.array(cost11)),
                (M12, N12, np.array(cost12)),
            ]

    # Set the parameters for the genetic algorithm
    ga = GeneticAlgorithm(
        pop_size=100,
        generations=500,
        mutation_rate=0.1,
        crossover_rate=0.7,
        tournament_size=5,
        elitism=True,
        random_seed=42
    )

    # Solve each problem and immediately write the results to the file
    for i, (M, N, student_times) in enumerate(problems, 1):
        best_allocation, total_time = ga(M=M, N=N, student_times=student_times)
        write_output_to_file(i, total_time)

    print("Results have been written to results.txt")