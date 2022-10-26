import typing as tp

import numpy as np
import tqdm

import nasge.utils as nasge_utils
from nasge.gramm_parser import InstanceBuilder, Individual


class GAEvolution:
    def __init__(self,
                 instance_builder: InstanceBuilder,
                 genoelement_range: tp.Tuple[int, int],
                 genotype_size: int,
                 population_size: int,
                 fitness_calculator: callable,
                 offspring_fraction: float,
                 crossover_probability: float,
                 individual_mutate_probability: float,
                 genoelem_mutate_probability: float,
                 epochs: int,
                 select_max: bool):
        self.instance_builder = instance_builder
        self.genoelement_range = genoelement_range
        self.genotype_size = genotype_size
        self.population_size = population_size
        self.fitness_calculator = fitness_calculator
        self.offspring_fraction = offspring_fraction
        self.crossover_probability = crossover_probability
        self.individual_mutate_probability = individual_mutate_probability
        self.genoelem_mutate_probability = genoelem_mutate_probability
        self.epochs = epochs
        self.select_max = select_max

        self.logger = nasge_utils.get_logger("GAEvolution")

    def run(self) -> tp.Tuple[float,
                              Individual,
                              tp.List[float],
                              tp.List[Individual]]:
        pop = None
        for epoch in range(self.epochs):
            self.logger.info(f"epoch #{epoch+1}")
            pop = self.create_new_population()
            fitnesses = list(map(self.fitness_calculator, tqdm.tqdm(pop)))
            offspring = self.select(pop, fitnesses,
                                    fraction=self.offspring_fraction)
            crossovered_offspring = self.crossover(pop[::2],
                                                   pop[1::2],
                                                   self.crossover_probability)
            mutated_offspring = self.mutate(pop,
                                            self.individual_mutate_probability,
                                            self.genoelem_mutate_probability)
            candidates = (offspring
                          + crossovered_offspring
                          + mutated_offspring)

            candidates_fitnesses = list(
                map(self.fitness_calculator, tqdm.tqdm(candidates)))

            pop = self.select(
                candidates,
                candidates_fitnesses,
                count=self.population_size)
        fitnesses = list(map(self.fitness_calculator, tqdm.tqdm(pop)))
        elements = [(score, elem) for score, elem in zip(fitnesses, pop)]
        elements = sorted(elements, key=lambda x: x[0])
        if self.select_max:
            elements = elements[::-1]
        best_score, best_individ = elements[0]
        return best_score, best_individ, fitnesses, pop

    def create_new_population(self) -> tp.List[Individual]:
        return [self.instance_builder.build()
                for _ in range(self.population_size)]

    def mutate(self,
               population: tp.List[Individual],
               probability: float,
               mut_probability: float):
        result = []
        for el in population:
            if np.random.random() < probability:
                result.append(self.mutation(el, mut_probability))
        return result

    def mutation(self,
                 individual: Individual,
                 mut_probability: float) -> Individual:
        geno = []
        for el in individual.genotype.values:
            if np.random.random() < mut_probability:
                new_value = np.random.choice(
                    range(*self.genoelement_range), size=1)[0]
            else:
                new_value = el
            geno.append(new_value)
        return self.instance_builder.build_with_genotype_as_seq(geno)

    def crossover(self,
                  group1: tp.List[Individual],
                  group2: tp.List[Individual],
                  crossover_probability: float) -> tp.List[Individual]:
        result = []
        for ind1, ind2 in zip(group1, group2):
            if np.random.random() < crossover_probability:
                ind1_geno = ind1.genotype.values
                ind2_geno = ind2.genotype.values
                N = int(len(ind1_geno)/2)
                child1_geno = ind1_geno[:N] + ind2_geno[N:]
                child2_geno = ind2_geno[:N] + ind1_geno[N:]

                child1_individ = (
                    self
                    .instance_builder
                    .build_with_genotype_as_seq(child1_geno))
                child2_individ = (
                    self
                    .instance_builder
                    .build_with_genotype_as_seq(child2_geno))
                result.append(child1_individ)
                result.append(child2_individ)
        return result

    def select(self,
               population: tp.List[Individual],
               fitnesses: tp.List[float],
               fraction: float = None,
               count: int = None) -> tp.List[Individual]:
        ixs = np.argsort(fitnesses)
        if fraction is not None:
            if self.select_max:
                n = ixs.shape[0] - int(ixs.shape[0]*fraction)
            else:
                n = int(ixs.shape[0]*fraction)
        elif count is not None:
            if self.select_max:
                n = ixs.shape[0] - count - 1
            else:
                n = count + 1
        indexes = ixs > n if self.select_max else ixs < n
        return [el for el, is_valid in zip(population, indexes) if is_valid]