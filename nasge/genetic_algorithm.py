import typing as tp
import numpy as np
from dataclasses import dataclass
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class Individual:
    genotype: tp.List[int]
    phenotype: tp.List[str]
    model: nn.Sequential
    full_model: nn.Module = None


class GAEvolution:
    """

    """
    def __init__(self, population_builder):
        self.population_builder = population_builder

    def init_population(self) -> tp.List[Individual]:
        return self.population_builder.create_population()

    def create_new_population(self) -> tp.List[Individual]:
        return self.init_population()

    def mutate(self,
                 population: tp.List[Individual],
                 probability: float,
                 mut_probability: float):
        result = []
        for el in population:
            if np.random.random() > probability:
                result.append(self.mutation(el, mut_probability))
        return result

    def mutation(self,
                 individual: Individual,
                 mut_probability: float) -> Individual:
        geno = []
        for el in individual.genotype:
            if np.random.random() > mut_probability:
                new_value = np.random.choice(
                    range(*self.population_builder.phenotype_range), size=1)[0]
            else:
                new_value = el
            geno.append(new_value)
        pheno, model = self.population_builder.ptmw.get_phenotype_and_model(geno)
        return Individual(geno, pheno, model)

    def crossover(self,
                  group1: tp.List[Individual],
                  group2: tp.List[Individual],
                  crossover_probability: float) -> tp.List[Individual]:
        result = []
        for ind1, ind2 in zip(group1, group2):
            if np.random.random() > crossover_probability:
                ind1_geno = ind1.genotype
                ind2_geno = ind2.genotype
                N = int(len(ind1_geno)/2)
                child1_geno = ind1_geno[:N] + ind2_geno[N:]
                child2_geno = ind2_geno[:N] + ind1_geno[N:]

                child1_pheno, child1_model = (
                    self
                    .population_builder
                    .ptmw
                    .get_phenotype_and_model(child1_geno))
                child2_pheno, child2_model = (
                    self
                    .population_builder
                    .ptmw
                    .get_phenotype_and_model(child2_geno))
                result.append(Individual(child1_geno, child1_pheno, child1_model))
                result.append(Individual(child2_geno, child2_pheno, child2_model))
        return result

    def select(self,
               population: tp.List[Individual],
               fitnesses: tp.List[float],
               fraction: float = None,
               count: int = None) -> tp.List[Individual]:
        ixs = np.argsort(fitnesses)
        if fraction is not None:
            N = ixs.shape[0] - int(ixs.shape[0]*fraction)
        elif count is not None:
            N = ixs.shape[0] - count - 1
        offspring = [el for el, is_valid in zip(population, ixs > N) if is_valid]
        return offspring

    def eval(self,
             fitness_calculator: callable,
             offspring_fraction: float = 0.2,
             crossover_probability: float = 0.5,
             individual_mutate_probability: float = 0.5,
             genoelem_mutate_probability: float = 0.2,
             epochs: int = 10
    ) -> tp.List[Individual]:
        pop = None
        for epoch in range(epochs):
            pop = self.create_new_population()
            fitnesses = list(map(lambda x: fitness_calculator, pop))
            offspring = self.select(pop, fitnesses, fraction=offspring_fraction)
            crossovered_offspring = self.crossover(pop[::2],
                                                   pop[1::2],
                                                   crossover_probability)
            mutated_offspring = self.mutate(pop,
                                            individual_mutate_probability,
                                            genoelem_mutate_probability)
            candidates = (offspring
                          + crossovered_offspring
                          + mutated_offspring)
            print(f"candidates: {len(candidates)}")
            candidates_fitnesses = list(
                map(lambda x: fitness_calculator, candidates))
            pop = self.select(
                candidates,
                candidates_fitnesses,
                count=self.population_builder.population_size)
            print(f"candidates: {len(pop)}")
            return pop