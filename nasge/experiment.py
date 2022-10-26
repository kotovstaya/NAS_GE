import typing as tp

import torch.optim
from torch import nn
from torch.utils.data import DataLoader

from nasge import gramm_parser as gp
from nasge.genetic_algorithm import GAEvolution


class Model(nn.Module):
    def __init__(self,
                 input_model_builder,
                 output_model_builder,
                 inner_model_input_size,
                 inner_model_output_size,
                 inner_model):
        super().__init__()
        self.input_model = input_model_builder(inner_model_input_size)
        self.inner_model = inner_model
        self.output_model = output_model_builder(
            inner_model_output_size)

    def forward(self, x):
        x = self.input_model.forward(x)
        x = self.inner_model.forward(x)
        x = self.output_model.forward(x)
        return x


class BaseExperiment:
    def __init__(
            self,
            input_model_builder: callable,
            output_model_builder: callable,
            grammar_path: str,
            genotype_size: int,
            genoelement_range: tp.Tuple[int, int],
            grammar_builder: tp.Type[gp.ContextFreeGrammar],
            genotype_builder: tp.Type[gp.InstanceGenotype],
            phenotype_builder: tp.Type[gp.InstancePhenotype],
            model_builder: tp.Type[gp.InstancePyTorchModel],
            instance_builder_class: tp.Type[gp.InstanceBuilder],
            ga_evolution_class: tp.Type[GAEvolution],
            population_size: int,
            train_dloader: DataLoader,
            valid_dloader: DataLoader,
            offspring_fraction: float,
            crossover_probability: float,
            individual_mutate_probability: float,
            genoelem_mutate_probability: float,
            epochs: int,
            select_max: bool,
            optimizer_class: tp.Type[torch.optim.Optimizer],
            criterion,
            lr: float,
    ):
        self.input_model_builder = input_model_builder
        self.output_model_builder = output_model_builder
        self.grammar_path = grammar_path
        self.genotype_size = genotype_size
        self.genoelement_range = genoelement_range
        self.optimizer_class = optimizer_class
        self.criterion = criterion
        self.lr = lr

        instance_builder = instance_builder_class(
            grammar_builder=grammar_builder,
            genotype_builder=genotype_builder,
            phenotype_builder=phenotype_builder,
            model_builder=model_builder,
            grammar_params={'filepath': grammar_path},
            genotype_params={'genoelement_range': genoelement_range,
                             'genotype_size': genotype_size},
            phenotype_params={},
            model_params={}
        )

        self.evalution = ga_evolution_class(
            instance_builder=instance_builder,
            genoelement_range=genoelement_range,
            genotype_size=genotype_size,
            population_size=population_size,
            fitness_calculator=self.train_network(train_dloader, valid_dloader),
            offspring_fraction=offspring_fraction,
            crossover_probability=crossover_probability,
            individual_mutate_probability=individual_mutate_probability,
            genoelem_mutate_probability=genoelem_mutate_probability,
            epochs=epochs,
            select_max=select_max
        )

    def train_network(self,
                      train_dataloader: DataLoader,
                      valid_dataloader: DataLoader):
        pass

    def init_model(self,
                   input_model_builder: callable,
                   output_model_builder: callable,
                   inner_model: nn.Sequential,
                   inner_model_input_size: int,
                   inner_model_output_size: int
                   ):

        return Model(
            input_model_builder,
            output_model_builder,
            inner_model_input_size,
            inner_model_output_size,
            inner_model
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, *args, **kwargs):
        raise NotImplementedError()
