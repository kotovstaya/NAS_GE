import torch
from torch.utils.data import DataLoader

from experiment import (
    load_train_valid_dataset,
    MNISTExperiment,
    input_model_builder,
    output_model_builder
)
from nasge import genetic_algorithm as ga
from nasge import gramm_parser as gp
from nasge import utils as nasge_utils


if __name__ == "__main__":
    config = nasge_utils.load_yaml("config.yaml")
    parameters = config["parameters"]

    train_dataset, valid_dataset = load_train_valid_dataset(
                                                parameters["train_fraction"])

    train_dloader = DataLoader(train_dataset,
                               batch_size=parameters["train_batch_size"],
                               shuffle=True)
    valid_dloader = DataLoader(valid_dataset,
                               batch_size=parameters["valid_batch_size"],
                               shuffle=True)

    ptmt = MNISTExperiment(
        input_model_builder=input_model_builder(parameters["input_size"]),
        output_model_builder=output_model_builder(parameters["class_count"]),
        grammar_path=parameters["grammar_path"],
        genotype_size=parameters["genotype_size"],
        genoelement_range=parameters["genoelement_range"],
        grammar_builder=gp.ContextFreeGrammar,
        genotype_builder=gp.InstanceGenotype,
        phenotype_builder=gp.InstancePhenotype,
        model_builder=gp.InstancePyTorchModel,
        instance_builder_class=gp.InstanceBuilder,
        ga_evolution_class=ga.GAEvolution,
        population_size=parameters["population_size"],
        train_dloader=train_dloader,
        valid_dloader=valid_dloader,
        offspring_fraction=parameters["offspring_fraction"],
        crossover_probability=parameters["crossover_prob"],
        individual_mutate_probability=parameters["individual_mutate_prob"],
        genoelem_mutate_probability=parameters["genoelem_mutate_prob"],
        epochs=parameters["epochs"],
        select_max=parameters["select_max"],
        optimizer_class=eval(parameters["optimizer"]),
        criterion=eval(parameters["criterion"]),
        lr=parameters["lr"]
    )

    best_score, best_individ, fitnesses, pops = ptmt.run()
    ptmt.save(parameters["model_path"])
    
