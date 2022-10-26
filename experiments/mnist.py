import typing as tp
import numpy as np
from torch import nn
from nasge.experiment import BaseExperiment
from torch.utils.data import DataLoader, Dataset
from nasge.genetic_algorithm import Individual, GAEvolution
from nasge import gramm_parser as gp
from nasge import utils as nasge_utils
from torch import optim
import torch
import torchvision.datasets as datasets
import warnings

warnings.filterwarnings("ignore")


class MNISTExperiment(BaseExperiment):
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
        super().__init__(
            input_model_builder, output_model_builder, grammar_path,
            genotype_size, genoelement_range, grammar_builder,
            genotype_builder, phenotype_builder, model_builder,
            instance_builder_class, ga_evolution_class, population_size,
            train_dloader, valid_dloader, offspring_fraction,
            crossover_probability, individual_mutate_probability,
            genoelem_mutate_probability, epochs, select_max, optimizer_class,
            criterion, lr
        )
        self.logger = nasge_utils.get_logger("MNISTExperiment")

    def train_network(self,
                      train_dataloader: DataLoader,
                      valid_dataloader: DataLoader):
        def inner_train(individ: Individual):
            individ.full_model = self.init_model(self.input_model_builder,
                                                 self.output_model_builder,
                                                 individ.model.model,
                                                 individ.model.input_size,
                                                 individ.model.output_size)
            optimizer = self.optimizer_class(individ.full_model.parameters(),
                                             lr=self.lr)
            epochs = 1
            for epoch in range(epochs): #tqdm.tqdm(range(epochs)):
                for ix, (X_train, y_train) in enumerate(train_dataloader):
                    preds = individ.full_model.forward(X_train)
                    loss = self.criterion(preds, y_train[:,0])
                    loss.backward()
                    optimizer.step()
                    # if ix > 4:
                    #     break
            with torch.no_grad():
                individ.full_model.eval()
                X_valid, y_valid = next(iter(valid_dataloader))
                valid_preds = individ.full_model.forward(X_valid)
                # loss = criterion(valid_preds, y_valid[:, 0])

                pred_cls = (valid_preds
                            .argmax(dim=1, keepdim=True)[:, 0]
                            .detach()
                            .numpy())
                true_cls = y_valid[:, 0].detach().numpy()
                accuracy = np.sum(pred_cls == true_cls)/true_cls.shape[0]
                return np.round(accuracy, 3)
        return inner_train

    def run(self):
        best_score, best_individ, fitnesses, pops = self.evalution.run()
        return best_score, best_individ, fitnesses, pops


class MNISTDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        img = np.array(image)/255
        return (torch.FloatTensor(img[:, :, None]),
                torch.LongTensor([label]))


def input_model_builder(input_size: int):
    def builder(inner_model_input_size: int):
        return nn.Sequential(*[nn.Flatten(),
                               nn.Linear(input_size, inner_model_input_size)])
    return builder


def output_model_builder(output_size: int):
    def builder(inner_model_output_size: int):
        return nn.Sequential(*[nn.Linear(inner_model_output_size, output_size)])
    return builder


def load_dataset(train_fraction: float):
    mnist_trainset = datasets.MNIST(root='./../data',
                                    train=True,
                                    download=True,
                                    transform=None)
    N = len(mnist_trainset)
    train_N = int(N * train_fraction)
    train_ixs = np.random.choice(range(N), size=train_N, replace=False)
    valid_ixs = np.asarray(list(set(range(N)) - set(train_ixs)))

    train_dataset = MNISTDataset(np.array(mnist_trainset)[train_ixs])
    valid_dataset = MNISTDataset(np.array(mnist_trainset)[valid_ixs])
    return train_dataset, valid_dataset


if __name__ == "__main__":
    config = nasge_utils.load_yaml("./../experiments/mnist_config.yaml")
    parameters = config["parameters"]

    train_dataset, valid_dataset = load_dataset(parameters["train_fraction"])

    train_dloader = DataLoader(train_dataset,
                               batch_size=parameters["train_batch_size"],
                               shuffle=True)
    valid_dloader = DataLoader(valid_dataset,
                               batch_size=parameters["valid_batch_size"],
                               shuffle=True)

    ptmt = MNISTExperiment(
        input_model_builder=input_model_builder(int(28*28)),
        output_model_builder=output_model_builder(10),
        grammar_path=parameters["grammar_path"],
        genotype_size=parameters["genotype_size"],
        genoelement_range=parameters["genoelement_range"],
        grammar_builder=gp.ContextFreeGrammar,
        genotype_builder=gp.InstanceGenotype,
        phenotype_builder=gp.InstancePhenotype,
        model_builder=gp.InstancePyTorchModel,
        instance_builder_class=gp.InstanceBuilder,
        ga_evolution_class=GAEvolution,
        population_size=parameters["population_size"],
        train_dloader=train_dloader,
        valid_dloader=valid_dloader,
        offspring_fraction=parameters["offspring_fraction"],
        crossover_probability=parameters["crossover_prob"],
        individual_mutate_probability=parameters["individual_mutate_prob"],
        genoelem_mutate_probability=parameters["genoelem_mutate_prob"],
        epochs=parameters["epochs"],
        select_max=parameters["select_max"],
        optimizer_class=optim.SGD,
        criterion=nn.CrossEntropyLoss(),
        lr=parameters["lr"]
    )

    best_score, best_individ, fitnesses, pops = ptmt.run()
