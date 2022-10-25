import typing as tp

import torch
import torchvision.datasets as datasets
from nasge import context_free_grammar as cfg
from torch import nn
from torch.utils.data import DataLoader
from nasge.genetic_algorithm import GAEvolution, Individual
from torch import optim
import numpy as np
from torch.utils.data import Dataset


class BaseExperiment:
    def __init__(
            self,
            input_model,
            eval_class,
            grammer_class,
            model_builder,
            parser_class,
            input_size,
            output_size,
            phenotype_size,
            phenotype_range,
            population_size
    ):
        self.input_model = input_model
        self.grammer_class = grammer_class
        self.model_builder = model_builder
        self.parser_class = parser_class
        self.input_size = input_size
        self.output_size = output_size
        self.phenotype_size = phenotype_size
        self.phenotype_range = phenotype_range
        self.population_size = population_size

        self.population_builder = cfg.ContextFreePopulation(
            self.grammer_class,
            self.model_builder,
            self.parser_class,
            self.input_size,
            self.output_size,
            self.phenotype_size,
            self.phenotype_range,
            self.population_size
        )
        self.evaluator = eval_class(self.population_builder)

    def init_model(self, input_model: nn.Sequential, model:nn.Sequential):
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def inference(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, *args, **kwargs):
        raise NotImplementedError()


class Experiment(BaseExperiment):
    def __init__(
            self,
            input_model,
            eval_class,
            grammer_class,
            model_builder,
            parser_class,
            input_size,
            output_size,
            phenotype_size,
            phenotype_range,
            population_size
    ):
        super().__init__(
            input_model,
            eval_class,
            grammer_class,
            model_builder,
            parser_class,
            input_size,
            output_size,
            phenotype_size,
            phenotype_range,
            population_size
        )

    def init_model(self, input_model: nn.Sequential, model: nn.Sequential):
        class Model(nn.Module):
            def __init__(self, input_model: nn.Sequential, model: nn.Sequential):
                super().__init__()
                self.input_model = input_model
                self.model = model

            def forward(self, x):
                x = self.input_model.forward(x)
                out = self.model.forward(x)
                return out

        return Model(input_model, model)

    def train_network(self,
                      train_dataloader: DataLoader,
                      valid_dataloader: DataLoader):
        def inner_train(individ: Individual):
            individ.full_model = self.init_model(self.input_model, individ.model)
            optimizer = optim.SGD(individ.full_model.parameters(),
                                  lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            epochs = 10
            for epoch in range(epochs):
                for ix, X_train, y_train in enumerate(train_dataloader):
                    preds = individ.full_model.forward(X_train)
                    loss = criterion(preds, y_train)
                    loss.backward()
                    optimizer.step()
            with torch.no_grad():
                individ.full_model.eval()
                X_valid, y_valid = next(iter(valid_dataloader))
                valid_preds = individ.full_model.forward(X_valid)
                loss = criterion(valid_preds, y_valid)
                return loss.item()
        return inner_train

    def run(self,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
            offspring_fraction: float = 0.2,
            crossover_probability: float = 0.5,
            individual_mutate_probability: float = 0.5,
            genoelem_mutate_probability: float = 0.2,
            epochs: int = 10,):
        self.evaluator.eval(
            fitness_calculator=self.train_network(train_dataloader, valid_dataloader),
            offspring_fraction=offspring_fraction,
            crossover_probability=crossover_probability,
            individual_mutate_probability=individual_mutate_probability,
            genoelem_mutate_probability=genoelem_mutate_probability,
            epochs=epochs,
        )

class MNISTDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        img = np.array(image)/255
        return torch.FloatTensor(img[:, :, None]), torch.LongTensor([label])


if __name__ == "__main__":
    mnist_trainset = datasets.MNIST(root='./../data', train=True, download=True, transform=None)
    N = len(mnist_trainset)
    train_N = int(N * 0.9)
    train_ixs = np.random.choice(range(N), size=train_N, replace=False)
    valid_ixs = np.asarray(list(set(range(N)) - set(train_ixs)))

    train_data = np.array(mnist_trainset)[train_ixs]
    valid_data = np.array(mnist_trainset)[valid_ixs]

    train_dataset = MNISTDataset(train_data)
    valid_dataset = MNISTDataset(valid_data)

    train_dloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_dloader = DataLoader(valid_dataset, batch_size=6000, shuffle=True)

    class InputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()

        def forward(self, x):
            return self.flatten()

    ptmt = Experiment(
        input_model=InputModel(),
        eval_class=GAEvolution,
        grammer_class=cfg.DefaultGrammar,
        model_builder=cfg.PyTorchModelBuilder,
        parser_class=cfg.ContextFreeParser,
        input_size=100,
        output_size=1,
        phenotype_size=30,
        phenotype_range=(0, 30),
        population_size=3
    )

    ptmt.run(
        train_dloader,
        valid_dloader,
        offspring_fraction=0.2,
        crossover_probability=0.5,
        individual_mutate_probability=0.5,
        genoelem_mutate_probability=0.2,
        epochs=10,
    )


