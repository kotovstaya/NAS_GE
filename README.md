# NAS_grammatical_evolution

## Neural Architecture Search using Grammatical Evolution


### What is NAS?
If you want to learn more about Neural Architecture Search you can check this site
https://en.wikipedia.org/wiki/Neural_architecture_search


### What is a neural network?
If you want to learn more you can check this site
https://en.wikipedia.org/wiki/Neural_network
In this project I used a neural network for specific purposes.
I wanted to find the best solution for the problem. For example, the MNIST classification.
It's very easy task, yeah, but on this task I saw that it works and can be improved for 
further problems.

### What is a genetic algorithm?  
If you want to learn more you can check this site
https://en.wikipedia.org/wiki/Genetic_algorithm

Generally speaking, a genetic algorithm is a framework that helps you find the optimal solution using the rules of nature:
1. Creation of a population
2. Mutation
3. Crossover
4. Selection

You are creating populations. Let it be some architecture of a particular neural network.
And you are trying to use operators: mutation and crossover to get better and better options for this architecture.
In the end, you will get the best one and be able to use it in your project.

### What is a grammatical evolution?
If you want to learn more you can check this site
https://en.wikipedia.org/wiki/Grammatical_evolution

I just note that it's very easy to create some structure (for example, neural network structure) using a banch of rules.
These rules are fixed but the process of a generation neural network is semi-stochastic and based on the randomness and genetic algorithm.
You just write the grammar in the correct format. Then run the genetic algorithm and make tea or coffee. That's all.
In a few seconds/minutes/hours you'll have a solution.

Grammar example:
```
enter_expr: <expr>
grammar:
    <expr>:
        - <fc>$
        - <fc>$<expr>$
        - <fc>$<bn>$
        - <fc>$<do>$<bn>$
        - <fc>$<bn>$<do>$
        - <fc>$<bn>$<expr>$
    <fc>:
      - layer,fc,in_features:<in_ftrs>,out_features:<out_ftrs>
    <bn>:
      - layer,bn,num_features:<num_ftrs>
    <do>:
      - layer,do,p:float_rnd_0-1
    <in_ftrs>:
      - int_rnd_10-20
      - int_rnd_20-30
    <act>:
      - layer,sigmoid,empty
      - layer,tanh,empty
    <num_ftrs>:
      - int_rnd_10-20
    <out_ftrs>:
      - int_rnd_10-20
      - int_rnd_20-30
number_mapping:
  int_rnd_10-20: [10, 20]
  int_rnd_20-30: [20, 30]
  float_rnd_0-1: [0, 1]
```


### How use it?
There are 4 folders:
1. data - collect and save datasets
2. experiments - folder for experiments 
3. grammers - folder with grammars in .yaml format 
4. nasge - folder with .py files

See some implementation for MNIST dataset:

created config for MNIST experiment:
```
parameters:
  input_size: 784
  class_count: 10
  optimizer: torch.optim.SGD
  criterion: torch.nn.CrossEntropyLoss()
  genoelement_range: [0, 30]
  genotype_size: 20
  grammar_path: ./grammers/fc.yaml
  model_path: ./experiments/mnist/mnist_model.pcl
  population_size: 6
  offspring_fraction: 0.9
  crossover_prob: 0.9
  individual_mutate_prob:  0.9
  genoelem_mutate_prob: 0.9
  epochs: 1
  select_max: True
  train_fraction:  0.9
  train_batch_size: 256
  valid_batch_size: 6000
  test_batch_size: 1000
  lr: 0.001

```

Create an environment and run some .sh scripts

```
1. python -m pip install -r requirements.txt
2. ./train_mnist.sh
3. ./inference_mnist.sh
```

After these steps you'll see something like that
```
INFO:Inference:accuracy: 0.82
INFO:Inference:Model(
  (input_model): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=784, out_features=36, bias=True)
  )
  (inner_model): Sequential(
    (0): Linear(in_features=36, out_features=27, bias=True)
    (1): Sigmoid()
    (2): BatchNorm1d(27, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout1d(p=0.041, inplace=False)
    (4): Linear(in_features=27, out_features=28, bias=True)
    (5): BatchNorm1d(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
  )
  (output_model): Sequential(
    (0): Linear(in_features=28, out_features=10, bias=True)
  )
)
```
This is an optimal architecture for mnist task which have been found.
