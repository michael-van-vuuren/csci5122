---
title: XOR and Iris Examples
type: docs
prev: models/_index
next: models/rocket
math: true
weight: 1
---

## The Model

For these example problems, the NEAT program is set up with a population of 300 networks. Each network starts simple: all input and output nodes are fully connected, and there are no hidden nodes at first. Over time, structural mutations let the networks grow by adding or removing nodes and connections. The weights and biases of connections also evolve according to configured mutation rates.

### Configuration

Each genome uses a sigmoid activation function by default, but that can change to ReLU when mutated. The outputs from multiple connections are summed together, like in a usual neural network. To maintain diversity, networks are grouped into species based on a compatibility threshold, which decides when two networks are different enough to be part of two different species.

The algorithm also monitors stagnation to prevent progress from stalling and uses elitism and survival thresholds to ensure that the strongest networks in each species carry over to the next generation. The selection of high fitness individuals for crossing and carry over allows the fitness of the networks to improve and eventually converge over time. The mutation of each genome allows NEAT to explore a variety of solutions, helping it get out of local minima. 

The file used to configure the NEAT program is long, but if you are interested, check below:

{{% details title="Configuration" closed="true" %}}

A guide to understand what each variable controls: [NEAT-Python](https://neat-python.readthedocs.io/en/latest/config_file.html)

```ini {filename="config-feedforward.ini"}
[NEAT]
fitness_criterion = max
fitness_threshold = 118 ; set to 4.0 for the XOR problem
pop_size = 300
reset_on_extinction = False
no_fitness_termination = False

[TrackedGenome]
num_inputs = 4 ; set to 2 for the XOR problem
num_hidden = 0
num_outputs = 3 ; set to 1 for the XOR problem
activation_default = sigmoid
activation_mutate_rate = 0.1
activation_options = sigmoid relu
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.05
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob = 0.5
conn_delete_prob = 0.2
enabled_default = True
enabled_mutate_rate = 0.01
feed_forward = True
initial_connection = full
node_add_prob = 0.3
node_delete_prob = 0.2
response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_max_value = 30
weight_min_value = -30
weight_mutate_power = 0.5
weight_mutate_rate = 0.3
weight_replace_rate = 0.05
single_structural_mutation = false
structural_mutation_surer = default
bias_init_type = gaussian
response_init_type = gaussian
weight_init_type = gaussian
enabled_rate_to_true_add = 0.0
enabled_rate_to_false_add = 0.0

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 15
species_elitism = 2

[DefaultReproduction]
elitism = 20
survival_threshold = 0.2
min_species_size = 2
```

{{% /details %}}

## XOR

The XOR problem is a test in neural networks where the goal is to learn the exclusive-or (XOR) function, which outputs 1 if exactly one of the two binary inputs is 1, and otherwise outputs 0. The XOR function:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
|    0    |    0    |   0    |
|    0    |    1    |   1    |
|    1    |    0    |   1    |
|    1    |    1    |   0    |

### The Inputs

The network has 2 input nodes, which represent the two binary values an XOR operation takes (0 or 1).

### The Outputs

There is 1 output node, which corresponds to the prediction of the network of the XOR function (0 or 1).

### The Fitness Function

The fitness of each network is calculated by taking the mean squared error (MSE) between the predicted output and the correct XOR output across the four possible input combinations. A perfect fitness would be 4.0. Networks with lower MSE values are more fit, and are more likely to survive into further generations. 

### Results

![](/images/models/xor.png)

With no bypass connections enabled, after less than 49 generations, the highest fitness network discovered the most efficient possible network structure for solving the XOR problem. With bypass connections enabled, the highest fitness network found a solution that used only one hidden node to solve the XOR problem. This shows that neuroevolution is capable of finding efficient and optimal solutions on simple problems. 

## Iris

### The Inputs

The network has 4 input nodes, corresponding to the four features of each Iris flower: sepal length, sepal width, petal length, and petal width.

### The Outputs

There are 3 output nodes, one for each Iris species (Setosa, Versicolor, Virginica). The network predicts the species by producing the highest output value among the three nodes. Notice that we do not use a softmax function, even though there are three classes. Through evolution, the network instead learns to use the raw output values directly for making predictions (softmax can be integrated into the initial models as well if desired).

### The Fitness Function

Fitness is calculated using accuracy on the training set: the number of correctly classified samples out of the total. Since an 80-20 training-test split was used, a perfect fitness corresponds to correctly predicting all 120 training samples. Networks with higher accuracy are more likely to propagate to the next generation.

### Results

![](/images/models/iris.png)

Using NEAT, the network quickly evolved to achieve high accuracy on the Iris dataset. Over generations, the algorithm discovered networks that could correctly classify the training samples. The high fitness networks that were found were also very simple, which makes sense because the Iris dataset is relatively easy to classify. This shows that neuroevolution can solve small, multiclass classification problems.

The following GIF shows, for each generation, the highest fitness child networks along with their parents. Note that in some generations, the highest fitness child network does not have any parents, or only has a single parent. This is because NEAT allows for asexual reproduction and the creation of new species, so some networks can emerge without two-parent crossover or from newly spawned species that have no parents in the previous generation. 

![](/images/models/iris.gif)

Over successive generations, the fitness of the best performing networks increases steadily, eventually approaching an asymptote at 117.0. The networks do not reach the perfect fitness of 120.0 because a regularization term is applied to each network’s fitness score, which subtracts points based on the number of connections to encourage simpler, more efficient network structures.

## Links

