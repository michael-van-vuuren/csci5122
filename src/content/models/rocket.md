---
title: Rocket Landing
type: docs
prev: models/examples
next: models/_index
weight: 2
math: true
---

## The Model

For the rocket landing simulation, NEAT is used to evolve feedforward networks that act as policy controllers for the 2D rocket. Each network starts simple. Over generations, they grow through structural mutations that add or remove nodes and connections. Connection weights and biases evolve according to configured mutation rates.

### Configuration

The NEAT configuration for the rocket landing problem is set up to find the more complicated networks needed to control the rocket. The population size is 500 networks, which increases the variety of solutions compared to the [XOR and Iris examples](https://michael-van-vuuren.github.io/csci5122/models/examples/). Each network starts with 7 input nodes (rocket state features) and 3 output nodes (action probabilities), with no hidden nodes initially. The networks use hyperbolic tangent (tanh) activations by default, but ReLU, sigmoid, and hat functions are also allowed during mutation. Structural mutations add or remove nodes and connections, while weights and biases evolve with Gaussian initialization and configured mutation rates.

Speciation is controlled with a compatibility threshold of 2.7, which protects niches. Stagnation in fitness score is allowed for a maximum of 30 generations, and the elitism value means that the top 3 networks per species are carried over in each generation. 

The initial connections use the fs_neat method, which means that for each network, a random single input node is initially connected to all output nodes. This starts each genome at a different point in the solution space, similar to dropping multiple “balls” at different positions on a loss surface when performing gradient descent, allowing the population to explore diverse strategies from the start.

{{% details title="Configuration" closed="true" %}}

A guide to understand what each variable controls: [NEAT-Python](https://neat-python.readthedocs.io/en/latest/config_file.html)

```ini {filename="config-feedforward-rocket.ini"}
[NEAT]
fitness_criterion = max
fitness_threshold = 650
pop_size = 500
reset_on_extinction = False
no_fitness_termination = False

[TrackedGenome]
num_hidden = 0
num_inputs = 7
num_outputs = 3

activation_default = tanh
activation_mutate_rate = 0.1
activation_options = tanh relu sigmoid hat

aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum

bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.2
bias_mutate_rate = 0.8
bias_replace_rate = 0.02

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5

conn_add_prob = 0.15
conn_delete_prob = 0.05

enabled_default = True
enabled_mutate_rate = 0.01

feed_forward = True
initial_connection = fs_neat

node_add_prob = 0.05
node_delete_prob = 0.005

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
weight_mutate_power = 0.2
weight_mutate_rate = 0.9
weight_replace_rate = 0.02

single_structural_mutation = false
structural_mutation_surer = default
bias_init_type = gaussian
response_init_type = gaussian
weight_init_type = gaussian
enabled_rate_to_true_add = 0.0
enabled_rate_to_false_add = 0.0

[DefaultSpeciesSet]
compatibility_threshold = 2.7

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 30
species_elitism = 3

[DefaultReproduction]
elitism = 5
survival_threshold = 0.2
min_species_size = 2
```

{{% /details %}}

## The Inputs & Outputs

As was discussed in the [Data & Prep section](https://michael-van-vuuren.github.io/csci5122/eda/), at each time step in the simulation, the rocket records the following state variables:
>1. x-position
>2. y-position
>3. x-velocity
>4. y-velocity
>5. angle
>6. angular velocity
>7. wind sensor reading

These are the 7 input nodes each network receives at each time step during training. At the same time, the rocket can take any combination of three actions:
>1. apply thrust
>2. rotate left
>3. rotate right

These are the 3 output nodes of each network at each time step during training. They represent the probability that a model should take an action. An action is triggered if the corresponding output exceeds a threshold. Multiple actions can be chosen simultaneously (if the multiple action probabilities exceed the threshold).

## The Fitness Function

The fitness function for this problem is very important, because it directly influences the behavior that NEAT is optimizing for. For the problem of gently landing the rocket in the 2D simulation, fitness is calculated based on a combination of positional, angular, and efficiency criteria:

**1. Running rewards:**
  - Base fitness starts at -100.0 (arbitrary).
  - Being closer to the platform increases fitness (uses Euclidean distance).
  - Upright angle increases fitness.
  - More fuel used decreases fitness.
  - If the rocket is near screen edges, fitness decreases.

**2. End state rewards/penalties (assigned at the end of each simulation episode):**
  - **Exploded:**
    - Fitness decreases for crashing.
    - Crashed closer to center gives a small fitness bonus.
    - Slower crashes give a small bonus.
  - **Landed:**
    - Fitness increases for landing.
    - Landing closer to center increases fitness.
    - Slower landing increases fitness.
    - Less upright landing decreases fitness.
- **Time bonus/penalty:**
  - Faster landing gives a bonus.
  - If max time is reached and the rocket is still in the air, fitness decreases for hovering too long.

{{% details title="The Fitness Function" closed="true" %}}

```python {filename="fitness.py"}
# fitness function (kinda like a reward function in reinforcement learning)
def fitness_function(rocket):
    # calculate the euclidian distance from rocket to platform
    center_x = PLAT_X + PLAT_W / 2
    dtp_x = abs(rocket.x - center_x)
    dtp_y = abs(rocket.y - PLAT_Y)
    dtp = math.sqrt(dtp_x**2 + dtp_y**2)

    # calculate upright deviation angle
    angle_deviation = abs(rocket.theta - (-math.pi/2)) 
    
    fitness = -100.0                                         # 1. running rewards               fitness
    fitness += (1000 - dtp) / 10.0                           # closer distance to platform    = increase
    fitness += (3.0 - angle_deviation) * 10                  # upright angle                  = increase

    fuel_penalty = rocket.fuel_used * 0.5                    # more fuel use                  = decrease
    fitness -= fuel_penalty

    wall_threshold = 50                                      # use screen walls for stability = decrease
    if rocket.x < wall_threshold:
        fitness -= (wall_threshold - rocket.x) * 2
    elif rocket.x > SCREEN_W - wall_threshold:
        fitness -= (rocket.x - (SCREEN_W - wall_threshold)) * 2

    if rocket.game_state == 'EXPLODED':                      # 2. end state rewards
        fitness -= 50                                        # crashed                        = decrease
        fitness += (SCREEN_W/2 - dtp_x) * 0.4                # crashed closer to center       = increase
        fitness += max(0, 200 - rocket.speed) * 0.5          # softer crash                   = increase
    elif rocket.game_state == 'LANDED':
        fitness += 200                                       # landed                         = increase
        fitness += (200 - dtp_x)                             # landed closer to center        = increase
        fitness += max(0, (20 - rocket.speed) * 3)           # softer landing                 = increase
        fitness -= angle_deviation * 150                     # less upright landing           = decrease

        time_bonus = max(0, 500 - rocket.time_taken) * 0.5
        fitness += time_bonus
    if rocket.time_taken >= MAX_TIME and rocket.game_state == 'RUNNING':       
        fitness -= 200                                       # hovering in the air            = decrease

    return fitness
```

{{% /details %}}

## Links

