---
title: Results
type: results
math: true
---

Using NEAT, the networks evolved over generations to control the 2D rocket efficiently. In the early generations, rockets often crashed, hovered, or drifted off course. Over time, however, the algorithm discovered networks that consistently landed upright, near the center of the platform, while using minimal fuel. The highest fitness networks learned to adjust thrust and rotation dynamically, taking into account the rocket’s position, velocity, angle, and wind conditions.

## Training Overview

The population of 500 networks explored a wide variety of strategies. Speciation and a compatibility threshold of 2.7 protected diverse strategies, which allowed multiple viable approaches to emerge at the same time.
<br>
<br>
<img src="https://michael-van-vuuren.github.io/csci5122/images/results/stats.png" style="width: 50%; display: block; margin: 0 auto; border: 2px solid black;">

The image above shows the output of NEAT for generation 67. The first line represents the average fitness of all networks in the population, along with the standard deviation. The second line shows the network with the current highest fitness (as measured by the [fitness function](https://michael-van-vuuren.github.io/csci5122/models/rocket/#the-fitness-function)). The fourth line shows the average genetic distance of networks, which shows how diverse the population is.
 
The table lists each currently existing species. Notice that all species have relatively high maximum fitness scores within a similar range (~400–500). Despite similar performance, the species have sufficiently different network structures, meaning there are multiple distinct solutions that perform well. This diversity illustrates the strength of NEAT in exploring multiple valid strategies.

## Model Performance

The GIF below shows the highest fitness networks across 100 generations along with their parents. In this iteration, the wind speed was set to 0.

![](/images/results/perf_rocket_evolution.gif)

The highest fitness member of each species in each generation (100 total) is represented by a distinct colored line in the plot below. Notice how over time, many different species converge toward a high fitness score. This is a good sign that NEAT was able a variety of valid solutions. 

![](/images/results/perf_rocket_species_fitness.png)

The convergence of training could also be observed in the replays. As generations increased, the number of crashes decreased, the landings became more upright and toward the center of the platform, and the fuel usage reduced. The networks eventually learned that in order to minimize fuel usage, they should activate thrust at the last possible moment, which is similar to the ["suicide burns"](https://en.wiktionary.org/wiki/suicide_burn) that SpaceX rockets perform during landings.

## Scenarios

The following scenarios show the networks that emerged under different wind conditions and varying starting positions. For the wind experiments, each episode used a fixed wind speed that remained constant throughout the simulation. For the varying start position experiments, the rocket’s initial position was changed across episodes. In those cases, each network’s fitness was computed as the average performance across three episodes, helping evolution favor networks with strong adaptability to the changing initial conditions. This approach allowed NEAT to discover controllers that were robust to variation in their environment (these networks could not simply overfit to a single trajectory, they actually needed deeper structures).

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/results-perfect.gif" alt="Wind: 20 left" />
      <p><b>Wind: 20 left</b></p>
    </td>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/results-no-wind.gif" alt="No wind" />
      <p><b>No wind</b></p>
    </td>
  </tr>
</table>

## Links

