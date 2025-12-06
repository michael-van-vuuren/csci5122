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

## Scenarios & Examples

### Scenarios

The following scenarios show the networks that emerged under different wind conditions and varying starting positions. For the wind experiments, each episode used a fixed wind speed that remained constant throughout the simulation. For the varying start position experiments, the rocket’s initial position was changed across episodes. In those cases, each network’s fitness was computed as the average performance across three episodes, helping evolution favor networks with strong adaptability to the changing initial conditions. This approach allowed NEAT to discover controllers that were robust to variation in their environment (these networks could not simply overfit to a single trajectory, they actually needed deeper structures).

#### Varying wind:

We can see that the networks learn different actions based on whether there is wind or not. 

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/results-perfect.gif" style="border: 2px solid black;" />
      <p><b>Wind: 20 left</b></p>
      <p>‎</p>
    </td>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/results-no-wind.gif" style="border: 2px solid black;" />
      <p><b>No wind</b></p>
      <p>(before I made clouds move with wind)</p>
    </td>
  </tr>
</table>

#### Varying X-Spawn Location in Center Box

In each episode, the rockets horizontal spawn position was varied. The fitness of each network was based on its performance across three episodes. Using this criteria, the networks with the best adaptability to the changing spawn position were selected.

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none; width: 50%;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/915+(varying-x-center)(total).gif" style="border: 2px solid black;" />
      <p><b>No wind</b></p>
    </td>
    <td style="border: none; width: 50%;"></td>
  </tr>
</table>

The learned networks were deeper than in the simpler scenarios. The hidden nodes were needed to generalize to the random horizontal spawn positions.

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/925(varying-x-center).gif" style="border: 2px solid black;" />
      <p><b>No wind</b></p>
    </td>
    <td style="text-align: center; border: none; width: 40%;">
        <p><b>Generation 50</b></p>
        <img src="https://michael-van-vuuren.github.io/csci5122/images/results/gen50(varying-x-center).png" style="border: 1px dashed black;" />
        <p><b>Generation 500</b></p>
        <img src="https://michael-van-vuuren.github.io/csci5122/images/results/gen500(varying-x-center).png" style="border: 1px dashed black;" />
        <p><b>*Generation 925</b></p>
        <img src="https://michael-van-vuuren.github.io/csci5122/images/results/gen925(varying-x-center).png" style="border: 1px dashed black;" />
    </td>
  </tr>
</table>

#### Varying X-Spawn Location in Right Box

The same process as above was performed, but the spawn box was shifted to the right. 

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none; width: 50%;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/275+(varying-x-right)(total).gif" style="border: 2px solid black;" />
      <p><b>No wind</b></p>
    </td>
    <td style="border: none; width: 50%;"></td>
  </tr>
</table>

In this case, the learned networks were simpler, which makes sense because as you can see below, the rocket barely needed to perform any corrections due to spawning further right. 

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/283(varying-x-right).gif" style="border: 2px solid black;" />
      <p><b>No wind</b></p>
    </td>
    <td style="text-align: center; border: none; width: 40%;">
        <p><b>*Generation 283</b></p>
        <img src="https://michael-van-vuuren.github.io/csci5122/images/results/gen283(varying-x-right).png" style="border: 1px dashed black;" />
    </td>
  </tr>
</table>

#### Varying Y-Spawn Location in Right Box

The same process as above was performed, but now the vertical position of the rockets' spawns was randomized.  

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none; width: 50%;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/350+(varying-y)(total).gif" style="border: 2px solid black;" />
      <p><b>No wind</b></p>
    </td>
    <td style="border: none; width: 50%;"></td>
  </tr>
</table>

In this case, the learned networks were able to adapt to the varying vertical spawn positions very successfully. 

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/351(varying-y).gif" style="border: 2px solid black;" />
      <p><b>No wind</b></p>
    </td>
    <td style="text-align: center; border: none; width: 40%;">
        <p><b>Generation 116</b></p>
        <img src="https://michael-van-vuuren.github.io/csci5122/images/results/gen116(varying-y).png" style="border: 1px dashed black;" />
        <p><b>*Generation 351</b></p>
        <img src="https://michael-van-vuuren.github.io/csci5122/images/results/gen351(varying-y).png" style="border: 1px dashed black;" />
    </td>
  </tr>
</table>

### Examples

The following examples highlight interesting behaviors, edge cases, and strategies discovered by NEAT during training. While scenarios focus on the controllers adapt to  variations in the environment (such as wind or spawn location), these examples show interesting behavior that networks discovered. Some controllers converge to efficient but conservative landing patterns, while others discover surprisingly aggressive or unconventional trajectories. In several cases, the networks exploit physics in unexpected ways by hovering, tipping onto the edge of platforms, or performing dramatic late corrections. These examples show how different network topologies, even under similar conditions, can result in distinct landing styles.

#### Slow vs Fast Landing Example:

The controller on the left lands the rocket very slowly, while the one on the right rapidly falls and burns at the last second. The differnce in landing speed is reflected in the fitness scores for each: 474.8 and 612.2 respectively. 

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/wind10a.gif" style="border: 2px solid black;" />
      <p><b>Slower landing with more fuel consumption</b></p>
    </td>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/wind10b.gif" style="border: 2px solid black;" />
      <p><b>Faster landing with less fuel consumption</b></p>
    </td>
  </tr>
</table>

#### Unexpected Solutions Example:

The controller on the left hovers before landing, and the controller on the right lands on the very tip of the platform. Humans controlling the rockets would be unlikely to perform these behaviors, but the some of the networks eventually do due to NEAT's exploration of solution spaces. 

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/wind40a.gif" style="border: 2px solid black;" />
      <p><b>Hovering before landing</b></p>
    </td>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/wind40b.gif" style="border: 2px solid black;" />
      <p><b>Landing on the tip of the platform</b></p>
    </td>
  </tr>
</table>

#### Specific Network Example:

This example shows the difference in the network structures between an earlier (generation 59) and later (generation 89) generation's highest fitness networks. Notice that the later generation has an extra hidden node which seems to make it burn at a lower height. The lower burn results in a higher fitness value. 

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/59.gif" style="border: 2px solid black;" />
      <p><b>Generation 59</b></p>
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/frame59.png" style="border: 1px dashed black;" />
    </td>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/89.gif" style="border: 2px solid black;" />
      <p><b>Generation 89</b></p>
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/frame89.png" style="border: 1px dashed black;" />
    </td>
  </tr>
</table>

#### Adapting to Strong Wind Example:

The two controllers below learn how to land in very high wind speeds. It is challenging for a human to land the rocket in this scenario, but NEAT was able to evolve controllers that land in this wind. 

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/wind100a.gif" style="border: 2px solid black;" />
      <p><b>Generation 11</b></p>
    </td>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/wind100b.gif" style="border: 2px solid black;" />
      <p><b>Generation 48</b></p>
    </td>
  </tr>
</table>

#### Example of a Controller Struggling:

In some cases, the controllers can have hard time adapting to the conditions. Sometimes, the NEAT process can diverge, which causes instability in the solutions it finds. 

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none; width: 50%;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/results-windy-mishap.gif" style="border: 2px solid black;" />
      <p><b>Wind: 20 right</b></p>
    </td>
    <td style="border: none; width: 50%;"></td>
  </tr>
</table>

#### Interpretability Example:

The purpose of this example is to show that, although it requires thinking, some parts of the evolved networks are somewhat interpretable. Here, the wind is blowing with a value of 30 to the right. The network accounts for this with the negative edge that connects input node -7 (wind sensor reading) to output node 2 (rotate right action). This makes sense because the wind is already blowing the rocket to the right, so the network learns that given this strong wind, it should not turn to the right. 

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/450(interpretability).gif" style="border: 2px solid black;" />
      <p><b>Wind: 30 right</b></p>
    </td>
    <td style="border: none; width: 70%;">
      <img src="https://michael-van-vuuren.github.io/csci5122/images/results/frame_450(interpretability).png" style="border: 1px dashed black;" />
    </td>
  </tr>
</table>
