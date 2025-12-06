---
title: Introduction
type: introduction
---

## An Introduction

Being able to travel to space quickly and inexpensively has many benefits. A few of them are
>**Scientific research**, such as climate, astronomy, medicine, and material research.\
>**Satellite deployment and maintenance**, which supports GPS, weather monitoring, internet, and disaster management.\
>**Deep space exploration**, which will be necessary for survival in the future, and also could be useful for resource mining. 

To learn more, visit ([NASA](https://www.nasa.gov/humans-in-space/why-go-to-space/)).

Before reliable booster-landing technology existed, every part of a rocket except the payload was destroyed on each launch, making them extremely expensive. Reusing the first stage changes this dramatically: it can reduce launch costs by up to a half or more ([Engadget](https://www.engadget.com/2017-04-06-spacex-is-saving-a-ton-of-money-by-re-using-falcon-9-rockets.html)) while also saving the materials and energy required to build new boosters. This leads to faster launch turnarounds, lower waste, and more efficient access to space.

The goal of this project is to explore how a neural network architecture be evolved as a policy for adaptive rocket landing control. It is inspired by the SpaceX and Blue Origin first stage rocket landings. 

{{< cards col="2" >}}
  {{< card link="https://youtu.be/Aq7rDQx9jns?si=Nm0fn8YJxgg2JBnr&t=21" image="/images/introduction/spacex.png" tag="Click for video!" title="SpaceX's Falcon 9" target="_blank" >}}
  {{< card link="https://youtu.be/S2NlWjNWvXo?si=-DSH7t4r-i97BtYr&t=56" image="/images/introduction/blueorigin.png" tag="Click for video!" title="Blue Origin's New Glenn" target="_blank" >}}
{{< /cards >}}

For this project, the simulation is handwritten using [Pygame](https://www.pygame.org/docs/), but it is also a dramatic simplification of reality. For example, in real life, a rocket needs to account for drag forces, atmospheric conditions, variations in the Earth's gravitational field, and propellant flow rates, among other factors. In this simulation, the rocket must manage its position, velocity, angle, and angular speed, and in some cases account for wind. Moreover, in real life, rockets operate in three dimensions, whereas in this simulation, the rocket operates in two dimensions.

Also note that, in real aerospace engineering, rocket landings never (fully) rely on deep learning. Instead, they mostly rely on classical control systems like PID controllers, guidance laws, and model-based control. These methods are deterministic and verifiable, which is obviously necessary for rocket control.

Despite the simplification of reality, the focus of this project is on how neuroevolution can be used to evolve a policy network that enables an agent, which is a decision making controller acting within the environment, to autonomously and smoothly land a simulated rocket. It is more about the neural networks than it is about the rocket science.

## Why Care?

- Safe landings enable cheaper space flights.
- Automated systems can perform better than humans in many situations if built correctly. 
- Understanding how to create autonomously balancing and moving agents has applications in other fields like robotics.
- Neuroevolution is applicable to many problems, especially ones with sparse rewards or unsolvable gradients.

## Neuroevolution

Neuroevolution uses evolutionary algorithms to evolve the structure or weights of neural networks. Evolutionary algorithms are optimization methods inspired by natural selection, where the fittest solutions are more likely to survive and propagate. In this project, the networks are not preconfigured and trained using backpropagation. Rather, they emerge from the neuroevolution process. 

![](/images/introduction/NE.png)
![](/images/introduction/NE-steps.png)

Although neuroevolution should not be applied to all situations, it suits specific situations very well: 

![](/images/introduction/NEAT-benefits-downsides.png)

## NeuroEvolution of Augmenting Topologies

Also known as NEAT, this algorithm evolves both the weights and the structure of neural networks. Such networks are called topology and weight evolving artifical neural networks (TWEANNs).

![](/images/introduction/NEAT-paper.png)

The NEAT paper, written by Stanley and Miikkulainen in 2002, introduced three key techniques that made the efficient creation of TWEANNs possible:
1. **Tracking historical markings** by assigning innovation numbers to genes to align networks during crossover.
2. **Speciation** groups similar networks to protect new innovations from being discarded too early (similar to ecological niches in nature).
3. **Incremental growth** starts with simple networks and gradually adds nodes and connections to increase complexity.

![](/images/introduction/NEAT.png)

In this project, [NEAT-Python](https://neat-python.readthedocs.io/en/latest/), a python library that implements NEAT, is used to control the crossover, speciation, and mutation of networks.

![](/images/introduction/NEAT-process.png)

## Objectives

1. Explore how neuroevolution can evolve neural networks for a simple control task.
2. Evolve a policy network to autonomously land a simulated rocket.
3. Test and evaluate control strategies in a simplified Pygame simulation.
4. Configure and apply NEAT to improve network performance.

## The Code

All of the code can be found in the following GitHub repository: https://github.com/michael-van-vuuren/csci5122-workspace/