---
title: Conclusion
type: conclusion
---

## Main Takeaways

- With neuroevolution, we evolved controllers capable of stable landings.
- Networks adapted to conditions, such as strong wind and different spawn positions.
- They learned behaviors like
  - smoother descent paths,
  - less fuel use,
  - and improved control.
- Performance can struggle under extreme or out of distribution scenarios.
- Evolution can reveal unexpected strategies.

## The Importance

- The results support the idea that autonomous controllers can handle tasks that require non-trivial human input.
- Stable and efficient landings support the goal of lowering the cost of spaceflight.
- The same principles apply to robotics and other areas where agents need to balance, stabilize, or navigate on their own.
- Neuroevolution is a valuable option for problems where gradients are hard to define or rewards are sparse.

## Surprises

- Some networks discovered creative, unexpected solutions like hovering before landing, tipping onto a the edge of a platform, or performing a fast drop then burn at the last second.
- Small changes in the initialization or conditions sometimes led to very different landing styles.
- Networks often found strategies that were simpler or more efficient than anticipated. The resulting networks with high fitness often only had one or two hidden nodes. Of course the simulation was relatively simple, but still. 

## Conclusion

Overall, this project showed that NEAT can evolve effective control policy networks for landing a rocket in a simplified simulation. In addition to learning how to land the rocket, the evolved neural networks adapted to new conditions and developed distinct and interesting behaviors. In the future, I hope to increase the simulation’s fidelity, test more environmental conditions, and evolve deeper networks. I also would like to compare reinforcement learning and other techniques to neuroevolution. Finally, I want to explore integrating neurosymbolic AI into NEAT to augment mutation, crossover, and the evolution process. I want to enable faster, more precise, and more interpretable convergence.
