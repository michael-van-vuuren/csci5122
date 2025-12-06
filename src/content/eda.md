---
title: Data & Preparation
type: eda
math: true
---

## The Data

![](/images/dataprep/data-scene.png)

At each time step in the simulation, the rocket records the following state variables:
>- x-position
>- y-position
>- x-velocity
>- y-velocity
>- angle
>- angular velocity
>- wind sensor reading

At the same time, the rocket can take any combination of three actions:
>- apply thrust
>- rotate left
>- rotate right

These actions can be chosen individually or simultaneously. For example, the rocket could both apply thrust and rotate left at the same time.
 
The goal of the agent controlling the rocket is to learn a policy, which is a mapping from states to action probabilities, that enables it to land safely on the platform. In this case, the dataset consists of the rocket’s state variables (a 7D vector) as **features** and the action probabilities (a 3D vector) as **outputs**. Unlike traditional supervised learning, there are no ground truth labels (if the goal were instead to compare the neuroevolved policy to a reinforcement learning or expert policy, those policies would serve as the ground truth labels). Effective controllers emerge through the neuroevolution process, which explores the solution space and gradually selects for successful strategies.

## Data Cleaning

The data cleaning in this project is minimal because the dataset comes directly from the simulation. If the data were from a real world model rocket, more extensive cleaning, such as denoising sensor signals, would be needed. To improve learning for the evolving networks, the inputs are simply normalized. For instance, the x- and y-positions are divided by the screen width and height, respectively, to scale them between 0 and 1. Velocities are scaled down by 100.0 to reduce their magnitude. The angle and angular velocity are left unchanged since they are already small and expressed in radians and radians per second. Finally, the wind speed is scaled down by 20.0 to shrink its range.

```python {filename=""}
# network inputs
def get_inputs(self):
   return [
      self.x / SCREEN_W,      # -1 (x-pos)
      self.y / SCREEN_H,      # -2 (y-pos)
      self.vx / 100.0,        # -3 (x-velocity)
      self.vy / 100.0,        # -4 (y-velocity)
      self.theta,             # -5 (angle)
      self.omega,             # -6 (angular velocity)
      WIND_SPEED / 20.0       # -7 (wind sensor)
   ]
```

![](/images/dataprep/data-io.png)

## Links

