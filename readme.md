## Description
The project is considered as pet project and it goal consists in creating dataset and train a convolution neural network to segment particle position in Euler coordinate system.

## Overview

#### `scripts.ParticleGravity`
Class provides functional to assembly [Cauchy problem](https://en.wikipedia.org/wiki/Cauchy_problem), solve, visualize and save solution.

It is consider particle-particle interaction governed by [classical Newton gravitation law](https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation):

$$\frac{d^2 \mathbf{r_i}}{dt^2}=-\sum_{i\neq j}{g \cdot m_j \frac{\mathbf{r_i}-\mathbf{r_j}}{|\mathbf{r_i}-\mathbf{r_j}|^3}}, \quad i = \overline{1,..,N}$$

where $\mathbf{r_i}$ - radius vector of $i$ particle, $g$ - gravitation coefficient, $m_i$, - mass of $i$ particle, $N$ - number of test particles. Computation performed by built-in `ode45` algorithm of `scipy` library.

Example:
```python
import numpy as np
from scripts.tools import ParticleGravity
# define problem
label = 'example_p2_2d'
x = np.array([[-0.5,0],[0.5,0]])
dx = np.array([[0,0.3],[0,-0.3]])
m = np.array([1,1])
g = np.array([1])
t = np.linspace(0,10,500)
r0 = 1e-1
# solve problem 
pg = ParticleGravity()
pg.initialize(x,dx,m,g,r0)
pg.solve(t)
pg.animate(f'plots/{label}.gif',10,pax=dict(xlim=[-1,1],ylim=[-1,1]))
pg.save(f'data/{label}.npz')
```
![example_p2_2d](/plots/example_p2_2d.gif)

## Technical section

#### Prepare python virtual environment by `poetry`
```shell
poetry init
poetry config --local virtualenvs.in-project true
poetry config --local virtualenvs.create true    
poetry source add --priority supplemental torch https://download.pytorch.org/whl/cu128
poetry add --source torch torch torchvision torchaudio
```