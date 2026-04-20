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

#### `scripts.FieldGravity`

It is consider field-particle interaction governed by classical scalar field evolution equation: 

$$\frac{\partial^2 \varphi}{\partial t^2}-c^2\Delta \varphi=a\rho(t,\mathbf{r})$$

where $\varphi(t,\mathbf{r})$ - scalar potential of gravity field, $c$ - speed of field spreading, $a$ - arbitrary constant of source term, $\rho(t,\mathbf{r})$ - test particle density. 

Initial condition:

$$\varphi(0,\mathbf{r})=0 \quad \frac{\partial \varphi(0,\mathbf{r})}{\partial t}=0$$

Boundary condition:

$$\varphi(t,\mathbf{r}\in \partial \Omega)=0$$

Example:
```python
import numpy as np
from scripts.tools import ParticleGravity, FieldGravity, interpolate
# define problem
label = 'example_fg_p2_d2'
n = np.array([100,100])
x = np.array(np.meshgrid(*tuple(np.linspace(-1,1,i) for i in n),indexing='ij'))
t = pg.t
u0 = np.zeros(n)
du = np.zeros(n)
D = np.zeros(n)
k = 0.1
F = np.zeros(n)
G = np.zeros(n)
H = np.zeros(np.append(n.shape[0],n))
# create source term
f = interpolate(pg.x[:-1],x,2,1e-3)
s = tuple(None for _ in range(n.shape[0])) + (np.s_[:],)
f = -1e5*np.sum(f,axis=-1)
f = np.moveaxis(f,(-1),(0))
F = f
# solve problem
fg = FieldGravity()
fg.initialize(u0,du,t[:2],k,x)
fg.solve(t,F,D,G,[H[0],H[1]])
fg.animate(f'plots/{label}.gif',clim=[-5e3,1e3],pax=dict(xlabel='x',ylabel='y'),pclb=dict(label=r'$\varphi$'))
fg.save(f'data/{label}.npz')
```

![example_fg_p2_d2](/plots/example_fg_p2_d2.gif)

## Technical section

#### Prepare python virtual environment by `poetry`
```shell
poetry init
poetry config --local virtualenvs.in-project true
poetry config --local virtualenvs.create true    
poetry source add --priority supplemental torch https://download.pytorch.org/whl/cu128
poetry add --source torch torch torchvision torchaudio
```