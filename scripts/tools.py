import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ParticleGravity():
    def __init__(self) -> None:
        pass

    def initialize(self, x0: np.ndarray, dx0: np.ndarray, 
        m: np.ndarray, g: np.ndarray, r0: np.ndarray = None) -> None:
        # x0, dx0: [time, point, dimension]
        if x0.shape != dx0.shape:
            raise ValueError("'x0' and 'dx0' must be same shape")
        if x0.ndim == 2:
            x0 = x0[None,:]
            dx0 = dx0[None,:]
            
        self.np, self.ndim = x0.shape[1::]
        self.ndf = self.np * self.ndim # degree of freedom
        self.x, self.dx, self.m, self.g, self.r0 = x0, dx0, m, g, r0
        self.t = np.array([])
        self.mij = scipy.linalg.toeplitz(self.m) # cross-mass matrix

    def solve(self, t: np.ndarray) -> None:
        y0 = np.concatenate([self.x[-1].flatten(),self.dx[-1].flatten()])
        p = (self.mij,self.g,self.ndf,self.ndim,self.r0)
        self.solution = scipy.integrate.odeint(self.system, y0, t, args = p)
        self.t = np.concatenate([self.t,t])
        r = (-1,self.np,self.ndim)
        x = (self.solution[:,:self.ndf]).reshape(r)
        dx = (self.solution[:,self.ndf:]).reshape(r)
        self.x = np.concatenate([self.x,x])
        self.dx = np.concatenate([self.dx,dx])
        
    def system(self, x, t, *p):
        mij, g, ndf, nd, r0 = p
        r = np.reshape(x[:ndf], (-1, nd)) # unknown variable array
        dr = x[ndf:] # first derivative array
        # cross-difference matrix
        rd = np.subtract.outer(r,r)
        rd = np.array([rd[:,i,:,i] for i in range(nd)])
        # cross-distance matrix
        rij = np.linalg.norm(rd, ord = 2, axis = 0)
        rij = rij + np.eye(rij.shape[0]) # to avoid zero division
        # restriction potential
        if r0 is not None:
            rij[rij<r0] = -rij[rij<r0]
        # second derivative matrix
        ddr = np.sum(-mij*g*rd/np.pow(rij,3), axis = -1)
        y = np.concatenate([dr,(ddr.T).flatten()])
        return y

    def load(self, filename: str):
        data = np.load(filename)
        self.__dict__.update(data)
        self.np, self.ndim = self.x.shape[1::]

    def save(self, filename: str):
        data = dict(t = self.t, x = self.x, dx = self.dx, m = self.m, g = self.g)
        np.savez(filename, **data)
        
    def transform(self, k):
        x = np.moveaxis(self.x,(0,1),(1,0))
        match k:
            case "absolute":
                pax = dict(xlabel = 'x', ylabel = 'y')
            case "center":
                xc = np.mean(x,axis=0)
                x = x-xc[None,:]
                pax = dict(xlabel = 'x-xc', ylabel = 'y-yc')
            case _:
                if type(k) is int:
                    x = x-x[k]
                    pax = dict(xlabel = f"x-x{k}", ylabel = f"y-y{k}")
                else:
                    pax = dict(xlabel = 'x', ylabel = 'y')
        return x, pax

    def show(self, filename = None, transform = 'absolute', pax = {}):
        
        x, paxt = self.transform(transform)
        pax['aspect'] = 'equal'
        pax = {**pax, **paxt}
        fig, ax = plt.subplots()
        ax.set(**pax)
        [ax.plot(xi[:,0], xi[:,1], label = str(i), marker = '.') for i, xi in enumerate(x)]
        ax.legend(title = 'particle')
        ax.grid()
        if filename is not None:
            fig.save(filename)

    def animate(self, filename = None, dn = 1, interval = 50, 
        transform = 'absolute', pax = {}):

        x, paxt = self.transform(transform)
        pax['aspect'] = 'equal'
        pax = {**pax, **paxt}
        fig, ax = plt.subplots()
        def update(j):
            pax['title'] = f"t[{j}]={self.t[j]:.2f}"
            ax.cla()
            ax.set(**pax)
            p = [ax.plot(xi[:j,0], xi[:j,1], label = str(i), marker = '.') for i, xi in enumerate(x)]
            ax.legend(title = 'particle')
            ax.grid()
            return p
        ani = FuncAnimation(fig, update, frames = np.arange(0,x.shape[1]-1,dn), interval = interval)
        if filename is not None:
            ani.save(filename)
