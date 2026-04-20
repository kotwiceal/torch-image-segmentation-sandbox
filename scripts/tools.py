import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors

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


class FieldGravity:
    def __init__(self, device = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        torch.cuda.empty_cache()

    def initialize(self, u: np.ndarray, du: np.ndarray, t: np.ndarray, k: np.array, x:  np.ndarray = None):
        if u.ndim == 2:
            u = u[None,:]
        if du.ndim == 2:
            du = du[None,:]
        if x is None:
            x = np.mgrid[tuple(np.s_[0:i] for i in u.shape[1:])]
        self.u, self.du, self.x, self.t, self.k = u, du, x, t, k
        self.u = np.concatenate([self.u, (self.u[-1] + du[-1] * (self.t[-1] - self.t[-2]))[None,:]])
        self.generate()

    def mapping(self, n: np.ndarray, dtype = np.int16, flatten: bool = True):
        ndim = n.shape[0]
        k = np.mgrid[[np.s_[1:-2:-1] for _ in n]].astype(dtype)
        g = np.mgrid[[np.s_[0:i] for i in n]].astype(dtype)
        f = np.add.outer(k, g)
        f = np.array([f[tuple(np.s_[i] if np.any([0,1+ndim] == j) else np.s_[:]
            for j in np.arange(f.ndim))] for i in np.arange(ndim)])
        for i, ni in enumerate(n):
            ind = f[i] < 0 
            f[i][ind] = np.abs(f[i][ind])
            ind = f[i] > ni - 1
            f[i][ind] = ni - 1 - np.abs(f[i][ind] - ni + 1)
        a = np.ravel_multi_index(tuple(i for i in f), n, mode = 'wrap')
        m = [1,0,2] # kernel mask
        a = a[np.ix_(*tuple(m for _ in n))]
        i = a[tuple(np.s_[:] if i < ndim else np.s_[1:-1] for i in np.arange(a.ndim))]
        s = tuple(np.s_[0] for _ in n)
        s = ~np.isin(a[s],i[s])
        b = np.moveaxis(a,np.arange(ndim)+ndim,np.arange(ndim))[s]
        b = np.moveaxis(b, 0, -1)
        if flatten:
            r = tuple(3 for _ in range(ndim)) + (-1,)
            a = a.reshape(r)
            i = i.reshape(r)
        # center slice
        cs = tuple(np.s_[0] for _ in range(ndim))
        # forward step slice
        fs = [tuple(np.s_[1] if i == j else np.s_[0] for j in range(ndim)) for i in range(ndim)]
        fs = [s + (np.s_[:],) for s in fs]
        # backward step slice
        bs = [tuple(np.s_[-1] if i == j else np.s_[0] for j in range(ndim)) for i in range(ndim)]
        bs = [s + (np.s_[:],) for s in bs]
        return dict(node = (a,i,b), slice = (cs,fs,bs))

    def generate(self):
        self.ngrid = np.array(self.u.shape[1:])
        self.ndof = np.prod(self.ngrid)
        self.maps = self.mapping(self.ngrid)
        self.xg = tuple(self.x)
        self.x = np.array([x.flatten() for x in self.x])

    def laplacian(self, n: np.ndarray, x: np.ndarray, i: np.ndarray, c: list, f: list, b: list):
        # dimension of freedom
        nd = x.shape[0]
        ndf = np.prod(n)
        L = np.zeros((ndf,ndf))
        # forward step
        xsf = [x[i[f]]-x[i[c]] for x,f in zip(x,f)]
        # backward step
        xsb = [x[i[c]]-x[i[b]] for x,b in zip(x,b)]
        # assembly laplacian differential operator
        for j in range(nd):
            L[i[c],i[c]] += -1/xsf[j]/xsb[j]
            L[i[c],i[f[j]]] += 1/xsf[j]/(xsf[j]+xsb[j])
            L[i[c],i[b[j]]] += 1/xsb[j]/(xsf[j]+xsb[j])
        return L

    def assembly(self, env, *args, **kwags):
        c, f, ndf, nd, L, i, b, xsf, k, u, tf, tb, F, D, G, H = args
        # assembly linear system
        A = env.zeros((ndf,ndf), **kwags)
        B = env.zeros(ndf, **kwags)
        # source tern
        B[i[c]] += F[i[c]]
        # laplacian term
        A += -k*L
        # second time derivative term 
        A[i[c],i[c]] += 1/tf/(tf+tb)
        B[i[c]] += u[1][i[c]]/tf/tb - u[0][i[c]]/tb/(tf+tb)
        # dirichlet boundary condition term
        A[b[c],b[c]] += 1
        B[b[c]] += D[b[c]]
        # neumann boundary condition term
        B[b[c]] += G[b[c]]
        for j in range(nd):
            A[b[c],b[c]] += -H[j][b[c]]/xsf[j]
            A[b[c],b[f[j]]] += H[j][b[c]]/xsf[j]
        return A, B

    def solve(self, t, F: np.ndarray, D: np.ndarray, G: np.ndarray, H: list[np.ndarray]):

        env = torch
        kwgs = dict(dtype = env.float64, device = "cuda")

        nt = t.shape[0]
        n = self.ngrid
        nd = n.shape[0]
        ndf = np.prod(n)

        P = (F, D, G) + tuple(H)
        P = tuple(x[None,:] if x.shape == tuple(n) else x for x in P)
        r = [(x.shape[0],-1) if x.shape[1:] == tuple(n) else (x.shape[0],nd,-1) for x in P]
        P = tuple(x.reshape(r) for x,r in zip(P,r))
        
        n = list(self.ngrid)
        x, u = self.x, self.u
        k = self.k
    
        _, i, b = self.maps['node']
        c, f, bs = self.maps['slice']
        # define laplacian
        L = self.laplacian(n,x,i,c,f,bs)
        # forward step of boundary nodes
        xsf = np.array([x[b[f]]-x[b[c]] for x,f in zip(x,f)])

        W = (ndf, nd, L, i, b, xsf, k)

        # convert to tensor
        if env is torch:
            *W, t, u = tuple(env.tensor(x).to(self.device) for x in W + (t, u))
            W = tuple(W)
            P = tuple(env.tensor(x).to(self.device) for x in P)

        # solver iteration
        for j in range(2,nt):
            # self.t = np.append(self.t,t[j])
            # forward time step
            tf = t[j] - t[j-1]
            # backward time step
            tb = t[j-1] - t[j-2]
            # slice previous solution
            un = (u[-2:]).reshape((2,-1))
            Fi, Di, Gi, *Hi = tuple(x[0] if x.shape[0] == 1 else x[j] for x in P)
            # assembly linear system
            args = (un, tf, tb, Fi, Di, Gi, Hi)
            A, B = self.assembly(env, *((c, f) + W + args), **kwgs)
            # solve linear system
            s = torch.linalg.solve(A, B)
            # stack solution
            u = torch.cat((u, s.reshape(n)[None,:]))
        
        if u.device.type == "cuda":
            u = u.cpu()
            u = u.numpy()
        self.u = np.concatenate((self.u,u))
        self.t = np.append(self.t,t.cpu().numpy())

    def load(self, filename: str):
        data = np.load(filename)
        self.__dict__.update(data)
        self.generate()

    def save(self, filename: str):
        data = dict(t = self.t, x = self.x, u = self.u)
        np.savez(filename, **data)

    def plot(self, fig, ax, *x, clb = None, pax = {}, ppcm = {}, pclb = {}):
        ax.cla()
        ax.set(**pax)
        pcm = ax.pcolormesh(*x, **ppcm)
        if clb is None:
            clb = fig.colorbar(pcm, ax = ax, **pclb)
        else:
            clb.update_normal(pcm)
        return pcm, clb

    def show(self, i: int, clim = [None, None], norm = colors.Normalize, 
        pfig = {}, pax = {}, ppcm = {}, pclb = {}) -> None:
        u = self.u[i]
        pax['aspect'] = 'equal'
        pax['title'] = f"t[${i}]=${self.t[i]:.2f}"
        ppcm['norm'] = norm(vmin=clim[0],vmax=clim[1])
        x = self.xg + (u,)
        fig, ax = plt.subplots(**pfig)
        pcm, clb = self.plot(fig, ax, *x, pax = pax, ppcm = ppcm, pclb = pclb)
        return fig, ax, pcm, clb

    def animate(self, filename = None, dn = 1, interval = 50, 
        clim = [None, None], norm = colors.Normalize, 
        pfig = {}, pax = {}, ppcm = {}, pclb = {}):

        f = np.arange(0,self.u.shape[0],dn)
        u = self.u[f]

        fig, ax, _, clb = self.show(f[0], clim = clim, norm = norm, 
            pfig = pfig, pax = pax, ppcm = ppcm, pclb = pclb)
        
        def func(i):
            x = self.xg + (u[i],)
            pax['title'] = f"t[${i}]=${self.t[i]:.2f}"
            ppcm['norm'] = norm(vmin=clim[0],vmax=clim[1])
            pcm, _ = self.plot(fig, ax, *x, clb = clb, 
                pax = pax, ppcm = ppcm, pclb = pclb)
            return pcm

        ani = FuncAnimation(fig, func, frames = f, interval = interval)
        if filename is not None:
            ani.save(filename)


def interpolate(x: np.ndarray, X: np.ndarray, k: int = 3, sigma: float = 1e-2):
    n = X.shape[1:]
    m = x.shape[:2]
    nd = X.shape[0]
    p = X.reshape((nd,-1))
    tree = scipy.spatial.KDTree(p.T)
    Y = np.zeros(n+m)
    for i, xi in enumerate(x):
        for j, xj in enumerate(xi):
            _, ind = tree.query(xj, k = 1)
            y = np.zeros(n)
            ind = np.unravel_index(ind, n)
            y[ind] = 1
            s = tuple(np.s_[x-k:x+k] for x in ind)
            xg = X[(np.s_[:],)+s]
            s = (np.s_[:],) + tuple(None for _ in range(nd))
            xk = np.exp(-np.sum(np.pow(xg-xj[s],2),axis=0)/sigma)
            xk = xk/np.max(xk)
            y = scipy.signal.convolve(y,xk,mode='same')
            Y[tuple(np.s_[:] for _ in range(nd))+(i,j)] = y
    return Y
