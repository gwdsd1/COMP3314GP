"""DATER Figure 3 (Yan et al., CVPR 2005), with disclosed numeric choices.

Full identity initialization; sequential modewise generalized eigenproblems.
Columns have unit Euclidean length and canonical signs. No QR, ridge or
shrinkage. Positive definite within scatter is required rather than silently
changing the algorithm. The generalized-eigen update follows Figure 3; it
does not establish a global maximum of the joint tensor trace ratio.
"""
import numpy as np
from scipy.linalg import eigh

from csa import mode_product


def canonical_columns(u):
    u=u/np.linalg.norm(u,axis=0)
    peaks=np.argmax(np.abs(u),axis=0)
    signs=np.sign(u[peaks,np.arange(u.shape[1])])
    return u*np.where(signs==0,1,signs)


def mode_scatter(projected,labels,mode):
    # Each unfolded column is a cluster, not a new independent identity.
    z=np.moveaxis(projected,mode+1,1).reshape(len(projected),projected.shape[mode+1],-1)
    mean=z.mean(0);sw=np.zeros((z.shape[1],z.shape[1]));sb=np.zeros_like(sw)
    for label in np.unique(labels):
        samples=z[labels==label];cm=samples.mean(0)
        delta=samples-cm;diff=cm-mean
        sw+=np.einsum('nac,nbc->ab',delta,delta,optimize=True)
        sb+=len(samples)*(diff@diff.T)
    return (sb+sb.T)/2,(sw+sw.T)/2


class DATER:
    def __init__(self,ranks,*,max_iter=20,epsilon=.001):
        self.ranks=tuple(ranks);self.max_iter=max_iter;self.epsilon=epsilon

    def fit(self,x,labels):
        x=np.asarray(x,dtype=float);labels=np.asarray(labels)
        if x.ndim<2 or len(labels)!=len(x) or not np.isfinite(x).all():
            raise ValueError('Expected finite tensors and one gallery label per tensor')
        self.shape=x.shape[1:]
        if len(self.shape)!=len(self.ranks) or any(not 1<=r<=m for r,m in zip(self.ranks,self.shape)):
            raise ValueError('Invalid mode ranks')
        if self.max_iter<3 or self.epsilon<=0 or len(np.unique(labels))<2:
            raise ValueError('Invalid iteration settings or class labels')
        # Figure 3 uses I_m, not the truncated rectangular CSA starting basis.
        self.bases=[np.eye(m) for m in self.shape]
        self.history=[];self.updates=[];self.stop_reason='max_iter'
        for iteration in range(1,self.max_iter+1):
            old=[u.copy() for u in self.bases];mode_changes=[]
            for k,rank in enumerate(self.ranks):
                projected=x
                for j,u in enumerate(self.bases):
                    if j!=k:projected=mode_product(projected,u.T,j)
                sb,sw=mode_scatter(projected,labels,k)
                sw_values=eigh(sw,eigvals_only=True)
                tolerance=max(sw.shape)*np.finfo(float).eps*sw_values[-1]
                if sw_values[0]<=tolerance:
                    raise ValueError(f'DATER Sw numerically singular at iteration{iteration},mode{k}; no fallback authorized')
                eigenvalues,u=eigh(sb,sw)
                eigenvalues=eigenvalues[-rank:][::-1];u=canonical_columns(u[:,-rank:][:,::-1])
                self.bases[k]=u
                residual=float(np.linalg.norm(sb@u-(sw@u)*eigenvalues)/max(1,np.linalg.norm(sb@u)))
                if not np.isfinite(u).all() or residual>1e-8:
                    raise ValueError('Unreliable generalized eigenvectors')
                change=None if old[k].shape!=u.shape else float(np.linalg.norm(u-old[k])/(self.shape[k]*rank))
                mode_changes.append(change)
                self.updates.append(dict(iteration=iteration,mode=k,sb=sb,sw=sw,basis=u.copy(),eigenvalues=eigenvalues,
                    sw_condition=float(sw_values[-1]/sw_values[0]),relative_residual=residual))
            core=self.transform(x).reshape(len(x),-1);mean=core.mean(0)
            within=between=0.
            for label in np.unique(labels):
                samples=core[labels==label];cm=samples.mean(0)
                within+=float(np.sum((samples-cm)**2));between+=len(samples)*float(np.sum((cm-mean)**2))
            self.history.append(dict(iteration=iteration,normalized_basis_changes=mode_changes,
                between_energy=between,within_energy=within,DTC_ratio=between/within))
            if iteration>2 and all(c<self.epsilon for c in mode_changes):
                self.stop_reason='paper_basis_change';break
        return self

    def transform(self,x):
        x=np.asarray(x,dtype=float)
        if x.shape[1:]!=self.shape:raise ValueError('Shape differs from gallery learning input')
        for k,u in enumerate(self.bases):x=mode_product(x,u.T,k)
        return x
