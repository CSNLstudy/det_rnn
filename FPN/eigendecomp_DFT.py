# Written by Hyunwoo Gu (@SNU) 21.2.22
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

# Generate von-Mises W (HL)
nunit, kE, kI = 20, 5, 0.1
x     = np.linspace(0,np.pi*(nunit-1)/nunit,nunit)
W     = np.zeros([nunit,nunit]) # weight matrix
for i in range(nunit):
    wE = np.exp(kE*np.cos(2*(x-x[i])))/(2*np.pi*scipy.special.iv(0,kE))
    wI = np.exp(kI*np.cos(2*(x-x[i])))/(2*np.pi*scipy.special.iv(0,kI))
    W[i,:] = wE - wI

# 1. numpy eigendecomposition (implemented via LAPACK routines)
evals, evecs = np.linalg.eig(W)   # since W=W^T, we have real evals
evals_sort   = np.argsort(-evals) # mysteriously, numpy does not perfectly sort eigenvalues

# 2. DFT 
w        = W[:,0]                 # target timeseries of DFT
ns       = np.arange(nunit)
X        = np.exp(-2j * np.pi * np.outer(ns,ns) / nunit, dtype=np.complex256) # DFT matrix
lams     = (X @ w).real           # note that they are "almost" real due to numerical precision
Xtilde   = np.zeros(X.shape)      # I believe there is a slighly more elegant way than this
for k in range(nunit):
    m = int(nunit/2)
    if (k < m) | ((k%2 == 0) & (k == m)) : Xtilde[:,k] = np.cos(np.arange(nunit)/nunit*np.pi*2*k)
    else: Xtilde[:,k] = np.sin(np.arange(nunit)/nunit*np.pi*2*(nunit-k))
lams_sort = np.argsort(-lams)

# Compare
numpy_eval_sort = evals[evals_sort]
numpy_evec_sort = evecs[:,evals_sort]
dft_eval_sort   = lams[lams_sort]
dft_evec_sort   = Xtilde[:,lams_sort]

fig, ax = plt.subplots(1,3, figsize=(17,5))
im0 = ax[0].plot(numpy_eval_sort,linewidth=5,label='numpy'); im0 = ax[0].plot(dft_eval_sort,linewidth=5,label='DFT'); ax[0].set_xlabel("Index(sort)"); ax[0].legend(); ax[0].set_title("Eigenvalues")
im1 = ax[1].imshow(numpy_evec_sort, aspect='auto'); ax[1].set_title("Eigenvectors, numpy"); plt.colorbar(im1, ax=ax[1])
im2 = ax[2].imshow(dft_evec_sort, aspect='auto'); ax[2].set_title("Eigenvectors, DFT"); plt.colorbar(im2, ax=ax[2])
fig.tight_layout(pad=2.0)
plt.show()
