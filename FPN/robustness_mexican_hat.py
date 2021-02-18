import os, sys, scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group # for generating orthonormal fixed points
from fixed_point_network import LinearFixedPointNetwork # linear fixed-point network

# ============================
# Defining W, inputs
# ============================
## Generate von-Mises W (HL)
nunit, nt, noise, kE, kI, nori, mu, n_trial = 20, 500, 0.1, 5, 0.1, 18, np.pi, 128
x     = np.linspace(0,np.pi*(nunit-1)/nunit,nunit)
w     = np.zeros([nunit,nunit]) # weight matrix
for i in range(nunit):
    wE = np.exp(kE*np.cos(2*(x-x[i])))/(2*np.pi*scipy.special.iv(0,kE))
    wI = np.exp(kI*np.cos(2*(x-x[i])))/(2*np.pi*scipy.special.iv(0,kI))
    w[i,:] = wE - wI
    
## Generate von-Mises input (HL)
oris  = np.linspace(0,np.pi-(np.pi/nori),nori)
X     = np.zeros([nunit,nori]) # input matrix
for i,ori in enumerate(oris):
    wE = 2*np.exp(kE*np.cos(2*(x-ori)))/(2*np.pi*scipy.special.iv(0,kE))
    X[:,i] = wE

# ============================
# Defining Networks
# ============================
## 1. HL version
lams, xis      = np.linalg.eig(w)
identity_W     = np.eye(nunit) # input representation to the network is the same with the input itself
lams_corrected = lams.copy()
lams_corrected[lams_corrected>1.] = 1.
fpn_HL         = LinearFixedPointNetwork(input_mat=X, encoding_dim=nunit, eigenvalues=lams_corrected)
fpn_HL.fit(xis, Win=identity_W)

## 2. Random matrix version
Q          = ortho_group.rvs(xis.shape[0]) # arbitrary orthonormal eigenvectors 

### 2.1. HL's lambdas (only a few of them are 1s)
fpn_Random_SomeOne = LinearFixedPointNetwork(input_mat=X, encoding_dim=nunit, eigenvalues=lams_corrected)
fpn_Random_SomeOne.fit(Q, Win=identity_W)

### 2.2. all-one lambdas (all of them are 1s)
fpn_Random_AllOne = LinearFixedPointNetwork(input_mat=X, encoding_dim=nunit) # if not specified, all ones used
fpn_Random_AllOne.fit(Q, Win=identity_W)

# ============================
# Simulations
# ============================
## Generate neural inputs
neural_input = np.random.normal(scale=noise, size=[nt,n_trial,nunit])
for i_t in range(n_trial):
    ori = np.random.randint(nori)
    inp = 2*np.exp(kE*np.cos(2*(x-ori)))/(2*np.pi*scipy.special.iv(0,kE))    
    neural_input[5:10,i_t,:] = inp

# Prediction
pred_HL             = fpn_HL.predict(neural_input, apply_Wout=False)
pred_Random_SomeOne = fpn_Random_SomeOne.predict(neural_input, apply_Wout=False)
pred_Random_AllOne  = fpn_Random_AllOne.predict(neural_input, apply_Wout=False)

# Figure: weight matrices
fig, ax = plt.subplots(1,3, figsize=(17,5))
im0 = ax[0].imshow(fpn_HL.Wrec, aspect='auto'); ax[0].set_title("HL: Mexican-Hat Weight"); plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(fpn_Random_SomeOne.Wrec, aspect='auto'); ax[1].set_title("Random: with few 1-eigenvalues"); plt.colorbar(im1, ax=ax[1])
im2 = ax[2].imshow(fpn_Random_AllOne.Wrec, aspect='auto'); ax[2].set_title("Random: with full 1-eigenvalues"); plt.colorbar(im2, ax=ax[2])
fig.tight_layout(pad=2.0)
plt.show()

# Figure: Predictions
TEST_TRIAL = np.random.randint(n_trial)
fig, ax = plt.subplots(2,3, figsize=(24,8))

im00 = ax[0,0].imshow(neural_input[:,TEST_TRIAL,:][:,2:].T, aspect='auto'); ax[0,0].set_title("Neural Input"); plt.colorbar(im00, ax=ax[0,0])
im10 = ax[1,0].imshow(pred_HL[:,:,TEST_TRIAL], aspect='auto'); ax[1,0].set_title("Network Output of HL: Mexican-Hat Weight"); plt.colorbar(im10, ax=ax[1,0]); ax[1,0].set_xlabel("Time(a.u.)")

im01 = ax[0,1].imshow(neural_input[:,TEST_TRIAL,:][:,2:].T, aspect='auto'); ax[0,1].set_title("Neural Input"); plt.colorbar(im01, ax=ax[0,1])
im11 = ax[1,1].imshow(pred_Random_SomeOne[:,:,TEST_TRIAL], aspect='auto'); ax[1,1].set_title("Network Output of Random: with few 1-eigenvalues"); plt.colorbar(im11, ax=ax[1,1]); ax[1,1].set_xlabel("Time(a.u.)")

im02 = ax[0,2].imshow(neural_input[:,TEST_TRIAL,:][:,2:].T, aspect='auto'); ax[0,2].set_title("Neural Input"); plt.colorbar(im02, ax=ax[0,2])
im12 = ax[1,2].imshow(pred_Random_AllOne[:,:,TEST_TRIAL], aspect='auto'); ax[1,2].set_title("Network Output of Random: with full 1-eigenvalues"); plt.colorbar(im12, ax=ax[1,2]); ax[1,2].set_xlabel("Time(a.u.)")

fig.tight_layout(pad=2.0)
plt.show()


