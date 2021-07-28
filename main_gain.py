# ============================================
# Gain modulation in a circuit with reference memory
# ============================================
from det_rnn import *
import det_rnn.train as dt
import pickle
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mc

# ============================================
# Training a circuit constrained by pretrained weights 
# ============================================
## "shorter" structure
par['design'].update({'iti'  : (0, 0.3),
                      'stim' : (0.3, 0.6),                      
                      'decision': (0.9, 1.4),
                      'delay'   : ((0.6, 0.9),(1.4, 1.7)),
                      'estim' : (1.7, 2.0)})

## parameter adjustment 
par['dt']             = 20
dt.hp['dt']           = 20
dt.hp['gain']         = 0 # 1e-3  # amplitude of random initialization (makes dynamics chaotic) 
dt.hp['noise_rnn_sd'] = 0.1
dt.hp['w_rnn11_fix']  = True
dt.hp['w_rnn21_fix']  = True  
dt.hp['w_in_dm_fix']  = True  
dt.hp['w_in_em_fix']  = True  
dt.hp['w_out_dm_fix'] = True   # assume linear voting from two separate populations
dt.hp['w_out_em_fix'] = True   # assume circular voting from two separate populations
dt.hp['DtoE_off']     = False  # Connection

## Training
## WANING: note the (*0) terms in the model.py! They are just zero terms.  
dt.hp             = dt.update_hp(dt.hp)
par               = update_parameters(par)
stimulus          = Stimulus()
ti_spec           = dt.gen_ti_spec(stimulus.generate_trial())
max_iter          = 500
n_print           = 10
model             = dt.initialize_rnn(ti_spec)
model_performance = {'perf_dm': [], 'perf_em': [], 'loss': [], 'perf_loss_dm': [], 'perf_loss_em': [], 'spike_loss': []}

for iter in range(max_iter):
    trial_info        = dt.tensorize_trial(stimulus.generate_trial())
    Y, Loss           = model(trial_info, dt.hp)
    model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par)
    
    # Print
    if iter % n_print == 0: dt.print_results(model_performance, iter)

model.model_performance = dt.tensorize_model_performance(model_performance)


# ============================================
# Visualization
# ============================================
## Okay visualization with some long task structure
par['design'].update({'iti'     : (0, 0.3),
                      'stim'    : (0.3, 0.6),                      
                      'decision': (0.9, 3.0), 
                      'delay'   : ((0.6, 0.9),(3.0, 5.0)), # 0.3 to 3s
                      'estim'   : (5.0, 5.5)})
ref_dist          = np.ones(len(par['reference']))
par['ref_dist']   = ref_dist
par               = update_parameters(par)
stimulus          = Stimulus(par)
trial_info        = dt.tensorize_trial(stimulus.generate_trial())
pred_output_DM, pred_output_EM, HR, H1, H2 = model.rnn_model(trial_info['neural_input1'], trial_info['neural_input2'], dt.hp)
neural_input1 = trial_info['neural_input1'].numpy()
neural_input2 = trial_info['neural_input2'].numpy()

TEST_TRIAL = np.random.randint(par['batch_size'])
print(trial_info['stimulus_ori'][TEST_TRIAL], trial_info['reference_ori'][TEST_TRIAL])
fig, axes = plt.subplots(9,1, figsize=(9,12))
axes[0].imshow(neural_input1[:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input1")
axes[1].imshow(neural_input2[:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Neural Input2")
axes[2].imshow(trial_info['desired_decision'][:,TEST_TRIAL,:].numpy().T, aspect='auto', interpolation='none'); axes[2].set_title("Desired Decision")
axes[3].imshow(trial_info['desired_estim'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[3].set_title("Desired Estimation")
axes[4].imshow(pred_output_DM[:,TEST_TRIAL,:].numpy().T,  aspect='auto', interpolation='none'); axes[4].set_title("Predicted Decision")
axes[5].imshow(pred_output_EM[:,TEST_TRIAL,:].numpy().T,  aspect='auto', vmin=0); axes[5].set_title("Predicted Estimation")
axes[6].imshow(HR[:,TEST_TRIAL,:].numpy().T,  aspect='auto'); axes[6].set_title("HR")
axes[7].imshow(H1[:,TEST_TRIAL,:].numpy().T,  aspect='auto'); axes[7].set_title("H1")
axes[8].imshow(H2[:,TEST_TRIAL,:].numpy().T,  aspect='auto'); axes[8].set_title("H2")
fig.tight_layout(pad=2.0)
plt.show()


# ============================================
# Load "idealized" weights
# ============================================
## Networks were trained multiple times.
## Trained weights were averaged, and further averaged into a characteristic vector that defines the weights' circular properties
## The resulting weights were saved.
with open('weights.pkl', 'rb') as f: res = pickle.load(f)
HRg_T, H1g_T, H2g_T = res['HRg_T'], res['H1g_T'], res['H2g_T'] # averaged pre-decision population activities
w_rnnr1_ideal, w_rnn12_ideal = res['w_rnnr1_ideal'], res['w_rnn12_ideal']

hp = {k:dt.hp[k].numpy() for k in dt.hp if tf.is_tensor(dt.hp[k])}
hp['noise_rnn_sd'] = 0.1

stimulus     = Stimulus()
tuning_input = stimulus.tuning_input[:,0,:]
inputs       = (tuning_input @ dt.hp['w_in2'].numpy()).T
# inputs_ref   = np.tile((dt.hp['w_in1'].numpy()).T, 2).T
inputs_ref   = dt.hp['w_in1'].numpy()
alpha        = 0.2

## Important functions
def sigmoid(x): return tf.nn.sigmoid(x).numpy()
def softmax(x): return tf.nn.softmax(x).numpy()

ori_support = np.linspace(0,np.pi,24,endpoint=False)
sinr        = np.sin(2.*ori_support) # bases for making 
cosr        = np.cos(2.*ori_support)
def circular_mean(matrix, vector=False): 
    if vector:
        decode = np.arctan2(np.sum(matrix*sinr), np.sum(matrix*cosr))/2. * 24/np.pi
        if decode<0:
            decode += 24.
        elif decode>24:
            decode -= 24.
    else:
        decode = np.arctan2(matrix.T @ sinr, matrix.T @ cosr)/2. * 24/np.pi
        decode[decode<0]  += 24 
        decode[decode>24] -= 24 
    return decode

def roll_column(A,n):
    B    = np.zeros_like(A)
    nrow = B.shape[0]
    for i_row in range(nrow): 
        B[i_row, :] = np.roll(A[i_row, :],n)
    return B

## One-step transition: numpy version of _run_cell
def one_step(_HR, _H1, _H2, _wr1, _w12, shift, turn_on_ref=1, gain = 1):

    _HRp1 = (1-alpha) * _HR + alpha * sigmoid( 
        turn_on_ref*roll_column(inputs_ref,shift).T + (hp['w_rnnrr'].T @ _HR.T).T )

    _H1p1 = (1-alpha) * _H1 + alpha * sigmoid( ( hp['w_rnn11'].T  @  _H1.T).T + \
        (hp['w_rnn21'].T  @ _H2.T).T + gain * (_wr1.T @ _HR.T).T ) 

    _H2p1 = (1-alpha) * _H2 + alpha * sigmoid( (_w12.T @  _H1.T).T )

    return _HRp1, _H1p1, _H2p1


# ============================================
# Noise-free simulations of gain modulation
# ============================================
## Gain Parameters (IMPORTANT): Try different sets
gain_reference, gain_delay, gain_estimation = 10, 1.5, 3.5    # task-dependent gain (Best matches neural data!)
# gain_reference, gain_delay, gain_estimation = 10, 0, 0      # input-only gain 
# gain_reference, gain_delay, gain_estimation = 1, 1, 1       # contant gain

## Other Parameters
Nt      = 330
decodes = np.zeros((9, Nt)) * np.nan
HRTEST  = np.zeros((24, 9, Nt, 24))
H1TEST  = np.zeros((24, 9, Nt, 48))
H2TEST  = np.zeros((24, 9, Nt, 24))

for i_ref, ref in enumerate(np.arange(-4,4+1,1)):
    _hrtest = HRg_T.copy()
    _h1test = H1g_T.copy()
    _h2test = H2g_T.copy()

    for i in range(330):
        _readout = softmax(hp['w_out_em'].T @ _h2test[0,:])
        _decode  = np.arctan2(sum(_readout * sinr), sum(_readout * cosr))/2.
        decodes[i_ref, i]   = _decode
        HRTEST[:, i_ref, i, :] = _hrtest
        H1TEST[:, i_ref, i, :] = _h1test
        H2TEST[:, i_ref, i, :] = _h2test

        if i < 30: 
            _hrtest, _h1test, _h2test = one_step(_hrtest, _h1test, _h2test, w_rnnr1_ideal, w_rnn12_ideal, shift=ref, turn_on_ref = 1, gain = gain_reference)
        elif i > 240:
            _hrtest, _h1test, _h2test = one_step(_hrtest, _h1test, _h2test, w_rnnr1_ideal, w_rnn12_ideal, shift=ref, turn_on_ref = 0, gain = gain_estimation)
        else:
            _hrtest, _h1test, _h2test = one_step(_hrtest, _h1test, _h2test, w_rnnr1_ideal, w_rnn12_ideal, shift=ref, turn_on_ref = 0, gain = gain_delay)


## =====================================================
# Inspection & Analysis
## =====================================================
reds  = mc.ListedColormap(sns.color_palette("Reds",6))
blues = mc.ListedColormap(sns.color_palette("Blues",6))
gains = np.zeros(Nt)
gains[:30]    = gain_reference
gains[30:240] = gain_delay
gains[240:]   = gain_estimation

## Bias patterns 
fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize = [6,5], sharex = True)
for i in range(9):
    if i == 4: continue
    if i >  4: col=reds.colors[(i-5)+2]
    else: col=blues.colors[5-i]
    ax[0].plot(np.arange(Nt)/20, decodes[i,:] * 180/np.pi, color=col, linewidth=2)    
ax[0].set_ylabel('Bias(deg)')
ax[0].axvspan(0,  1.5, facecolor='grey', alpha=0.2)
ax[0].axvspan(12, 16.5, facecolor='grey', alpha=0.2)

ax[1].plot(np.arange(Nt)/20, gains, color='black', linewidth=3)
ax[1].set_ylim([-1,11])
ax[1].axvspan(0,  1.5, facecolor='grey', alpha=0.2)
ax[1].axvspan(12, 16.5, facecolor='grey', alpha=0.2)
ax[1].set_ylabel('Gain(g)')
ax[1].set_xlabel('Time from reference onset(sec)')
plt.show()


## Decision vector trace
fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize = [6,5], sharex=True)
ax[0].imshow(H1TEST[12,3,:,:].T, aspect='auto', cmap='gray', extent=[0,16.5,-1,1])
ax[0].set_yticklabels([])
ax[1].plot(np.arange(Nt)/20, gains, color='black', linewidth=3)
ax[1].set_ylim([-1,11])
ax[1].axvspan(0,  1.5, facecolor='grey', alpha=0.2)
ax[1].axvspan(12, 16.5, facecolor='grey', alpha=0.2)
ax[1].set_ylabel('Gain(g)')
ax[1].set_xlabel('Time from reference onset(sec)')
plt.show()