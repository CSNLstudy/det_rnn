# Alternation training of decision and estimation
import sys, pickle, copy, cmath
import scipy.stats as stats
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mc

sys.path.append('../')
from det_rnn import *
import det_rnn.train as dt
import det_rnn.analysis as da

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def mean(angles, deg=True):
    '''Circular mean of angle data(default to degree)
    '''
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    mean = cmath.phase(angles_complex.sum()) % (2 * np.pi)
    return round(np.rad2deg(mean) if deg else mean, 7)

def corrcoef(x, y, deg=True, test=False):
    '''Circular correlation coefficient of two angle data(default to degree)
    Set `test=True` to perform a significance test.
    '''
    convert = np.pi / 180.0 if deg else 1
    sx = np.frompyfunc(np.sin, 1, 1)((x - mean(x, deg)) * convert)
    sy = np.frompyfunc(np.sin, 1, 1)((y - mean(y, deg)) * convert)
    r = (sx * sy).sum() / np.sqrt((sx ** 2).sum() * (sy ** 2).sum())

    if test:
        l20, l02, l22 = (sx ** 2).sum(),(sy ** 2).sum(), ((sx ** 2) * (sy ** 2)).sum()
        test_stat = r * np.sqrt(l20 * l02 / l22)
        p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        return tuple(round(v, 7) for v in (r, test_stat, p_value))
    return round(r, 7)

stim_list   = np.arange(24)
ref_list    = np.arange(-4,4+1)
gt          = np.repeat((np.arange(24)/24.*np.pi - np.pi/2.) % np.pi - np.pi/2., 10)
def error_cal(twoD):
    res_vec = np.ones((twoD.shape[0],)) * np.nan
    for i in range(len(res_vec)):
        raw_error = twoD[i,:] - np.pi/48. - gt
        cor_error = (raw_error - np.pi/2.) % (np.pi) - np.pi/2.
        res_vec[i] = np.mean(np.abs(cor_error))
    return(res_vec)

#############################
# Prediction of the output
#############################



#############################
# Activity profile of E/I neurons
#############################
Univar = np.ones((30,2,9,750)) * np.nan
for i in range(30):
    with open('/Volumes/Data_CSNL/project/RNN_study/20-10-15/HG/output/univariate/sequential' + str(i) + '.pkl', 'rb') as f:
        Univar[i,:,:,:] = pickle.load(f)

husl = mc.ListedColormap(sns.color_palette("husl",9))
with plt.style.context('fivethirtyeight'):
    fig = plt.figure(figsize=(8, 5))
    for i_r in range(9):
        plt.plot(np.arange(750), np.mean(Univar[:, 1, i_r, :], axis=0), label=str(ref_list[i_r]), color=husl.colors[i_r], linewidth=1.5)
    # plt.plot([3,11],[3,11], 'k:', linewidth=1.)
    # plt.scatter(180./np.pi*behav_stddat.STDFar, 180./np.pi*behav_stddat.STDNear, s=80, c=behav_stddat.index.values, cmap=husl)
    plt.xlabel("Time"); plt.ylabel("Spiking Rate"); plt.legend()
    plt.savefig("/Users/hyunwoogu/Dropbox/rnn_study201015/RnnInh.pdf",bbox_inches='tight',transparent=True)
    plt.show()




#############################
# Pattern similarity 
#############################
PatternMat = np.ones((30,9,75,24,24))
for i in range(30):
    with open('/Volumes/Data_CSNL/project/RNN_study/20-10-15/HG/output/pattern_similarity/sequential' + str(i) + '.pkl', 'rb') as f:
        PatternMat[i,:,:,:,:] = pickle.load(f)


#
fig, ax = plt.subplots(1,5, figsize=(18,5))
ax[0].imshow(np.mean(PatternMat[:,:,10,:,:], axis=(0,1))); ax[0].set_title("ITI")
ax[1].imshow(np.mean(PatternMat[:,:,25,:,:], axis=(0,1))); ax[1].set_title("Stimulus")
ax[2].imshow(np.mean(PatternMat[:,:,35,:,:], axis=(0,1))); ax[2].set_title("PreDecision")
ax[3].imshow(np.mean(PatternMat[:,:,50,:,:], axis=(0,1))); ax[3].set_title("Decision")
ax[4].imshow(np.mean(PatternMat[:,:,65,:,:], axis=(0,1))); ax[4].set_title("Estimation")
plt.tight_layout(pad=2.0); 
plt.savefig("/Users/hyunwoogu/Dropbox/rnn_study201015/RnnPattern.pdf",bbox_inches='tight',transparent=True)
plt.show()


fig, ax = plt.subplots(1,5, figsize=(18,5))
ax[0].imshow(np.mean(PatternMat[:,0,25,:,:], axis=0), vmin=.6, vmax=0.8); ax[0].set_title("-4")
ax[1].imshow(np.mean(PatternMat[:,3,25,:,:], axis=0), vmin=.6, vmax=0.8); ax[1].set_title("-1")
ax[2].imshow(np.mean(PatternMat[:,4,25,:,:], axis=0), vmin=.6, vmax=0.8); ax[2].set_title("0")
ax[3].imshow(np.mean(PatternMat[:,5,25,:,:], axis=0), vmin=.6, vmax=0.8); ax[3].set_title("+1")
ax[4].imshow(np.mean(PatternMat[:,8,25,:,:], axis=0), vmin=.6, vmax=0.8); ax[4].set_title("+4")
plt.tight_layout(pad=2.0); plt.show()


# #
# plt.imshow(np.mean(PatternMat[]))
# plt.show()
# PatternMat.shape

#############################
# Decision-congruent biases
#############################
behav_dict   = {}
behav_DF     = pd.DataFrame({})
behav_stdmat = np.ones((30,24,9)) * np.nan
behav_iqrmat = np.ones((30,24,9)) * np.nan
for i in range(30):
    with open('/Volumes/Data_CSNL/project/RNN_study/20-10-15/HG/output/decision_standard/analyses/behavior/sequential'+str(i)+'.pkl', 'rb') as f:
        behav_dict[str(i)] = pickle.load(f)
    behav_DF = behav_DF.append(behav_dict[str(i)])
    for i_s, s_t in enumerate(stim_list):
        for i_r, ref in enumerate(ref_list):
            behav_iqrmat[i,i_s,i_r] = stats.iqr(behav_dict[str(i)].corrError[(behav_dict[str(i)].reference_ori_total == ref) & (behav_dict[str(i)].stimulus_ori_total == s_t)])
            behav_stdmat[i, i_s, i_r] = np.std(behav_dict[str(i)].corrError[(behav_dict[str(i)].reference_ori_total == ref) & (behav_dict[str(i)].stimulus_ori_total == s_t)])
        print(i, s_t)

behav_prep = behav_DF.assign(choice = (1 - behav_DF['choice']))
behav_prep = behav_prep.assign(choice = (1 - behav_prep['choice']))
behav_prep = behav_prep[np.isin(behav_prep['reference_ori_total'],[-4., -1., 0., 1., 4.])]
behav_prep = behav_prep.assign(lapse  = ((behav_prep['reference_ori_total'] == -4.) & (behav_prep['choice'] == 0.)) | \
                               ((behav_prep['reference_ori_total'] == 4.) & (behav_prep['choice'] == 1.)))
behav_prep = behav_prep.assign(corrErrorDeg = behav_prep['corrError']*180./np.pi)

# sns.boxplot(x='reference_ori_total', y='corrError', hue='choice', data=behav_DF, palette="Set1", showfliers=False)
# plt.xlabel("Reference"); plt.ylabel("Error")
# plt.show()

Set1Par = sns.color_palette("Set1")
with plt.style.context('fivethirtyeight'):
    fig = plt.figure(figsize=(8, 5))
    g = sns.violinplot(x="reference_ori_total", y="corrErrorDeg", hue="choice", data=behav_prep[~behav_prep['lapse']], palette=Set1Par[:2][::-1], linewidth=1., showfliers = False, bw=.005)
    g.legend(loc='center left', bbox_to_anchor=(1., 0.5), ncol=1)
    plt.xlabel("Reference Level"); plt.ylabel("Error(Deg)")
    plt.ylim([-60,60])
    plt.savefig("/Users/hyunwoogu/Dropbox/rnn_study201015/RnnViolin.pdf",bbox_inches='tight',transparent=True)
    plt.show()
    
#############################
# Psychometric curve
#############################
behav_prep = behav_prep.assign(neg_ref = -behav_prep['reference_ori_total'])
with plt.style.context('fivethirtyeight'):
    fig = plt.figure(figsize=(8, 5))
    g = sns.regplot(x='neg_ref',y='choice', \
            data=behav_prep[~behav_prep['lapse']].iloc[np.random.randint(behav_prep[~behav_prep['lapse']].shape[0], size=1000),:],logistic=True)
    plt.xlabel("Negative Reference Level"); plt.ylabel("Choice")
    # plt.ylim([-60,60])
    plt.savefig("/Users/hyunwoogu/Dropbox/rnn_study201015/RnnPsychometric.pdf",bbox_inches='tight',transparent=True)
    plt.show()
    
#############################
# Decision-induced variability
#############################
husl = mc.ListedColormap(sns.color_palette("husl",30))
with plt.style.context('fivethirtyeight'):
    fig = plt.figure(figsize=(8, 5))
    for i in range(30):
        plt.plot(np.arange(-4,4+1), 180./np.pi*np.mean(behav_iqrmat[i,:,:],axis=0), color=husl.colors[i])
    # plt.plot([3,11],[3,11], 'k:', linewidth=1.)
    # plt.scatter(180./np.pi*behav_stddat.STDFar, 180./np.pi*behav_stddat.STDNear, s=80, c=behav_stddat.index.values, cmap=husl)
    plt.xlabel("Reference Level"); plt.ylabel("IQR(Deg)")
    plt.savefig("/Users/hyunwoogu/Dropbox/rnn_study201015/RnnDIVcurve.pdf",bbox_inches='tight',transparent=True)
    plt.show()


behav_stddat = pd.DataFrame({'STDNear':np.mean(behav_stdmat[:,:,np.array([3,5])],axis=(1,2)),
                             'STDFar' :np.mean(behav_stdmat[:,:,np.array([0,8])],axis=(1,2)),})
husl = mc.ListedColormap(sns.color_palette("husl",30))
with plt.style.context('fivethirtyeight'):
    fig = plt.figure(figsize=(8, 8))
    plt.plot([3,11],[3,11], 'k:', linewidth=1.)
    plt.scatter(180./np.pi*behav_stddat.STDFar, 180./np.pi*behav_stddat.STDNear, s=80, c=behav_stddat.index.values, cmap=husl)
    plt.xlabel("STD(Far)"); plt.ylabel("STD(Near)")
    plt.savefig("/Users/hyunwoogu/Dropbox/rnn_study201015/RnnDIV.pdf",bbox_inches='tight',transparent=True)
    plt.show()


with plt.style.context('fivethirtyeight'):
    fig = plt.figure(figsize=(8, 5))
    g = sns.regplot(x='neg_ref',y='choice', \
            data=behav_prep[~behav_prep['lapse']].iloc[np.random.randint(behav_prep[~behav_prep['lapse']].shape[0], size=1000),:],logistic=True)
    plt.xlabel("Negative Reference Level"); plt.ylabel("Choice")
    # plt.ylim([-60,60])
    plt.savefig("/Users/hyunwoogu/Dropbox/rnn_study201015/RnnPsychometric.pdf",bbox_inches='tight',transparent=True)
    plt.show()
    
# let us rely on R...
# behav_prep[~behav_prep['lapse']].iloc[np.random.randint(behav_prep[~behav_prep['lapse']].shape[0], size=10000),:].to_csv("/Users/hyunwoogu/Dropbox/rnn_study201015/RNNbehav.csv")


#############################
# Decoding performance
#############################
decode_dict    = {}
decode_errmat  = np.ones((30,75,9)) * np.nan
decode_corrmat = np.ones((30,75,9)) * np.nan
for i in range(30):
    with open('/Volumes/Data_CSNL/project/RNN_study/20-10-15/HG/output/gen_decoded/sequential' + str(i) + '.pkl', 'rb') as f:
        decode_dict[str(i)] = pickle.load(f)
    for i_r in range(9):
        decode_errmat[i,:,i_r] = error_cal(decode_dict[str(i)][i_r, :, :])
        for i_t in range(75):
            decode_corrmat[i, i_t, i_r] = corrcoef(np.repeat(np.arange(24), 10) / 24. * np.pi,
                                           decode_dict[str(i)][i_r, i_t, :] % np.pi, deg=False, test=False)

    print(i, i_r)


decode_errdat  = pd.DataFrame({'model':[], 'time':[], 'perf':[], 'ref':[]})
decode_corrdat = pd.DataFrame({'model':[], 'time':[], 'perf':[], 'ref':[]})
for i in range(30):
    for i_r in range(9):
        decode_errdat  = decode_errdat.append(pd.DataFrame({'model':np.repeat(i,75), 
                                                           'time':np.arange(75), 
                                                           'perf':-decode_errmat[i,:,i_r], 
                                                           'ref':np.repeat(ref_list[i_r],75)}))
        decode_corrdat = decode_errdat.append(pd.DataFrame({'model':np.repeat(i,75), 
                                                           'time':np.arange(75), 
                                                           'perf':decode_corrmat[i,:,i_r], 
                                                           'ref':np.repeat(ref_list[i_r],75)}))
        
decode_errdat = decode_errdat.assign(NF = np.isin(decode_errdat['ref'], [-4,-1,1,4])*1 + 1*(np.abs(decode_errdat['ref'] < 2)) )
decode_corrdat= decode_corrdat.assign(NF = np.isin(decode_corrdat['ref'], [-4,-1,1,4])*1 + 1*(np.abs(decode_corrdat['ref'] < 2)) )

##
Set1Par = sns.color_palette("Set1")
with plt.style.context('fivethirtyeight'):
    fig = plt.figure(figsize=(8, 5))
    g = sns.lineplot(x='time',y='perf',hue='NF', data=decode_errdat[decode_errdat['NF']>0], palette="Set2", linewidth = 2.)
    g.legend(loc='center left', bbox_to_anchor=(1., 0.5), ncol=1)
    plt.xlabel("Time"); plt.ylabel("Performance(-AbsoluteError)")
    # plt.ylim([-60,60])
    plt.savefig("/Users/hyunwoogu/Dropbox/rnn_study201015/RnnDecodePerf.pdf",bbox_inches='tight',transparent=True)
    plt.show()
    
husl = mc.ListedColormap(sns.color_palette("husl",9))
with plt.style.context('fivethirtyeight'):
    fig = plt.figure(figsize=(8, 5))
    for i_r in range(9):
        plt.plot(-np.mean(decode_errmat[:,:,i_r],axis=0), label=str(ref_list[i_r]), color=husl.colors[i_r])
    # plt.plot([3,11],[3,11], 'k:', linewidth=1.)
    # plt.scatter(180./np.pi*behav_stddat.STDFar, 180./np.pi*behav_stddat.STDNear, s=80, c=behav_stddat.index.values, cmap=husl)
    plt.xlabel("Time"); plt.ylabel("Performance(-AbsoluteError)"); plt.legend()
    plt.savefig("/Users/hyunwoogu/Dropbox/rnn_study201015/RnnDecodePerfRef.pdf",bbox_inches='tight',transparent=True)
    plt.show()



sns.lineplot(x='time',y='perf',hue='NF', data=decode_corrdat[decode_corrdat['NF']>0], palette="Set1")

for i_r in range(9):
    plt.plot(-np.mean(decode_errmat[:,:,i_r],axis=0), label=str(ref_list[i_r]))
plt.legend()
plt.show()


decode_corrmat
for i_r in range(9):
    plt.plot(np.mean(decode_corrmat[:,:,i_r],axis=0), label=str(ref_list[i_r]))
plt.legend()
plt.show()



##
for i_r in range(9):
    plt.plot(stim_perf[i_r,:], label=str(ref_list[i_r]))
plt.legend()
plt.show()






