import pandas as pd
import seaborn as sns


H1 = pd.DataFrame({
    'neuron':np.repeat(np.arange(100),240),
    'ori':np.tile(np.repeat(np.arange(24),10),100),
    'resp':H_total[100,:,:].T.reshape((-1,))
    })

H1['Time'] = '1s'

H2 = pd.DataFrame({
    'neuron':np.repeat(np.arange(100),240),
    'ori':np.tile(np.repeat(np.arange(24),10),100),
    'resp':H_total[200,:,:].T.reshape((-1,))
    })

H2['Time'] = '2s'

H3 = pd.DataFrame({
    'neuron':np.repeat(np.arange(100),240),
    'ori':np.tile(np.repeat(np.arange(24),10),100),
    'resp':H_total[250,:,:].T.reshape((-1,))
    })

H3['Time'] = '2.5s'

H4 = pd.DataFrame({
    'neuron':np.repeat(np.arange(100),240),
    'ori':np.tile(np.repeat(np.arange(24),10),100),
    'resp':H_total[301,:,:].T.reshape((-1,))
    })

H4['Time'] = '3.1s'


H5 = pd.DataFrame({
    'neuron':np.repeat(np.arange(100),240),
    'ori':np.tile(np.repeat(np.arange(24),10),100),
    'resp':H_total[500,:,:].T.reshape((-1,))
    })

H5['Time'] = '5s'

Hagg = H1.append(H2).append(H3).append(H4).append(H5)

plt.figure(figsize=(10,6))
sns.lineplot(x='ori',y='resp', hue='Time', data=Hagg[np.isin(Hagg['neuron'],argE[20:])], palette='Set3')
plt.xticks(np.arange(24,step=2), labels=np.arange(0, 180+7.5, 15))
plt.xlabel('Orientation')
plt.show()


plt.figure(figsize=(10,6))
sns.lineplot(x='ori',y='resp', hue='neuron', data=Hagg[(Hagg['Time'] == '5s') & np.isin(Hagg['neuron'],argI)], legend=False, palette='Set3')
plt.xticks(np.arange(24,step=2), labels=np.arange(0, 180+7.5, 15))
plt.xlabel('Orientation')
plt.show()


sns.lineplot(x='ori',y='resp', hue='neuron', data=Hagg[(Hagg['Time'] == '2s') & np.isin(Hagg['neuron'],argI)], legend=False, palette='Set3')


for i_neuron in argE[:20]:
    plt.plot(H_total[200,:,i_neuron])
plt.show()


for i_neuron in argE[20:]:
    plt.plot(H_total[200,:,i_neuron])
plt.show()

for i_neuron in argI:
    plt.plot(H_total[200,:,i_neuron])
plt.show()


plt.plot(np.mean(H_total[200,:,:],axis=-1))


for i_neuron in argI:
    plt.plot(H_total[200,:,i_neuron])
plt.show()



## 
neuron_label = np.ones((100,)) * np.nan
maxfire_label  = np.ones((100,)) * np.nan
meanfire_label = np.ones((100,)) * np.nan
stdfire_label  = np.ones((100,)) * np.nan

for i_neuron in range(100):
    neuron_label[i_neuron] = np.argmax([np.mean(H_total[200,(10 * i):(10 * (i+1)),i_neuron]) for i in range(24)])
    maxfire_label[i_neuron]  = np.max([np.mean(H_total[200,(10 * i):(10 * (i+1)),i_neuron]) for i in range(24)])

for i_neuron in range(100):
    meanfire_label[i_neuron] = np.mean(H_total[200,(10 * int(neuron_label[i_neuron])):(10 * (int(neuron_label[i_neuron])+1)),i_neuron])
    stdfire_label[i_neuron]  = np.std(H_total[200,(10 * int(neuron_label[i_neuron])):(10 * (int(neuron_label[i_neuron])+1)),i_neuron])

fano_label = np.power(stdfire_label,2)/(meanfire_label+np.finfo(float).eps)


plt.figure(figsize=(10,6))
plt.hist(neuron_label[argE[40:]], bins=24)
plt.hist(neuron_label[argI], bins=24)
plt.xticks(np.arange(24,step=2), labels=np.arange(0, 180+7.5, 15))
plt.xlabel('Orientation')
plt.show()

plt.show()

plt.hist(firing_label[argE[30:]])
plt.hist(firing_label[argI])

plt.hist(meanfire_label[argE[30:]])
plt.xlim([0,18])

plt.hist(meanfire_label[argI])
plt.xlim([0,18])

plt.hist(stdfire_label[argE[30:]])
plt.hist(stdfire_label[argI])

plt.hist(fano_label[argE[30:]])
plt.hist(fano_label[argI])



plt.scatter()

## 

### Wout_discrim 
plt.imshow(np.maximum(load_model.var_dict['w_in'].numpy(),0))
plt.show()

plt.plot(np.mean(np.maximum(load_model.var_dict['w_in'].numpy(),0),0))


InputDrive = np.mean(np.maximum(load_model.var_dict['w_in'].numpy(),0),0)
InputMax   = np.argmax(np.maximum(load_model.var_dict['w_in'].numpy(),0), axis=0)

plt.scatter(InputMax[argE[:30]], neuron_label[argE[:30]])
plt.scatter(InputMax[argE[30:]], neuron_label[argE[30:]])
plt.scatter(InputMax[argI], neuron_label[argI])


##

np.where(InputDrive == 0)

InputDrive[82]


argE = np.argsort(InputDrive)[(np.arange(100)<70)[np.argsort(InputDrive)]]
argI = np.argsort(InputDrive)[(np.arange(100)>=70)[np.argsort(InputDrive)]]

np.sort(InputDrive)





##

