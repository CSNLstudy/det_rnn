# `det_rnn`: A **recurrent neural network** model for the **delayed estimation task**

Here, a task-optimized biologically plausible recurrent neural network(RNN) model simulates human behaviors. 

+ Project led by RNN project group @[Cognitive and Systems Neuroscience Lab(CSNL)](https://www.snu-csnl.com/), Seoul National University
+ The codes are written in `Python 3.8.0`, mainly based on [`Masse *et al.*, 2019, Nature Neuroscience`](https://github.com/nmasse/Short-term-plasticity-RNN). 
+ Requires >`Tensorflow 2.x.x`


---
Basic structure of `det_rnn` has been set as follows.

```
base/ 
	_stimulus.py
	_parameters.py
	functions.py

train/
	trainer.py
	hyper.py
	model.py

analysis/
	inspect_behav.py
```

+ The core part is `base/`, which is dedicated to generating stimuli(as intended for `det_rnn`). 
+ The other parts are for training a model(which is inherited from `tf.module`)

Please check `examples/` for some examples for training and improving RNNs.
