# `det_rnn`

+ A workbench that unifies stimulus-output generation process
+ RNN project group @CSNL, Seoul National University


---

### Version 1.1 updated

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
