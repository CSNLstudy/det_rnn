# `det_rnn`: A **recurrent neural network** model for the **delayed estimation task**

**`det_rnn`** is a task-optimized biologically plausible recurrent neural network(RNN) model that simulates human behaviors. It is develped by a project group at [Cognitive and Systems Neuroscience Lab(CSNL)](https://www.snu-csnl.com/), Seoul National University. Written in `Python 3.8.0`, it is mainly based on [`Masse *et al.*, 2019, Nature Neuroscience`](https://github.com/nmasse/Short-term-plasticity-RNN). 

---
### Features

- **RNN-based emulation of human working memory task** that involves both decision-making and working memory 
- **Key human behavioral features** are emergent from the model
  - **Decision-congruent bias**: reproduction error patterns consistent to the self-made decisions
  - **Decision-induced variability**: an apparent suboptimality where subjects' reproduction errors are more variable after the decision-making with "near" reference.

---
### How to run

`Tensorflow 2.x.x` required. First, make sure you specify the saving directory `save_dir` at 27th line of `main.py`. Then, at the console run 

```
python main.py
```

It will start to train a model that performs a task jointly composed of decision-making and estimation. The resulting model will be saved in `save_dir` as you specified.


---
### Source structure 
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
