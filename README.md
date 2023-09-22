# BranchAndBoundBisim

To run this, first create a conda environment using

`conda env create -f environment.yml`

Then compile the cython code using
`python setup.py build_ext --inplace --force`

You can then run the project using
`python main.py`

you can override the current config through the command line, e.g. 
`python main.py env.harden_gaps=0.0 env.num_steps=200 optimization.lr=3e-4 training_scheme.clip_coef=0.1 model.hidden_dim=512 optimization.batchsize=128 model.depth=7 model.n_layers=2`

