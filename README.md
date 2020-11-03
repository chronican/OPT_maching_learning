# Constrained optimization for deep learning

| Student's name | SCIPER |
| -------------- | ------ |
| Wei Jiang | 313794  |
| Xiaoqi Ma | 308932  |
| Shuo Wen  | 307251 |

## Project Description
The objective of this project is to explore different constrained algorithms --
Frank Wolfe(FWGD) and projetion(PGD) algorithms and optimize their hyper-parameters. The experiment gets the insight of the effect of constraint shape(LP norm), constraint size(kappa),
number of hidden units in network, and step size and learning rate in FWGD and PSGD respectively. The code is run in Python 3.7 and mainly built by Pytorch.



### Folders and Files
- `result_log`: contrains the logging of the result for the data in the report
- `constraints.py`: contains the FWGD and PGD algorithm to constrain parameter update.
- `model.py`: contains four-layer MLP model
- `until.py`: contains utility function for calculating projection
- `dataset.py`: generates function to generating MNIST dataset
- `train.py`: contains functions for training and evaluating
- `requirements.txt`: contrains package used for this project
- `run.ipynb`: contains the result of our experiment.


## Getting Started
- Run `pip install -r requirements.txt` in bash to install the package for the project.
- Run `run.ipynb` to reproduce the result of our experiment on SGD, FWGD and PGD.
- The logging of the result will automatically resave into the `result_log` folder if rerunning the code for generating the result in `run.ipynb`.
