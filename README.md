# From Inverse Optimization to Feasibility to ERM
This codebase contains implementation for all the methods and experiments in the paper (Ours, MOM, ST, BB and QPTL). 

To run this code, there are three main steps. 

## Installation: 
Install the conda environment from the environment.yaml file.
```
conda env create --file=environment.yaml
```

## Data-generation

To generate synthetic data, run the Data/generate_dataset.py with the required config file. Default config is in config/configDataGen.yaml. Parameters can be overwritten at runtime from terminal. 


```
cd Data
python synthetic_generator.py dataGen=sp
```

This generates shortest path synthetic data with default parameters in config/dataGen/sp.yaml file. To generate knapsack data, run the same script with dataGen=knapsack.

To generate the warcraft shortest path, mnist perfect matching, first download the dataset from this [repository](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers).


After downloading the dataset, run the corresponding generator file to generate the dataset in the form required by the code.

## Running the method 

To run the code for our method for a dataset with adagrad optimizer and lr=1.0

```
python train.py data_path=<path to data>  optimizer=adagrad optimizer.lr=1.0 method=revgrad
```

To run other method, replace revgrad with other method.