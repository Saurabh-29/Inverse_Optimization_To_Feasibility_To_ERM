# Revgrad
The original codebase was repurposed to write this code.  The codebase consists of Two
Main folder,  Algorithm and train. 

I implemented the train folder with all the algorithms (Ours(revgrad), MOM, ST, BB and optnet). 

To run this code, there are three main steps. 
Installation: Install the python libraries of torch, numpy, scipy, wandb, hydra and gurobi. Hydra is the config manager and gurobi is LP solver.

Data-generation: To generate data, run the generate_dataset.py with the required config file. Default config is in config/configDataGen.yaml. Parameters can be overwritten at runtime from terminal. 

Example: python generate_dataset.py dataGen.N_train=6 dataGen.name=temp_6_deg1 dataGen.degree=1 with these three parameters changed from the default configs. 


Running the code: To run the code, go to train folder and run the train.py, by default, it will choose the default config (../config/config.yaml) which one can add his own or change some parameters from command-line. 

Example: python train.py method=revgrad debug=1 data_path=../temp_6_deg1.pkl optimizer=armijo, runs the revgrad(our method) for the temp_6_deg1 dataset with armijo optimizer. 

To run GD with constant lr 0.1: use optimizer=sgd optimizer.lr=0.1  

Other methods are: naivesol, redcost, identity, blackbox. debug=1 doesn't logs the values on wandb for that run. 

Traning files are stored in train/ folder with each method has its own method.py file. Common utils are stored in utils.py file. To run a training with method with adagrad optimizer with given lr, we can use:

```
python train.py data_path=../dataset/dataset_1_replaced.pkl  optimizer=adagrad optimizer.lr=1.0 method=optnet
```

More setting are defined in the config file, which can be changed at runtime, by default, it will run armijo line search. To run the code for stochastic setting, just change the batch-size.

Default model used is a linear layer, for running resnet, use model_params=sp_resnet from config/model_params/sp_resnet. The code for model is in train/models.py. Data_utils is in train/data_utils.py