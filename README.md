# Heart MRI 
## Data
The data for all the experiments can be found in the `/dfs/scratch0/ashwinir/heart_mri/data/` folder on dawn. 

## Software
The scripts use the following repositories. Please download them and add them to the path: 
- metal: https://github.com/HazyResearch/metal
- heart-MRI-pytorch: https://github.com/HazyResearch/heart-MRI/tree/pytorch

The specifications to build the conda environment is provided in the `env_deps.txt` file

## Executing scripts
The path to the train, dev and test sets need to be specified as command line arguments. For example, to execute the script `oc_train.py` from the scripts folder use: `python oc_train.py --train ../data/open_close_200/train --dev ../data/open_close_200/dev --test ../data/open_close_200/test`.
