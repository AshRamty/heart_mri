# Heart MRI 
## Data
The data can be downloaded from dawn. The path to the data is "/lfs/1/heartmri/". Copy the three folders "train32","dev32" and "test32".

## Software
The scripts use the following repositories. Please download them and add them to the path: 
- metal: https://github.com/HazyResearch/metal
- heart-MRI-pytorch: https://github.com/HazyResearch/heart-MRI/tree/pytorch

## Executing the "test_cnn_lstm.py" script
The path to the train, dev and test sets need to be specified as command line arguments. For example, to execute the script from the scripts folder use: "python test_cnn_lstm.py --train ../data/train32 --dev ../data/dev32 --test ../data/test32".
