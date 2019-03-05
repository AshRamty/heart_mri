# Heart MRI 
## Data
The data can be downloaded from dawn6 or dawn11. The path to the data is "/lfs/1/heartmri/". Copy the three folders "train32","dev32" and "test32".

## Software
The scripts use the following repositories. Please download them and add them to the path: 
- metal: https://github.com/HazyResearch/metal
- heart-MRI-pytorch: https://github.com/HazyResearch/heart-MRI/tree/pytorch

## Executing scripts
The path to the train, dev and test sets need to be specified as command line arguments. For example, to execute the script `cnn_lstm_1.py` from the scripts folder use: `python cnn_lstm_1.py --train /lfs/1/heartmri/train32 --dev /lfs/1/heartmri/dev32 --test /lfs/1/heartmri/test32`.
