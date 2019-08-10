'''
Runs supervised learning on single-frame MR data
Does not include hyperparameter tuning

'''
import sys, os
sys.path.append('../')
sys.path.append('../metal')
sys.path.append('../../heart-MRI')

import numpy as np
import argparse
import importlib
import torch
import pandas
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.models as torch_models

from metal.end_model import EndModel
from metal.contrib.modules import Encoder
from metal.tuners import RandomSearchTuner
from metal.logging.tensorboard import TensorBoardWriter


from dataloader_4ch import UKBB_MR_Framewise
#from frame_encoder import FrameEncoderOC
from sampler import ImbalancedDatasetSampler

from utils import *
from metrics import *
from dataloaders import *
from transforms import *
import warnings
try:
    # for python2
    import cPickle
except ImportError:
    # for python3
    import _pickle as cPickle

# load dataset
def load_dataset(args):
    '''
    Loading LAX 4ch data
    '''
    DataSet = UKBB_MR_Framewise
    train = DataSet(args.train, args.mask, seed=args.data_seed, preprocess = args.preprocess)
    dev = DataSet(args.dev, args.mask, seed=args.data_seed, preprocess = args.preprocess)
    if args.test:
        test = DataSet(args.test, args.mask, seed=args.data_seed, preprocess = args.preprocess)
    else:
        test = None

    return train, dev, test

# dataloader
def get_data_loader(train, dev, test=None, batch_size=4, num_workers=1):
	
    data_loader = {}
    #data_loader["train"] = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    data_loader["train"]  = DataLoader(train,sampler=ImbalancedDatasetSampler(train), batch_size=batch_size, num_workers=num_workers,pin_memory=True)
    data_loader["dev"]   = DataLoader(dev, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True)
    data_loader["test"]  = None if not test else DataLoader(test, batch_size=batch_size,shuffle=False, num_workers=num_workers,pin_memory=True)

    return data_loader


def train_model(args):

    # Create datasets and dataloaders
    train, dev, test = load_dataset(args)
    print('train size:',len(train)) # 
    print('dev size:',len(dev)) # 
    print('test size:',len(test)) # 
    # data in tuple of the form (frame,label)
    # frame shape (3,224,224)
    
    config_in = importlib.import_module(args.config)
    em_config = config_in.em_config
    train_config = em_config["train_config"]
    search_space = config_in.search_space
    log_config = config_in.log_config
    tuner_config = config_in.tuner_config
    dl_config = train_config["data_loader_config"]


    #import pdb; pdb.set_trace()

    data_loader = get_data_loader(train, dev, test, dl_config["batch_size"])

    num_classes = 2
    encode_dim = 1000

	# Define input encoder - can use the same 
	#cnn_encoder = FrameEncoderOC


	# Initializing model object
    input_module = torch_models.resnet18(pretrained=em_config["pretrained"])
    last_layer_input_size = int(input_module.fc.weight.size()[1])
    input_module.fc = torch.nn.Linear(last_layer_input_size, 2)

    
    init_args = [[last_layer_input_size,2]]
    init_kwargs = {'input_module' : input_module}
    init_kwargs.update(em_config)
    max_search = tuner_config['max_search']
    metric = train_config['validation_metric']
    

    searcher = RandomSearchTuner(
        EndModel, **log_config, log_writer_class=TensorBoardWriter
    )
    end_model = searcher.search(
        search_space,
        data_loader["dev"],
        train_args=[data_loader["train"]],
        init_args=init_args,
        init_kwargs=init_kwargs,
        train_kwargs=train_config,
        max_search=max_search
    )
        
        

    end_model.score(data_loader["dev"], verbose=True, metric=['accuracy', 'precision', 'recall', 'f1','roc-auc'])

	#import ipdb; ipdb.set_trace()
	# saving dev set performance
    Y_p, Y, Y_s = end_model._get_predictions(data_loader["dev"], break_ties='random', return_probs=True)
    dev_labels = dev.labels
    Y_s_0 =list(Y_s[:,0]) ; Y_s_1 = list(Y_s[:,1]); 
    dev_ID = list(dev_labels["ID"]);dev_LABEL = list(dev_labels["LABEL"])  
    Y_p = list(Y_p); Y = list(Y); 
    Y_p.insert(0,"Y_p"),Y.insert(0,"Y"),
    Y_s_0.insert(0,"Y_s_0");Y_s_1.insert(0, "Y_s_1")
    dev_ID.insert(0,"ID"); dev_LABEL.insert(0,"LABEL")
    dev_pth = os.path.join(args.mr_result_filename, "_dev")
    np.save(dev_pth,np.column_stack((dev_ID,dev_LABEL,Y_p, Y, Y_s_0, Y_s_1)))

	# saving test set performance
    Y_p, Y, Y_s = end_model._get_predictions(data_loader["test"], break_ties='random', return_probs=True)
    test_labels = test.labels
    Y_s_0 =list(Y_s[:,0]) ; Y_s_1 = list(Y_s[:,1]); 
    test_ID = list(test_labels["ID"]);test_LABEL = list(test_labels["LABEL"])  
    Y_p = list(Y_p); Y = list(Y); 
    Y_p.insert(0,"Y_p"),Y.insert(0,"Y"),
    Y_s_0.insert(0,"Y_s_0");Y_s_1.insert(0, "Y_s_1")
    test_ID.insert(0,"ID"); test_LABEL.insert(0,"LABEL")
    test_pth = os.path.join(args.mr_result_filename, "_test")
    np.save(test_pth,np.column_stack((test_ID,test_LABEL,Y_p, Y, Y_s_0, Y_s_1)))



if __name__ == "__main__":
	# Checking to see if cuda is available for GPU use
    cuda = torch.cuda.is_available()

	# Parsing command line arguments
    argparser = argparse.ArgumentParser(description="Loading LAX 4Ch data")

    argparser.add_argument("--train", type=str, default=None, help="training set")
    argparser.add_argument("--dev", type=str, default=None, help="dev (validation) set")
    argparser.add_argument("--test", type=str, default=None, help="test set")

    argparser.add_argument(
    '--config', 
    required=True, 
    type=str,
    help='path to config dict'
    )
    
    argparser.add_argument("--mask",type=str,default=False,help="Selects whether to use segmented data")
    argparser.add_argument("--data_seed", type=int, default=123, help="random sample seed")
    argparser.add_argument("--checkpoint_dir", type=str, default="mr_checkpoints", help="dir to save checkpoints")
    argparser.add_argument("--preprocess", type=bool, default=False, help="Selects whether to apply preprocessing (histogram equalization) to data")
	
    argparser.add_argument("--mr_result_filename", type=str, default="mr_framewise_results/mr_test", help="filename to save result")
    args = argparser.parse_args()

    # print summary of this run
    logger.info("python " + " ".join(sys.argv))
    print_key_pairs(args.__dict__.items(), title="Command Line Args")

    train_model(args)
