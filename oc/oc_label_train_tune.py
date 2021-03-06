'''
Model takes saved labels from other scripts
Runs the end model on it  

'''
import sys
import os
sys.path.append('../')
sys.path.append('../metal')
sys.path.append('../heart-MRI-pytorch')
sys.path.append('../data')

import pickle
import numpy as np
import argparse
import torch
import time
import logging
import warnings
import pandas
from glob import glob
from scipy.sparse import csr_matrix
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from dataloader_4ch import UKBB_LAX_Roll
from models.frame.densenet_av import densenet_40_12_bc
from utils import *
from frame_encoder import FrameEncoderBAV, FrameEncoderOC

from metal.label_model import LabelModel
from metal.label_model.baselines import MajorityLabelVoter
from metal.end_model import EndModel
from metal.contrib.modules import Encoder, LSTMModule
from metal.analysis import lf_summary, confusion_matrix
from metal.tuners.random_tuner import RandomSearchTuner
from metal.logging import LogWriter
from metal.logging.tensorboard import TensorBoardWriter
from sampler import ImbalancedDatasetSampler

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def read_labels(label_list):
	L = []
	for index in range(len(label_list)):
		L.append(np.load(label_list[index]))

	L = np.squeeze(np.array(L))
	
	# reshaping array from (PID,frames,) -> (PID*frames,)
	m = L.shape[0]
	n = L.shape[1]
	if(len(L.shape) == 2): # true labels 
		L = np.reshape(L,(m*n,))
		L = 2 - L # changing from 0-indexing to 1-indexing - used to be L = L + 1
	else:
		L = csr_matrix(np.reshape(L,(m*n,L.shape[2])))

	return L


def csv2list(csv_file, root_dir, cat_dir, pid_str = ''):
	df = pd.read_csv(f"{root_dir}/{csv_file}")
	pids = list(df.ID)
	paths = [f"{root_dir}/{cat_dir}/{pid}{pid_str}.npy" for pid in pids]
	return paths


def load_labels(args):
	#path = os.getcwd()
	L = {}
	Y = {}

	#L["train"] = read_labels(glob(args.train + '/lf_labels/*.npy'))
	L['train'] = read_labels(csv2list(args.train_csv, args.train, "lf_labels"))
	L['dev'] = read_labels(csv2list(args.dev_csv, args.dev, "lf_labels"))
	L['test'] = read_labels(csv2list(args.test_csv, args.test, "lf_labels"))
	
	Y["dev"] = read_labels(csv2list(args.dev_csv, args.dev, "true_labels","_labels"))
	Y["test"] = read_labels(csv2list(args.test_csv, args.test, "true_labels","_labels"))   

	return L,Y

def load_dataset(args,Ytrain,Ydev,Ytest):
	'''
	Loading LAX 4ch data
	'''
	DataSet = UKBB_LAX_Roll
	train = DataSet(args.train, Ytrain, seed=args.data_seed, mask = args.mask)
	#dev = DataSet(args.dev, np.expand_dims(Ydev,1), seed=args.data_seed)
	dev = DataSet(args.dev, Ydev, seed=args.data_seed, mask = args.mask)

	if args.test:
		#test = DataSet(args.test, np.expand_dims(Ytest,1), seed=args.data_seed)
		test = DataSet(args.test, Ytest, seed=args.data_seed, mask = args.mask)
	else:
		test = None

	return train, dev, test


# dataloader 
def get_data_loader(train, dev, test=None, batch_size=4, num_workers=1):
	
	data_loader = {}
	data_loader["train"] = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	#train_loader = DataLoader(train,sampler=ImbalancedDatasetSampler(train), batch_size=batch_size, num_workers=num_workers)
	data_loader["dev"]   = DataLoader(dev, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	data_loader["test"]  = None if not test else DataLoader(test, batch_size=batch_size,shuffle=False, num_workers=num_workers)

	return data_loader



def train_model(args):

    #global args
    #args = parser.parse_args()

	hidden_size =128 
	num_classes = 2
	encode_dim = 1000 # using get_frm_output_size()

	L,Y = load_labels(args) 

	Ytrain_p = np.load(args.train_labels)	

	# End Model
	# Create datasets and dataloaders
	train, dev, test = load_dataset(args, Ytrain_p, Y["dev"], Y["test"])
	data_loader = get_data_loader(train, dev, test, args.batch_size, args.num_workers)
	#print(len(data_loader["train"])) # 18850 / batch_size
	#print(len(data_loader["dev"])) # 1500 / batch_size
	#print(len(data_loader["test"])) # 1000 / batch_size 

	# Define input encoder
	cnn_encoder = FrameEncoderOC

	if(torch.cuda.is_available()):
		device = 'cuda'
	else:
		device = 'cpu'

	# Define LSTM module
	lstm_module = LSTMModule(
		encode_dim,
		hidden_size,
		bidirectional=False,
		verbose=False,
		lstm_reduction=args.lstm_reduction,
		encoder_class=cnn_encoder,
		encoder_kwargs = {"requires_grad":args.requires_grad}
		)

	train_args = [data_loader["train"]]

	train_kwargs = {
	'seed':args.seed,
	'progress_bar':True,
	'log_train_every':1}

	init_args = [
	[hidden_size, num_classes]
	]

	init_kwargs = {
	"input_module": lstm_module, 
	"optimizer": "adam",
	"verbose": False,
	"input_batchnorm": True,
	"use_cuda":torch.cuda.is_available(),
	'checkpoint_dir':args.checkpoint_dir,
	'seed':args.seed,
	'device':device}
	
	search_space = {
	'n_epochs':[10],
	'batchnorm':[True],
	'dropout': [0.1,0.25,0.4],
	'lr':{'range': [1e-3, 1e-2], 'scale': 'log'}, 
	'l2':{'range': [1e-5, 1e-4], 'scale': 'log'},#[ 1.21*1e-5],
	#'checkpoint_metric':['f1'],
	}	
	
	log_config = {
	"log_dir": args.checkpoint_dir,
	"run_name": 'cnn_lstm_oc'
	}

	max_search = 5
	tuner_config = {"max_search": max_search }

	validation_metric = 'accuracy'

	# Set up logger and searcher
	tuner = RandomSearchTuner(
	EndModel, 
	**log_config,
	log_writer_class=TensorBoardWriter,
	validation_metric=validation_metric,
	seed=1701 )
	
	disc_model = tuner.search(
	search_space,
	valid_data = data_loader["dev"],
	train_args=train_args,
	init_args=init_args,
	init_kwargs=init_kwargs,
	train_kwargs=train_kwargs,
	max_search=tuner_config["max_search"],
	clean_up=False,
	)

	# evaluate end model
	print('Dev Set Performance')
	disc_model.score(data_loader["dev"], verbose=True, metric=['accuracy','precision', 'recall', 'f1'])
	print('Test Set Performance')
	disc_model.score(data_loader["test"], verbose=True, metric=['accuracy','precision', 'recall', 'f1'])


if __name__ == "__main__":
	# Checking to see if cuda is available for GPU use
	cuda = torch.cuda.is_available()

	# Parsing command line arguments
	argparser = argparse.ArgumentParser(description="Loading LAX 4Ch data")

	argparser.add_argument("--train", type=str, default=None, help="training set")
	argparser.add_argument("--dev", type=str, default=None, help="dev (validation) set")
	argparser.add_argument("--test", type=str, default=None, help="test set")

	argparser.add_argument("--train_csv", type=str, default="pid_list.csv", help="training set PID csv file")
	argparser.add_argument("--dev_csv", type=str, default="pid_list.csv", help="dev (validation) set PID csv file")
	argparser.add_argument("--test_csv", type=str, default="pid_list.csv", help="test set PID csv file")

	#argparser.add_argument("-c", "--config", type=str, default=None, help="load model config JSON")
	argparser.add_argument("--num_workers",type=int,default=8,help = "number of workers")
	argparser.add_argument("-B", "--batch_size", type=int, default=4, help="batch size")
	argparser.add_argument("--quiet", action="store_true", help="suppress logging")
	argparser.add_argument("-H", "--host_device", type=str, default="gpu", help="Host device (GPU|CPU)")
	argparser.add_argument("--data_seed", type=int, default=123, help="random sample seed")

	argparser.add_argument("--lr","--learning-rate",default=0.001,type=float,help="initial learning rate")
	argparser.add_argument("--momentum", default=0.9, type=float, help="momentum")
	argparser.add_argument("--weight-decay","--wd",default=1e-4,type=float,help="weight decay (default: 1e-4)")
	argparser.add_argument("-E", "--n_epochs", type=int, default=1, help="number of training epochs")
	argparser.add_argument("--seed",type=int,default=123,help="random seed for initialisation")
	argparser.add_argument("--lstm_reduction",type=str,default="attention",help="LSTM reduction at output layer")

	argparser.add_argument("--mask",type=bool,default=False,help="Selects whether to use segmented data")
	argparser.add_argument("--checkpoint_dir", type=str, default="oc_checkpoints_all/oc_checkpoint_label", help="dir to save checkpoints")

	argparser.add_argument("--requires_grad", type=bool, default=False, help="Selects whether to freeze or finetune frame encoder")
	argparser.add_argument("--train_labels", type=str, default="../data/temporal_labels/ours_400.npy", help="Path to labels for training")

	args = argparser.parse_args()

	if not args.quiet:
		logging.basicConfig(format='%(message)s', stream=sys.stdout, level=logging.INFO)

		if not torch.cuda.is_available() and args.host_device.lower() == 'gpu':
			logger.error("Warning! CUDA not available, defaulting to CPU")
			args.host_device = "cpu"

			if torch.cuda.is_available():
				logger.info("CUDA PyTorch Backends")
				logger.info("torch.backends.cudnn.deterministic={}".format(torch.backends.cudnn.deterministic))

	# print summary of this run
	logger.info("python " + " ".join(sys.argv))
	print_key_pairs(args.__dict__.items(), title="Command Line Args")

	train_model(args)


