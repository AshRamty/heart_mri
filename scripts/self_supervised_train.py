'''
Runs a self-supervised network on open / close data

'''

import os
import sys
sys.path.append('../metal')
sys.path.append('../heart-MRI-pytorch')
sys.path.append('../data')

import numpy as np
import argparse
import pickle
import torch
import pandas
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from metal.end_model import EndModel
from metal.contrib.modules import Encoder, LSTMModule

from dataloader_4ch import UKBB_LAX_SelfSupervised
from frame_encoder import FrameEncoderOC
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

def load_dataset(args):
	'''
	Loading LAX 4ch data
	'''
	DataSet = UKBB_LAX_SelfSupervised
	train = DataSet(args.train, seed=args.data_seed, mask = args.mask)
	#dev = DataSet(args.dev, np.expand_dims(Ydev,1), seed=args.data_seed)
	dev = DataSet(args.dev, seed=args.data_seed, mask = args.mask)

	if args.test:
		#test = DataSet(args.test, np.expand_dims(Ytest,1), seed=args.data_seed)
		test = DataSet(args.test, seed=args.data_seed, mask = args.mask)
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

    # Create datasets and dataloaders
	train, dev, test = load_dataset(args)
	data_loader = get_data_loader(train, dev, test, args.batch_size, args.num_workers)

	hidden_size =128 
	num_classes = 2
	encode_dim = 1000 # using get_frm_output_size()

	# Define input encoder
	cnn_encoder = FrameEncoderOC

	if(torch.cuda.is_available()):
		device = 'cuda'
	else:
		device = 'cpu'
	#import ipdb; ipdb.set_trace()

	# Define LSTM module
	lstm_module = LSTMModule(
		encode_dim,
		hidden_size,
		bidirectional=False,
		verbose=False,
		lstm_reduction=args.lstm_reduction,
		encoder_class=cnn_encoder,
		)

	init_kwargs = {
	"layer_out_dims":[hidden_size, num_classes],
	"input_module": lstm_module, 
	"optimizer": "adam",
	"verbose": False,
	"input_batchnorm": False,
	"use_cuda":cuda,
	'seed':args.seed,
	'device':device}

	end_model = EndModel(**init_kwargs)
	
	if not os.path.exists(args.checkpoint_dir):
		os.mkdir(args.checkpoint_dir)
	
	with open(args.checkpoint_dir+'/init_kwargs.pickle', "wb") as f:
		pickle.dump(init_kwargs,f,protocol=pickle.HIGHEST_PROTOCOL)

	dropout = 0.4
	# Train end model
	end_model.train_model(
		train_data=data_loader["train"],
		valid_data=data_loader["dev"],
		l2=args.weight_decay,
		lr=args.lr,
		n_epochs=args.n_epochs,
		log_train_every=1,
		verbose=True,
		progress_bar = True,
		#loss_weights = [0.55,0.45],
		batchnorm = False,
		input_dropout = 0.1,
		middle_dropout = dropout,
		checkpoint_dir = args.checkpoint_dir,
		log_valid_metrics = ['accuracy', 'f1'],
		checkpoint_metric='f1',
		)

	end_model.score(data_loader["dev"], verbose=True, metric=['accuracy', 'precision', 'recall', 'f1','roc-auc'])





if __name__ == "__main__":
	# Checking to see if cuda is available for GPU use
	cuda = torch.cuda.is_available()

	# Parsing command line arguments
	argparser = argparse.ArgumentParser(description="Loading LAX 4Ch data")

	argparser.add_argument("--train", type=str, default=None, help="training set")
	argparser.add_argument("--dev", type=str, default=None, help="dev (validation) set")
	argparser.add_argument("--test", type=str, default=None, help="test set")

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

	argparser.add_argument("--mask",type=str,default=False,help="Selects whether to use segmented data")
	argparser.add_argument("--checkpoint_dir", type=str, default="oc_checkpoints", help="dir to save checkpoints")

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
