'''
Runs supervised learning on MR data
Does not include hyperparameter tuning

'''
import sys
sys.path.append('../')
sys.path.append('../metal')
sys.path.append('../../heart-MRI')

import numpy as np
import argparse
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

#import metal.contrib.modules.resnet_cifar10 as resnet
#from dataloaders.ukbb import UKBBCardiacMRI
#from models.frame.densenet_av import DenseNet3, densenet_40_12_bc
from dataloader_4ch import UKBB_LAX_MR
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

# load dataset
def load_dataset(args):
	'''
	Loading LAX 4ch data
	'''
	DataSet = UKBB_LAX_MR
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
	data_loader["train"]  = DataLoader(train,sampler=ImbalancedDatasetSampler(train), batch_size=batch_size, num_workers=num_workers)
	data_loader["dev"]   = DataLoader(dev, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	data_loader["test"]  = None if not test else DataLoader(test, batch_size=batch_size,shuffle=False, num_workers=num_workers)

	return data_loader


def train_model(args):

	# Create datasets and dataloaders
	train, dev, test = load_dataset(args)
	#print('train size:',len(train)) # 250
	#print('dev size:',len(dev)) # 250
	#print('test size:',len(test)) # 250
	# data in tuple of the form (series,label)
	# series shape (50,3,224,224)

	#import pdb; pdb.set_trace()

	data_loader = get_data_loader(train, dev, test, args.batch_size)

	hidden_size = 128 
	num_classes = 2
	encode_dim = 1000

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
		lstm_reduction="attention",
		encoder_class=cnn_encoder,
		)

	# Define end model
	end_model = EndModel(
		input_module=lstm_module,
		layer_out_dims=[hidden_size, num_classes],
		optimizer="adam",
		use_cuda=cuda,
		batchnorm=True,
		seed=args.seed,
		verbose=False,
		)
	

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
		#loss_weights = [0.9,0.1],
		batchnorm = 'False',
		log_valid_metrics = ['accuracy','f1'],
		checkpoint_metric = 'f1',
		checkpoint_dir = args.checkpoint_dir,
		#validation_metric='accuracy',
		#input_dropout = 0.1,
		middle_dropout = dropout,
		)

	print('Dev Set Performance')
	end_model.score(data_loader["dev"], verbose=True, metric=['accuracy', 'precision', 'recall', 'f1','roc-auc'])
	print('Test Set Performance')
	end_model.score(data_loader["test"], verbose=True, metric=['accuracy', 'precision', 'recall', 'f1','roc-auc'])


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
	argparser.add_argument("--mask",type=str,default=False,help="Selects whether to use segmented data")
	argparser.add_argument("--checkpoint_dir", type=str, default="mr_checkpoints/test", help="dir to save checkpoints")
	argparser.add_argument("--preprocess", type=bool, default=False, help="Selects whether to apply preprocessing (histogram equalization) to data")
	
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
