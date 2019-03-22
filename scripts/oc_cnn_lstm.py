'''
Classifies mitral valve open / close using CNN LSTM in Metal
Does not include hyperparameter tuning

'''
import sys
import os
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
		L = L+1 # changing from 0-indexing to 1-indexing
	else:
		L = csr_matrix(np.reshape(L,(m*n,L.shape[2])))

	return L


def load_labels(args):
	#path = os.getcwd()
	L = {}
	Y = {}

	#train_lf_list = glob(args.train + '/lf_labels/*.npy') 
	L["train"] = read_labels(glob(args.train + '/lf_labels/*.npy'))
	L["dev"] = read_labels(glob(args.dev + '/lf_labels/*.npy'))
	L["test"] = read_labels(glob(args.test + '/lf_labels/*.npy'))

	#import ipdb; ipdb.set_trace()
	Y["dev"] = read_labels(glob(args.dev + '/true_labels/*.npy'))
	Y["test"] = read_labels(glob(args.test + '/true_labels/*.npy'))	

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

	# Label Model
	# labelling functions analysis
	print(lf_summary(L["dev"], Y = Y["dev"]))

	# training label model
	label_model = LabelModel(k=num_classes, seed=123)
	label_model.train_model(L["train"], Y["dev"], n_epochs = 500, log_train_every = 50)

	# evaluating label model
	print('Trained Label Model Metrics:')
	label_model.score((L["dev"], Y["dev"]), metric=['accuracy','precision', 'recall', 'f1'])

	# comparison with majority vote of LFs
	mv = MajorityLabelVoter(seed=123)
	print('Majority Label Voter Metrics:')
	mv.score((L["dev"], Y["dev"]), metric=['accuracy','precision', 'recall', 'f1'])

	Ytrain_p = label_model.predict_proba(L["train"])
	#print(Ytrain_ps.shape) #(377*50,2)
	#Ydev_p = label_model.predict_proba(L["dev"])

	# test models
	#label_model.score((Ltest,Ytest), metric=['accuracy','precision', 'recall', 'f1'])

	# End Model
	# Create datasets and dataloaders
	train, dev, test = load_dataset(args, Ytrain_p, Y["dev"], Y["test"])
	data_loader = get_data_loader(train, dev, test, args.batch_size, args.num_workers)
	#print(len(data_loader["train"])) # 18850 / batch_size
	#print(len(data_loader["dev"])) # 1500 / batch_size
	#print(len(data_loader["test"])) # 1000 / batch_size 
	#import ipdb; ipdb.set_trace()

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

	'''
	# Define end model
	end_model = EndModel(
		input_module=lstm_module,
		layer_out_dims=[hidden_size, num_classes],
		optimizer="adam",
		#use_cuda=cuda,
		batchnorm=False,
		seed=args.seed,
		verbose=False,
		device = device,
		)
	'''

	init_kwargs = {
	"layer_out_dims":[hidden_size, num_classes],
	"input_module": lstm_module, 
	"optimizer": "adam",
	"verbose": False,
	"input_batchnorm": False,
	"use_cuda":cuda,
	'seed':args.seed,
	'device':device}

	model = EndModel(**init_kwargs)

	os.mkdir(args.checkpoint_dir)
	with open(args.checkpoint_dir+'/init_kwargs.pkl', "wb") as f:
		pickle.dump(f,init_kwargs)

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
		loss_weights = [0.55,0.45],
		batchnorm = False,
		input_dropout = 0.1,
		middle_dropout = dropout,
		checkpoint_dir = args.checkpoint_dir,
		#validation_metric='f1',
		)

	# evaluate end model
	end_model.score(data_loader["dev"], verbose=True, metric=['accuracy','precision', 'recall', 'f1'])
	#end_model.score((Xtest,Ytest), verbose=True, metric=['accuracy','precision', 'recall', 'f1'])
	


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


