'''
Classifies mitral valve open / close using CNN LSTM in Metal

'''
import sys
import os
sys.path.append('../metal')
sys.path.append('../heart-MRI-pytorch')
sys.path.append('../data')

import numpy as np
import argparse
import torch
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

from metal.label_model import LabelModel
from metal.label_model.baselines import MajorityLabelVoter
from metal.end_model import EndModel
from metal.contrib.modules import Encoder, LSTMModule
from metal.analysis import lf_summary, confusion_matrix
from metal.tuners.random_tuner import RandomSearchTuner


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


def train_model(args):

    #global args
    #args = parser.parse_args()

	hidden_size = 128 
	num_classes = 2
	encode_dim = 108 # using get_frm_output_size()

	L,Y = load_labels(args) 

	# Label Model
	# labelling functions analysis
	print(lf_summary(L["dev"], Y = Y["dev"]))

	# majority vote of LFs
	mv = MajorityLabelVoter(seed=123)
	print('Majority Label Voter Metrics:')
	mv.score((L["dev"], Y["dev"]), metric=['accuracy','precision', 'recall', 'f1'])


	# training label model
	#label_model = LabelModel(k=num_classes, seed=123)
	#label_model.train_model(L["train"], Y["dev"], n_epochs = 500, log_train_every = 50)

	# evaluating label model
	#print('Trained Label Model Metrics:')
	#label_model.score((L["dev"], Y["dev"]), metric=['accuracy','precision', 'recall', 'f1'])

	print('Performing Hyperparameter Search:')
	train_args = [L["train"], Y["dev"]]
	train_kwargs = {}
	init_args = [ num_classes ]
	init_kwargs = {
	"optimizer": "sgd",
	#"input_batchnorm": True,
	#"use_cuda":torch.cuda.is_available(),
	'seed':123}
	
	search_space = {
	'seed' : [123],
	'n_epochs': [500],
	'learn_class_balance':[False,True],
	'lr': {'range': [1e-2, 1e-1], 'scale': 'log'},
	'momentum': {'range': [0.7, 0.95], 'scale': 'log'},
	#'l2':{'range': [1e-5, 1e-3], 'scale': 'log'},
	'log_train_every': [50],
	#'validation_metric': 'accuracy',
	}
	
	log_config = {
	"log_dir": "./run_logs", 
	"run_name": 'oc_label_model'
	}

	max_search = 25
	tuner_config = {"max_search": max_search }

	validation_metric = 'accuracy'

	# Set up logger and searcher
	tuner = RandomSearchTuner(	LabelModel, 
								#**log_config, 
								#log_writer_class=TensorBoardWriter, 
								validation_metric=validation_metric,
								seed=1701)
	
	disc_model = tuner.search(
	search_space,
	valid_data = (L["dev"], Y["dev"]),
	train_args=train_args,
	init_args=init_args,
	init_kwargs=init_kwargs,
	train_kwargs=train_kwargs,
	max_search=tuner_config["max_search"],
	clean_up=False
	)

	print('Trained Label Model Metrics:')
	disc_model.score((L["dev"], Y["dev"]), metric=['accuracy','precision', 'recall', 'f1'])

	Ytrain_p = disc_model.predict_proba(L["train"])
	#print(Ytrain_ps.shape) #(377*50,2)
	#Ydev_p = label_model.predict_proba(L["dev"])

	# test models
	#label_model.score((Ltest,Ytest), metric=['accuracy','precision', 'recall', 'f1'])
	
	#import ipdb; ipdb.set_trace()
	

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

	argparser.add_argument("-B", "--batch_size", type=int, default=4, help="batch size")
	argparser.add_argument("--quiet", action="store_true", help="suppress logging")
	argparser.add_argument("-H", "--host_device", type=str, default="gpu", help="Host device (GPU|CPU)")
	argparser.add_argument("--data_seed", type=int, default=4321, help="random sample seed")

	argparser.add_argument("--lr","--learning-rate",default=0.001,type=float,help="initial learning rate")
	argparser.add_argument("--momentum", default=0.9, type=float, help="momentum")
	argparser.add_argument("--weight-decay","--wd",default=1e-4,type=float,help="weight decay (default: 1e-4)")
	argparser.add_argument("-E", "--n_epochs", type=int, default=1, help="number of training epochs")

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


