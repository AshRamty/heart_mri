"""
MeTaL CNN-LSTM module using BAV data
with hyperparameter tuning

"""

import sys
sys.path.append('../metal')
sys.path.append('../heart-MRI-pytorch')
sys.path.append('../data')

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
import metal.contrib.modules.resnet_cifar10 as resnet
from metal.tuners.random_tuner import RandomSearchTuner

from metal.logging import LogWriter
from metal.logging.tensorboard import TensorBoardWriter

from dataloaders.ukbb import UKBBCardiacMRI
from models.frame.densenet_av import DenseNet3, densenet_40_12_bc
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
	
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# dataset
def load_dataset(args):
	"""
	Load UKBB 

	Image centering statistics

	/lfs/1/heartmri/coral32/flow_250_tp_AoV_bh_ePAT@c/
	    max:  192
	    mean: 27.4613475359
	    std:  15.8350095314

	/lfs/1/heartmri/coral32/flow_250_tp_AoV_bh_ePAT@c_P/
	    max:  4095
	    mean: 2045.20689212
	    std:  292.707986212

	/lfs/1/heartmri/coral32/flow_250_tp_AoV_bh_ePAT@c_MAG/
	    max:  336.0
	    mean: 24.1274
	    std:  14.8176

	:return: train, dev, test, classes
	"""

	DataSet = UKBBCardiacMRI
	classes = ("TAV", "BAV")

	# Get Preprocessing and Augmentation params
	preprocessing, augmentation, postprocessing = get_data_config(args)
	print_dict_pairs(preprocessing, title="Data Preprocessing Args")
	print_dict_pairs(augmentation, title="Data Augmentation Args")
	preprocessing["n_frames"] = args.n_frames

	# Preprocessing data should be computed on ALL datasets (train, val,
	#   and test). This includes:
	#       - Frame Selection
	#       - Rescale Intensity
	#       - Gamma Correction
	if (args.series == 3):
		preprocess_data = compose_preprocessing_multi(preprocessing)
	else:
		preprocess_data = compose_preprocessing(preprocessing)

	# HACK ignore augmentations (for now)
	#   data augmentations only to be used during training
	augment_train = None
	if (augmentation is not None):
		augment_train = compose_augmentation(augmentation, seed=args.data_seed)

	postprocess_data = None
	if (postprocessing is not None):
		if (args.series == 3):
			postprocess_data = compose_postprocessing_multi(postprocessing)
		else:
			postprocess_data = compose_postprocessing(postprocessing)

	train = DataSet("{}/{}".format(args.train, args.labelcsv), args.train,
		series=args.series, N=args.n_frames,
		image_type=args.image_type,
		preprocess=preprocess_data,
		augmentation=augment_train,
		postprocess=postprocess_data,
		rebalance=args.rebalance,
		threshold=args.data_threshold,
		seed=args.data_seed,
		sample=args.sample,
		sample_type=args.sample_type,
		sample_split=args.sample_split,
		n_samples=args.n_samples,
		pos_samples=args.pos_samples,
		neg_samples=args.neg_samples,
		frame_label=args.use_frame_label,
		rebalance_strategy=args.rebalance_strategy,
		semi=args.semi, semi_dir=args.semi_dir, semi_csv=args.semi_csv)

	# randomly split dev into stratified dev/test sets
	if args.stratify_dev:
		df = stratified_sample_dataset("{}/labels.csv".format(args.dev), args.seed)
		dev = DataSet(df["dev"], args.dev,
			series=args.series, N=args.n_frames,
			image_type=args.image_type,
			preprocess=preprocess_data,
			postprocess=postprocess_data,
			seed=args.data_seed)
		test = DataSet(df["test"], args.dev,
			series=args.series, N = args.n_frames,
			image_type=args.image_type,
			preprocess=preprocess_data,
			postprocess=postprocess_data,
			seed=args.data_seed)
	else: 
	# use manually defined dev/test sets
		dev = DataSet("{}/labels.csv".format(args.dev), args.dev,
			series=args.series, N=args.n_frames,
			image_type=args.image_type,
			preprocess=preprocess_data,
			postprocess=postprocess_data,
			seed=args.data_seed)
		if args.test:
			test = DataSet("{}/labels.csv".format(args.test), args.test,
				series=args.series, N=args.n_frames,
				image_type=args.image_type,
				preprocess=preprocess_data,
				postprocess=postprocess_data,
				seed=args.data_seed)
		else:
			test = None

		return train, dev, test, classes


# dataloader 
def data_loader(train, dev, test=None, batch_size=4, num_workers=1):
	
	#train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	train_loader = DataLoader(train,sampler=ImbalancedDatasetSampler(train), batch_size=batch_size, num_workers=num_workers)
	dev_loader   = DataLoader(dev, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	test_loader  = None if not test else DataLoader(test, batch_size=batch_size,shuffle=False, num_workers=num_workers)

	return train_loader, dev_loader, test_loader


class FrameEncoderBAV(Encoder):
	
	# from Dense4012FrameNet class in mri.py
	def __init__(self,encoded_size, **kwargs):
		super().__init__(encoded_size)
		#self.n_classes  = n_classes
		#self.use_cuda   = use_cuda
		input_shape         = kwargs.get("input_shape", (3, 32, 32))
		layers              = kwargs.get("layers", [64, 32])
		dropout             = kwargs.get("dropout", 0.2)
		pretrained          = kwargs.get("pretrained", True)
		requires_grad       = kwargs.get("requires_grad", False)

		self.cnn           = densenet_40_12_bc(pretrained=pretrained, requires_grad=requires_grad)
		self.encoded_size     = self.get_frm_output_size(input_shape) # 132

	def get_frm_output_size(self, input_shape):
		input_shape = list(input_shape)
		input_shape.insert(0,1) # [1, 3, 32, 32]
		#print(input_shape)

		dummy_batch_size = tuple(input_shape)
		x = torch.autograd.Variable(torch.zeros(dummy_batch_size))
		
		frm_output_size =  self.cnn.forward(x).size()[1]
		#print(self.cnn.forward(x).size()) # [1,132,1,1]
		
		return frm_output_size

	def encode(self,x):
		
		if (len(x.shape) == 5): # if 5D
			n_batch,n_frame,ch,row,col = x.shape

			# reshape from 5D (batch,frames,3,img_row, img_col) -> 4D (batch*frames,3,img_row, img_col)
			x = torch.reshape(x,(n_batch*n_frame,ch,row,col))

			# forward pass
			out = self.cnn.forward(x) # dim (batch*frames,encode_dim,1,1)
			out = torch.squeeze(out) # dim (batch*frames,encode_dim)

			# reshape from 4D (batch*frames,encode_dim) -> 5D (batch,frames,encode_dim)
			encode_dim = out.shape[1]
			out = torch.reshape(out,(n_batch,n_frame,encode_dim))
			return out

		# else
		else : 
			return self.cnn.forward(x)    


# CNN-LSTM 
def train_model(args):

	#global args
	#args = parser.parse_args()

	# Create datasets and dataloaders
	train, dev, test, classes = load_dataset(args)
	#print('train size:',len(train)) # 106
	#print('dev size:',len(dev)) # 216
	#print('test size:',len(test)) # 90
	# data in tuple of the form (series,label)
	# series shape [30,3,32,32]

	train_loader, dev_loader, test_loader = data_loader(train, dev, test, args.batch_size)

	hidden_size = 128 
	num_classes = 2
	encode_dim = 132 # using get_frm_output_size()

	# Define input encoder
	cnn_encoder = FrameEncoderBAV

	# Define LSTM module
	lstm_module = LSTMModule(
		encode_dim,
		hidden_size,
		bidirectional=False,
		verbose=False,
		lstm_reduction="attention",
		encoder_class=cnn_encoder,
		)

	#import ipdb; ipdb.set_trace()
	
	if(torch.cuda.is_available()):
		device = 'cuda'
	else:
		device = 'cpu'

	train_args = [train_loader]

	train_kwargs = {
	'seed':123,
	'log_train_every':1,
	'checkpoint_metric':'f1',
	'log_valid_metrics':['accuracy','f1']
	}

	init_args = [
	[hidden_size, num_classes]
	]

	init_kwargs = {
	"input_module": lstm_module, 
	"optimizer": "adam",
	"verbose": False,
	"input_batchnorm": True,
	#"use_cuda":torch.cuda.is_available(),
	'device':device,
	'seed':123}
	
	'''
	search_space = {
	'seed' : [123],
	'n_epochs': [5],
	'batchnorm' : [True],
	'dropout': [0],
	'lr': [1e-3],
	'log_train_every': 1,
	'validation_metric': 'f1',
	}
	search_space = {
	'seed' : [123],
	'n_epochs': [30],
	'batchnorm' : [True, False],
	'dropout': [0.1,0.25,0.5],
	'lr': {'range': [1e-3, 1], 'scale': 'log'},
	'l2':{'range': [1e-5, 1e-3], 'scale': 'log'},
	'log_train_every': 1,
	'loss_weights':[[0.2,0.8],[0.4,0.6],[0.6,0.4],[0.8,0.2]],
	#'validation_metric': ['f1'],
	'validation_metric':[['roc-auc', 'accuracy', 'precision', 'recall', 'f1']],
	}
	'''
	search_space = {
	'n_epochs':[100],
	'batchnorm':[True],
	'dropout': [0.1,0.25,0.4],
	'lr':{'range': [1e-3, 1e-1], 'scale': 'log'}, 
	'l2':{'range': [1e-5, 1e-3], 'scale': 'log'},#[ 1.21*1e-5],
	'loss_weights':[[0.04,0.96]],
	}	
	
	log_config = {
	"log_dir": "./run_logs", 
	"run_name": 'cnn_lstm_bav'
	}

	max_search = 5
	tuner_config = {"max_search": max_search }

	validation_metric = 'f1'

	# Set up logger and searcher
	tuner = RandomSearchTuner(EndModel, 
	**log_config,
	log_writer_class=TensorBoardWriter,
	validation_metric=validation_metric,
	seed=1701)
	
	disc_model = tuner.search(
	search_space,
	valid_data = dev_loader,
	train_args=train_args,
	init_args=init_args,
	init_kwargs=init_kwargs,
	train_kwargs=train_kwargs,
	max_search=tuner_config["max_search"],
	clean_up=False,
	)

	#import ipdb; ipdb.set_trace()


if __name__ == "__main__":
	# Checking to see if cuda is available for GPU use
	cuda = torch.cuda.is_available()

	# Parsing command line arguments
	argparser = argparse.ArgumentParser(description="Training CNN LSTM on BAV data")
	#parser.add_argument( "--epochs", default=1, type=int, help="number of total epochs to run" )
	#parser.add_argument("-b","--batch-size", default=4, type=int, help="mini-batch size (default: 4)")
	argparser.add_argument("--lr","--learning-rate",default=0.001,type=float,help="initial learning rate")
	argparser.add_argument("--momentum", default=0.9, type=float, help="momentum")
	argparser.add_argument("--weight-decay","--wd",default=1e-4,type=float,help="weight decay (default: 1e-4)")

	argparser.add_argument("-d", "--dataset", type=str, default="UKBB", help="dataset name")
	argparser.add_argument("-L", "--labelcsv", type=str, default="labels.csv", help="dataset labels csv filename")

	argparser.add_argument("--train", type=str, default=None, help="training set")
	argparser.add_argument("--dev", type=str, default=None, help="dev (validation) set")
	argparser.add_argument("--test", type=str, default=None, help="test set")
	argparser.add_argument("--stratify_dev", action="store_true", help="split dev into stratified dev/test")

	argparser.add_argument("-c", "--config", type=str, default=None, help="load model config JSON")
	argparser.add_argument("-g", "--param_grid", type=str, default=None, help="load manual parameter grid from JSON")
	argparser.add_argument("-p", "--params", type=str, default=None, help="load `key=value,...` pairs from command line")
	argparser.add_argument("-o", "--outdir", type=str, default=None, help="save model to outdir")

	argparser.add_argument("-a", "--dconfig", type=str, default=None, help="load data config JSON")

	argparser.add_argument("-R", "--rebalance", action="store_true", help="rebalance training data")
	argparser.add_argument("--data_threshold", type=float, default=0.5, help="threshold cutoff to use when sampling patients")
	argparser.add_argument("--data_seed", type=int, default=4321, help="random sample seed")

	argparser.add_argument("--sample", action="store_true", help="sample training data")
	argparser.add_argument("--sample_type", type=int, default=0, choices=[0, 1, 2, 3],
		help="sample method to use [1: Random Sample, 1: Threshold Random Sample, 2: Top/Bottom Sample]")
	argparser.add_argument("--sample_split", type=float, default=0.5, help="ratio of 'positive' classes wanted")
	argparser.add_argument("--n_samples", type=int, default=100, help="number of patients to sample")
	argparser.add_argument("--pos_samples", type=int, default=0, help="number of positive patients to sample")
	argparser.add_argument("--neg_samples", type=int, default=0, help="number of negative patients to sample")
	argparser.add_argument("--rebalance_strategy", type=str, default="oversample", help="over/under sample")

	argparser.add_argument("-B", "--batch_size", type=int, default=4, help="batch size")
	argparser.add_argument("-N", "--n_model_search", type=int, default=1, help="number of models to search over")
	argparser.add_argument("-S", "--early_stopping_metric", type=str, default="roc_auc_score", help="the metric for checkpointing the model")
	argparser.add_argument("-T", "--tune_metric", type=str, default="roc_auc_score", help="the metric for "
		"tuning the threshold. str-`roc_auc_score` for metric, float-`0.6` for fixed threshold")
	argparser.add_argument("-E", "--n_epochs", type=int, default=1, help="number of training epochs")
	argparser.add_argument("-M", "--n_procs", type=int, default=1, help="number processes (per model, CPU only)")
	argparser.add_argument("-W", "--n_workers", type=int, default=1, help="number of grid search workers")
	argparser.add_argument("-H", "--host_device", type=str, default="gpu", help="Host device (GPU|CPU)")
	argparser.add_argument("-U", "--update_freq", type=int, default=5, help="progress bar update frequency")
	argparser.add_argument("-C", "--checkpoint_freq", type=int, default=5, help="checkpoint frequency")
	argparser.add_argument("-I", "--image_type", type=str, default='grey', choices=['grey', 'rgb'], help="the image type, grey/rgb")
	argparser.add_argument("--use_cuda", action="store_true", help="whether to use GPU(CUDA)")
	argparser.add_argument("--cache_data", action="store_true", help="whether to cache data into memory")
	argparser.add_argument("--meta_data", action="store_true", help="whether to include meta data in model")
	argparser.add_argument("--semi", action="store_true", help="whether to use semi model")
	argparser.add_argument("--semi_dir", type=str, default='/lfs/1/heartmri/train32', help="path to train folder in semi model")
	argparser.add_argument("--semi_csv", type=str, default="labels.csv", help="semi dataset labels csv filename")
	argparser.add_argument("-F", "--n_frames", type=int, default=30, help="number of frames to select from a series")
	argparser.add_argument("--use_frame_label", action="store_true", help="whether to use frame level labels.")

	argparser.add_argument("--pretrained", action="store_true", help="whether to load pre_trained weights.")
	argparser.add_argument("--requires_grad", action="store_true", help="whether to fine tuning the pre_trained model.")
	argparser.add_argument("--noise_aware", action="store_true", help="whether to train on probability labels.")
	argparser.add_argument("--series", type=int, default=0, choices=[0, 1, 2, 3], help="which series to load for training")
	argparser.add_argument("--report", action="store_true", help="generate summary plots")
	argparser.add_argument("--seed", type=int, default=1234, help="random model seed")
	argparser.add_argument("--quiet", action="store_true", help="suppress logging")
	argparser.add_argument("--verbose", action="store_true", help="print debug information to log")
	argparser.add_argument("--top_selection", type=int, default=None, help="the number of positive cases to select from the test set")
	argparser.add_argument("--tsne_save_coords", action="store_true", help="whether to save coords of tsne.")
	argparser.add_argument("--tsne_pred_only", action="store_true", help="whether to plot preds only in tsne.")
	argparser.add_argument("--tsne_classes", default=None, type=int, action='append', help="the classes used to plot tsne plots. defaultto read from labels Y_TRUE.")
	argparser.add_argument("--save_embeds", action="store_true", help="whether to save the embedding of test set.")

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
