"""
Tests dataloader of LAX 4Ch data

"""


import sys
sys.path.append('../metal')
sys.path.append('../heart-MRI-pytorch')
sys.path.append('../data')

import numpy as np
import argparse
import torch
import logging
import pandas
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from dataloader_4ch import UKBB_LAX_Sequence
from utils import *

from metal.end_model import EndModel
from metal.modules import EmbeddingsEncoder, Encoder, LSTMModule
from models.frame.densenet_av import densenet_40_12_bc

logger = logging.getLogger(__name__)



def load_dataset(args):
	'''
	Loading LAX 4ch data
	'''
	DataSet = UKBB_LAX_Sequence

	train = DataSet(args.train,
		seed=args.data_seed)

	dev = DataSet(args.dev,
		seed=args.data_seed)
	
	if args.test:
		test = DataSet(args.test,
		seed=args.data_seed)
	else:
		test = None

	return train, dev, test

# dataloader 
def data_loader(train, dev, test=None, batch_size=4, num_workers=1):
	
	train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	dev_loader   = DataLoader(dev, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	test_loader  = None if not test else DataLoader(test, batch_size=batch_size,shuffle=False, num_workers=num_workers)

	return train_loader, dev_loader, test_loader



class FrameEncoder(Encoder):
	
	'''
	def __init__(self):
		super(Encoder,self).__init__()

	self.cnn = torchvision.models.resnet18()
	def encode(self,x):
		return self.cnn.forward(x)
	'''
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
		#print('encode dim: ', self.encoded_size)
		#self.classifier = self._make_classifier(encode_dim, n_classes, layers, dropout)

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
			x = np.reshape(x,(n_batch*n_frame,ch,row,col))

			# forward pass
			out = self.cnn.forward(x) # dim (batch*frames,encode_dim,1,1)
			out = torch.squeeze(out) # dim (batch*frames,encode_dim)

			# reshape from 4D (batch*frames,encode_dim) -> 5D (batch,frames,encode_dim)
			encode_dim = out.shape[1]
			out = torch.reshape(out,(n_batch,n_frame,encode_dim))
			# print(out.shape) # [4,30,132]
			return out

		# else
		else : 
			return self.cnn.forward(x)    


class LSTMSequenceEndModel(EndModel):
    def _get_loss_fn(self):
        # Overwriting what was done in _build to have normal BCELoss
        self.criteria = nn.BCELoss()
        if self.config["use_cuda"]:
            criteria = self.criteria.cuda()
        else:
            criteria = self.criteria
        # This self.preprocess_Y allows us to not handle preprocessing
        # in a custom dataloader, but decreases speed a bit
        loss_fn = lambda X, Y: criteria(self.forward(X), Y)
        return loss_fn


def train_model(args):

    #global args
    #args = parser.parse_args()

	# Create datasets and dataloaders
	train, dev, test = load_dataset(args)
	#print('train size:',len(train)) # 347
	#print('dev size:',len(dev))  # 42
	#print('test size:',len(test)) # 42
	#(series,label) = train.__getitem__(0)
	# series shape [50,3,108,108]

	train_loader, dev_loader, test_loader = data_loader(train, dev, test, args.batch_size)

	hidden_size = 128 
	num_classes = 2

	# Define input encoder
	cnn_encoder = FrameEncoder
	
	# using get_frm_output_size()
	encode_dim = 132

	# Define LSTM module
	lstm_module = LSTMModule(
		encode_dim,
		hidden_size,
		bidirectional=False,
		verbose=False,
		lstm_reduction="none",
		encoder=cnn_encoder,
		)

	# Define end model
	end_model = LSTMSequenceEndModel(
		input_module=lstm_module,
		layer_out_dims=[hidden_size, num_classes],
		optimizer="adam",
		use_cuda=cuda,
		batchnorm=True,
		seed=1,
		verbose=True,
		)

	# Train end model
	end_model.train_model(
		train_data=train_loader,
		dev_data=dev_loader,
		l2=args.weight_decay,
		lr=args.lr,
		n_epochs=args.n_epochs,
		print_every=1,
		verbose=True,
		)

	# Test end model
	#end_model.score(test_loader, verbose=False)
	end_model.score( test_loader, metric=['accuracy','precision', 'recall', 'f1'])


if __name__ == "__main__":
	# Checking to see if cuda is available for GPU use
	cuda = torch.cuda.is_available()

	# Parsing command line arguments
	argparser = argparse.ArgumentParser(description="Loading LAX 4Ch data")

	argparser.add_argument("--train", type=str, default=None, help="training set")
	argparser.add_argument("--dev", type=str, default=None, help="dev (validation) set")
	argparser.add_argument("--test", type=str, default=None, help="test set")

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