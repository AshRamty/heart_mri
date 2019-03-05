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
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from metal.end_model import EndModel
from metal.contrib.modules import Encoder, LSTMModule
import metal.contrib.modules.resnet_cifar10 as resnet

from dataloaders.ukbb import UKBBCardiacMRI
from models.frame.densenet_av import DenseNet3, densenet_40_12_bc



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
			x = np.reshape(x,(n_batch*n_frame,ch,row,col))

			# forward pass
			out = self.cnn.forward(x) # dim (batch*frames,encode_dim,1,1)
			out = torch.squeeze(out) # dim (batch*frames,encode_dim)

			# reshape from 4D (batch*frames,encode_dim) -> 5D (batch,frames,encode_dim)
			encode_dim = out.shape[1]
			out = torch.reshape(out,(n_batch,n_frame,encode_dim))
			return out

		else : 
			return self.cnn.forward(x)   


class FrameEncoderOC(Encoder):
	
	def __init__(self,encoded_size, **kwargs):
		super().__init__(encoded_size)
		#self.n_classes  = n_classes
		self.use_cuda   	= torch.cuda.is_available()
		input_shape         = kwargs.get("input_shape", (3, 224, 224))
		#layers              = kwargs.get("layers", [64, 32])
		#dropout             = kwargs.get("dropout", 0.2)
		pretrained          = kwargs.get("pretrained", True)
		requires_grad       = kwargs.get("requires_grad", False)

		#self.cnn           = densenet_40_12_bc(pretrained=pretrained, requires_grad=requires_grad)
		self.cnn 			= models.resnet34(pretrained=pretrained) # try densenet121 
		self.encoded_size     = self.get_frm_output_size(input_shape) # 1000

		if(self.use_cuda):
			self.cnn = self.cnn.cuda()
		#print('encode dim: ', self.encoded_size)

	def get_frm_output_size(self, input_shape):
		input_shape = list(input_shape)
		input_shape.insert(0,1) # [1, 3, 224, 224]

		dummy_batch_size = tuple(input_shape)
		x = torch.autograd.Variable(torch.zeros(dummy_batch_size))		

		out = self.cnn.forward(x) 
		#print(out.shape) # [1,1000]
		frm_output_size = out.shape[1] # 1000 
		
		return frm_output_size 

	def encode(self,x):
		
		x = x.float()
		if(self.use_cuda):
				x = x.cuda()

		if (len(x.shape) == 5): # if 5D
			# reshape from 5D (batch,frames,3,img_row, img_col) -> 4D (batch*frames,3,img_row, img_col)
			n_batch,n_frame,ch,row,col = x.shape
			x = np.reshape(x,(n_batch*n_frame,ch,row,col))

			# forward pass
			out = self.cnn.forward(x) # dim (batch*frames,1000)

			# reshape from 4D (batch*frames,encode_dim) -> 5D (batch,frames,encode_dim)
			encode_dim = out.shape[1]
			out = torch.reshape(out,(n_batch,n_frame,encode_dim))
			#print(out.shape) # [4,50,1000]
			return out

		else :  # if 4D
			return self.cnn.forward(x) # dim [1,1000]

		
'''
class FrameEncoderOC(Encoder):
	
	def __init__(self,encoded_size, **kwargs):
		super().__init__(encoded_size)
		#self.n_classes  = n_classes
		#self.use_cuda   = use_cuda
		input_shape         = kwargs.get("input_shape", (3, 128, 128))
		layers              = kwargs.get("layers", [64, 32])
		dropout             = kwargs.get("dropout", 0.2)
		pretrained          = kwargs.get("pretrained", True)
		requires_grad       = kwargs.get("requires_grad", False)

		self.cnn           = densenet_40_12_bc(pretrained=pretrained, requires_grad=requires_grad)
		self.encoded_size     = self.get_frm_output_size(input_shape) # 2112
		#print('encode dim: ', self.encoded_size)

	def get_frm_output_size(self, input_shape):
		input_shape = list(input_shape)
		input_shape.insert(0,1) # [1, 3, 128, 128]

		dummy_batch_size = tuple(input_shape)
		x = torch.autograd.Variable(torch.zeros(dummy_batch_size))		
		#print(self.cnn.forward(x).size()) # [1,132,4,4]
		#frm_output_size =  self.cnn.forward(x).size()[1]* self.cnn.forward(x).size()[2]* self.cnn.forward(x).size()[3]
		
		out = self.cnn.forward(x) # [1,132,4,4]
		out = torch.reshape(out,(out.shape[0],-1)) # [1,2112]
		frm_output_size = out.shape[1]
		
		return frm_output_size # 132*4*4 = 2112

	def encode(self,x):
		
		#print(x.shape) # [1,3,128,128]
		if (len(x.shape) == 5): # if 5D
			x = x.float()
			n_batch,n_frame,ch,row,col = x.shape
			#print(x.shape)
			# reshape from 5D (batch,frames,3,img_row, img_col) -> 4D (batch*frames,3,img_row, img_col)
			x = np.reshape(x,(n_batch*n_frame,ch,row,col))

			# forward pass
			out = self.cnn.forward(x) # dim (batch*frames,132,4,4)
			#out = torch.squeeze(out) # dim (batch*frames,encode_dim)
			out = torch.reshape(out,(out.shape[0],-1)) # dim (batch*frames,encode_dim)

			# reshape from 4D (batch*frames,encode_dim) -> 5D (batch,frames,encode_dim)
			encode_dim = out.shape[1]
			out = torch.reshape(out,(n_batch,n_frame,encode_dim))
			#print(out.shape) # [4,50,2112]
			return out

		else :  # if 4D
			x = x.float()
			out = self.cnn.forward(x) # dim (num,132,4,4)
			out = torch.reshape(out,(out.shape[0],-1)) # dim (num,encode_dim)
			return out
'''