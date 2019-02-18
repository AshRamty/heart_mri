# Script to define dataloader for LAX data
# starting with LAX 4ch data
import os
import sys
import logging
from glob import glob
import numpy as np
import pandas as pd
from skimage.color import grey2rgb
from torch.utils.data import Dataset, DataLoader

class UKBB_LAX(Dataset):
	"""
	UK Biobank cardiac MRI dataset
	LAX series 

	"""
	def __init__(self, root_dir, seed=4321):
		# either load from CSV or just use the provided pandas dataframe
		#if frame_label:
		#    csv_data = "{}/labels_frame.csv".format(root_dir)

		#self.labels = pd.read_csv(csv_data) if type(csv_data) is str else csv_data
		self.root_dir = root_dir
		#self.series = series
		#self.preprocess = preprocess
		#self.augment = augmentation
		#self.postprocess = postprocess
		self.list = glob(root_dir+'/*.npy') 
		np.random.seed(seed)


	#def summary(self):
		"""
		Generate message summarizing data (e.g., counts, class balance)
		Assumes hard labels
		:return:
		"""
		#return "Instances: {}".format(len(self))

	#def get_labels(self):
		#return [(str(self.labels.iloc[i, 0]), data[1]) for i, data in enumerate(self)]

	def __len__(self):
		#return len(self.labels)
		return len(self.list)


	def __getitem__(self, idx):
		filename = self.list[idx]
		PID = filename[20:27] # to write a better way to find this

		series = np.load(filename)
		label = np.load(self.root_dir+'/labels/'+PID+'.npy')

		#print(series.shape) # (50,108,108)
		series = np.expand_dims(series,1)
		# converting from gray to RGB
		series = np.concatenate((series,series,series),axis=1)
		#print(series.shape) # (50,3,108,108)

		return (series, label)