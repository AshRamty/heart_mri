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

class UKBB_LAX_Sequence(Dataset):
	"""
	UK Biobank cardiac MRI dataset
	LAX series 
	A single example is a patient with 50 frames and 50 frame labels

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


class UKBB_LAX_Roll(Dataset):
	"""
	UK Biobank cardiac MRI dataset
	LAX series 
	expands the data from using patient indexing to frame indexing
	(num_eg, num_frames, img) --> (num_eg*num_frames, num_frames, img)
	labels: (num_eg, num_frames) --> (num_eg*num_frames)
	Data is rolled such that the data starts with the frame_num corresponding to label


	"""
	def __init__(self,root_dir,labels, seed=123, mask = False):
		self.root_dir = root_dir
		self.labels = labels
		if(mask):
			self.list = glob(root_dir+'/la_4ch_masked/*.npy') 
		else:
			self.list = glob(root_dir+'/la_4ch/*.npy') 

		np.random.seed(seed)
		#else:
		#	self.mode = 'value'

		#import ipdb; ipdb.set_trace()

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
		return len(self.labels)

	def __getitem__(self, idx):
		all_labels = self.labels
		if(len(all_labels.shape) == 2):
			label = self.labels[idx,:]
		else:
			label = self.labels[idx]
		#print(label.shape)
		
		# switching order to have minimal class = 1
		label = 3 - label		

		# finding patient id number
		p_idx = idx // 50
		frame_num = idx % 50 
		
		filename = self.list[p_idx]
		series = np.load(filename)
		series = series.astype(float) # type float64		

		# roll series based on frame_num
		# to verify this.
		series = np.roll(series,-frame_num,axis=0) 
		#print(series.shape) # (50,108,108)

		# padding to 128 x 128 - to remove later by changing the cropping
		#series = np.pad(series,((0,0),(10,10),(10,10)),'minimum')
		#print(series.shape) # (50,128,128)
		n_frames, m, n = series.shape
		if(m<224):
			pad_size = (( 225 - m ) // 2 ) 
			series = np.pad(series,((0,0),(pad_size,pad_size),(0,0)),'minimum')

		if(n<224):
			pad_size = (( 225 - n ) // 2 ) 
			series = np.pad(series,((0,0),(0,0),(pad_size,pad_size)),'minimum')		

		#print(series.shape) # (50,224,224)
		series = np.expand_dims(series,1) # (50,1,224,224)

		# converting from gray to RGB
		series = np.concatenate((series,series,series),axis=1)
		#print(series.shape) # (50,3,224,224)

		return (series, label)


class UKBB_LAX_MR(Dataset):
	"""
	UK Biobank cardiac MRI dataset
	LAX series 
	A single example is a patient with 50 frames and 1 label

	"""
	def __init__(self, root_dir, mask = False, seed=4321):
		# either load from CSV or just use the provided pandas dataframe
		
		self.root_dir = root_dir
		self.list = glob(root_dir+'/la_4ch/*.npy') 
		np.random.seed(seed)
		#self.series = series
		#self.preprocess = preprocess
		#self.augment = augmentation
		#self.postprocess = postprocess
		csv_data = "{}/labels.csv".format(root_dir)
		labels = pd.read_csv(csv_data)
		self.labels = labels
		self.mask = mask

		#if frame_label:
		#    csv_data = "{}/labels_frame.csv".format(root_dir)

		#self.labels = pd.read_csv(csv_data) if type(csv_data) is str else csv_data


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
		return len(self.labels)


	def __getitem__(self, idx):
		#filename = self.list[idx]		
		pid = self.labels.iloc[idx, 0]

		if(self.mask):
			data_filename = self.root_dir+'/la_4ch/'+str(pid)+'.npy'
			mask_filename = self.root_dir+'/la_4ch_mask/'+str(pid)+'.npy'
			temp_data = np.load(data_filename)
			temp_mask = np.load(mask_filename)
			series = np.multiply(temp_data,temp_mask)
		else:
			filename = self.root_dir+'/la_4ch/'+str(pid)+'.npy'
			#print(filename)
			series = np.load(filename)

				
		#print(series.shape) # (50,208,x)

		# padding to 224 x 224
		n_frames, m, n = series.shape
		if(m<224):
			pad_size = (( 225 - m ) // 2 ) 
			series = np.pad(series,((0,0),(pad_size,pad_size),(0,0)),'minimum')

		if(n<224):
			pad_size = (( 225 - n ) // 2 ) 
			series = np.pad(series,((0,0),(0,0),(pad_size,pad_size)),'minimum')	

		series = np.expand_dims(series,1)
		# converting from gray to RGB
		series = np.concatenate((series,series,series),axis=1)
		series = series.astype(float)
		#print(series.shape) # (50,3,224,224)

		label = self.labels.iloc[idx, 1]
		label = 2-label # converting to 1-indexing and making minority class = 1
		#print(label)

		return (series, label)


class UKBB_LAX_SelfSupervised(Dataset):
	"""
	UK Biobank cardiac MRI dataset
	LAX series 
	For the self-supervised task 
	Uses a similar structure to "UKB_LAX_Roll" which is used for the open / close task
	expands the data from using patient indexing to frame indexing
	(num_eg, num_frames, img) --> (num_eg*num_frames, num_frames, img)
	labels: (num_eg, num_frames) --> (num_eg*num_frames)
	Data is rolled such that the data starts with the frame_num corresponding to label
	
	The data is assigned class 1 if it has the correct temporal ordering
	class 2 if it has a shuffled temporal ordering 
	

	"""
	def __init__(self, root_dir, seed=123, mask = False, shuffle = False):
		self.root_dir = root_dir
		if(mask):
			self.list = glob(root_dir+'/la_4ch_masked/*.npy') 
		else:
			self.list = glob(root_dir+'/la_4ch/*.npy') 

		np.random.seed(seed)
		self.shuffle = shuffle

	def __len__(self):
		return len(self.list*50) # number of PIDs x number of frames - to write better

	def __getitem__(self, idx):
		# finding patient id number
		p_idx = idx // 50
		frame_num = idx % 50 
		
		filename = self.list[p_idx]
		series = np.load(filename)
		series = series.astype(float) # type float64		

		# roll series based on frame_num
		# to verify this.
		series = np.roll(series,-frame_num,axis=0) 
		#print(series.shape) # (50,108,108)

		# padding to 128 x 128 - to remove later by changing the cropping
		#series = np.pad(series,((0,0),(10,10),(10,10)),'minimum')
		#print(series.shape) # (50,128,128)
		n_frames, m, n = series.shape
		if(m<224):
			pad_size = (( 225 - m ) // 2 ) 
			series = np.pad(series,((0,0),(pad_size,pad_size),(0,0)),'minimum')

		if(n<224):
			pad_size = (( 225 - n ) // 2 ) 
			series = np.pad(series,((0,0),(0,0),(pad_size,pad_size)),'minimum')		

		#print(series.shape) # (50,224,224)
		
		if(self.shuffle == True):
			indices = np.arange(n_frames)
			np.random.shuffle(indices)
			series = series[indices,:,:]
			#np.random.shuffle(series) # by default shuffles first axis
		else:
			if(np.random.random() > 0.5): # sequential order maintained
				label = 1;
			else: 	# two random frames are swapped
				label = 2;
				num_shuffle = 5;
				for i in np.arange(num_shuffle):
					frame1 = np.random.randint(0,n_frames-1)
					frame2 = np.random.randint(0,n_frames-1)
					temp = series[frame1,:,:]
					series[frame1,:,:] = series[frame2,:,:]
					series[frame2,:,:] = temp
			

		# converting from gray to RGB
		series = np.expand_dims(series,1) # (50,1,224,224)
		series = np.concatenate((series,series,series),axis=1)
		#print(series.shape) # (50,3,224,224)

		return (series, label)
