# Script to define dataloader for LAX data
# starting with LAX 4ch data
import os
import sys
import logging
from glob import glob
import numpy as np
import pandas as pd
import cv2
from skimage.color import grey2rgb
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision

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
	def __init__(self,root_dir,labels, seed=123, mask = False, preprocess = True):
		self.root_dir = root_dir
		self.labels = labels
		self.preprocess = preprocess
		if(mask):
			self.list = glob(root_dir+'/la_4ch_masked/*.npy') 
		else:
			self.list = glob(root_dir+'/la_4ch/*.npy') 

		np.random.seed(seed)

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

		if(self.preprocess):
			temp_input = np.load(filename)
			temp = np.zeros(temp_input.shape)
			series = np.zeros(temp_input.shape)
			# min-max normalization ( to apply z -normalization? )
			for frame_num in range(series.shape[0]):
				temp[frame_num,:,:] = cv2.normalize(temp_input[frame_num,:,:], None, 0, 255, cv2.NORM_MINMAX)
			# histogram equalization
			temp = np.uint8(temp)
			clahe = cv2.createCLAHE(clipLimit=0.02)
			for frame_num in range(series.shape[0]):
				series[frame_num,:,:] = clahe.apply(temp[frame_num,:,:])
		else:
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


class UKBB_LAX_Roll_CV(Dataset):
	"""
	UK Biobank cardiac MRI dataset
	LAX series 
	expands the data from using patient indexing to frame indexing
	(num_eg, num_frames, img) --> (num_eg*num_frames, num_frames, img)
	labels: (num_eg, num_frames) --> (num_eg*num_frames)
	Data is rolled such that the data starts with the frame_num corresponding to label


	"""
	def __init__(self,data_list,labels, seed=123):
		self.list = data_list
		self.labels = labels
		np.random.seed(seed)

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
	def __init__(self, root_dir, mask = False, seed=4321, preprocess = True):
		# either load from CSV or just use the provided pandas dataframe
		
		self.root_dir = root_dir
		self.list = glob(root_dir+'/la_4ch/*.npy') 
		np.random.seed(seed)
		#self.series = series
		self.preprocess = preprocess
		#self.augment = augmentation
		#self.postprocess = postprocess
		csv_data = "{}/labels.csv".format(root_dir)
		labels = pd.read_csv(csv_data)
		self.labels = labels
		self.mask = mask

		#if frame_label:
		#    csv_data = "{}/labels_frame.csv".format(root_dir)

		#self.labels = pd.read_csv(csv_data) if type(csv_data) is str else csv_data

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

		if(self.preprocess):
			temp_input = np.copy(series)
			temp = np.zeros(temp_input.shape)
			series = np.zeros(temp_input.shape)
			# min-max normalization ( to apply z -normalization? )
			for frame_num in range(series.shape[0]):
				temp[frame_num,:,:] = cv2.normalize(temp_input[frame_num,:,:], None, 0, 255, cv2.NORM_MINMAX)
			# histogram equaliztempation
			temp = np.uint8(temp)
			clahe = cv2.createCLAHE(clipLimit=0.02)
			for frame_num in range(series.shape[0]):
				series[frame_num,:,:] = clahe.apply(temp[frame_num,:,:])


		series = series.astype(float) # type float64		

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
		#print(series.shape) # (50,3,224,224)

		label = self.labels.iloc[idx, 1]
		label = 2-label # converting to 1-indexing and making minority class = 1 - to change this
		#print(label)

		return (series, label)


class UKBB_MR_Framewise(Dataset):
	"""
	UK Biobank cardiac MRI dataset
	LAX series 
	A single example is a patient with 1 frames and 1 label

	"""
	def __init__(self, root_dir, mask = False, seed=4321, preprocess = False):		
		self.root_dir = root_dir
		self.list = glob(root_dir+'/la_4ch/*.npy') 
		np.random.seed(seed)
		self.preprocess = preprocess
		csv_data = "{}/labels.csv".format(root_dir)
		labels = pd.read_csv(csv_data)
		self.labels = labels
		self.mask = mask

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
			frame = np.multiply(temp_data,temp_mask)
		else:
			filename = self.root_dir+'/la_4ch/'+str(pid)+'.npy'
			#print(filename)
			frame = np.load(filename)
				
		#print(frame.shape) # (208,x)

		if(self.preprocess):
			temp_input = np.copy(frame)
			temp = cv2.normalize(temp_input, None, 0, 255, cv2.NORM_MINMAX)
			temp = np.uint8(temp)
			clahe = cv2.createCLAHE(clipLimit=0.02)
			frame = clahe.apply(temp)

		frame = frame.astype(np.float)	
		#frame = frame.astype(np.double)

		# padding to 224 x 224
		m, n = frame.shape

		if(m<224):
			pad_size = (( 225 - m ) // 2 ) 
			frame = np.pad(frame,((pad_size,pad_size),(0,0)),'minimum')

		if(n<224):
			pad_size = (( 225 - n ) // 2 ) 
			frame = np.pad(frame,((0,0),(pad_size,pad_size)),'minimum')	

		frame = np.expand_dims(frame,0)
		# converting from gray to RGB
		frame = np.concatenate((frame,frame,frame),axis=0) # should it be axis=0?
		#print(frame.shape) # (3,224,224)
		#frame = torchvision.transforms.ToTensor(frame)
		#frame = torch.from_numpy(frame)		

		label = self.labels.iloc[idx, 1]
		label = 2-label # converting to 1-indexing and making minority class = 1 - to change this
		#print(label)
		#frame = torch.tensor(frame)
		return (frame, label)



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
		
		if(np.random.random() > 0.5): # sequential order maintained
			label = 1;
		else: 	# two random frames are swapped
			label = 2;
			if(self.shuffle == True):
				indices = np.arange(n_frames)
				np.random.shuffle(indices)
				series = series[indices,:,:]
				#np.random.shuffle(series) # by default shuffles first axis
			else:
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


class UKBB_LAX_Roll2(Dataset):
	"""
	UK Biobank cardiac MRI dataset
	LAX series 
	expands the data from using patient indexing to frame indexing
	(num_eg, num_frames, img) --> (num_eg*num_frames, num_frames, img)
	labels: (num_eg, num_frames) --> (num_eg*num_frames)
	Data is rolled such that the data starts with the frame_num corresponding to label
	
	Data from both the 2 chamber view and 4 chamber view are used
	the views are concatenated - with the 4ch view series first and 2ch view next

	"""
	def __init__(self,root_dir,labels, seed=123, mask = False, preprocess = False):
		self.root_dir = root_dir
		self.labels = labels
		self.preprocess = preprocess
		if(mask):
			self.list4ch = glob(root_dir+'/la_4ch_masked/*.npy') 
			self.list2ch = glob(root_dir+'/la_2ch_masked/*.npy') 
		else:
			self.list4ch = glob(root_dir+'/la_4ch/*.npy') 
			self.list2ch = glob(root_dir+'/la_2ch/*.npy')

		np.random.seed(seed)

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
		
		filename4ch = self.list4ch[p_idx]
		filename2ch = self.list2ch[p_idx]

		if(self.preprocess):
			input4ch = np.load(filename4ch)
			input2ch = np.load(filename2ch)

			temp4ch = np.zeros(input4ch.shape)
			series4ch = np.zeros(input4ch.shape)
			# min-max normalization ( to apply z -normalization? )
			for frame_num in range(series4ch.shape[0]):
				temp4ch[frame_num,:,:] = cv2.normalize(input4ch[frame_num,:,:], None, 0, 255, cv2.NORM_MINMAX)
			# histogram equalization
			temp4ch = np.uint8(temp4ch)
			clahe = cv2.createCLAHE(clipLimit=0.02)
			for frame_num in range(series4ch.shape[0]):
				series4ch[frame_num,:,:] = clahe.apply(temp4ch[frame_num,:,:])

			temp2ch = np.zeros(input2ch.shape)
			series2ch = np.zeros(input2ch.shape)
			# min-max normalization ( to apply z -normalization? )
			for frame_num in range(series4ch.shape[0]):
				temp2ch[frame_num,:,:] = cv2.normalize(input2ch[frame_num,:,:], None, 0, 255, cv2.NORM_MINMAX)
			# histogram equalization
			temp2ch = np.uint8(temp2ch)
			clahe = cv2.createCLAHE(clipLimit=0.02)
			for frame_num in range(series2ch.shape[0]):
				series2ch[frame_num,:,:] = clahe.apply(temp2ch[frame_num,:,:])
		else:
			series4ch = np.load(filename4ch)
			series2ch = np.load(filename2ch)

		series4ch = series4ch.astype(float) # type float64	
		series4ch = np.roll(series4ch,-frame_num,axis=0) 
		if (series4ch.shape[1]<series4ch.shape[2]):
			series4ch = series4ch.transpose([0,2,1])

		n_frames, m, n = series4ch.shape
		if(m<224):
			pad_size = (( 225 - m ) // 2 ) 
			series4ch = np.pad(series4ch,((0,0),(pad_size,pad_size),(0,0)),'minimum')
		if(n<224):
			pad_size = (( 225 - n ) // 2 ) 
			series4ch = np.pad(series4ch,((0,0),(0,0),(pad_size,pad_size)),'minimum')	
		
		series4ch = np.expand_dims(series4ch,1) # (50,1,224,224)
		series4ch = np.concatenate((series4ch,series4ch,series4ch),axis=1)
	
		series2ch = series2ch.astype(float) # type float64	
		series2ch = np.roll(series2ch,-frame_num,axis=0)
		if (series2ch.shape[1]<series2ch.shape[2]):
			series2ch = series2ch.transpose([0,2,1])

		n_frames, m, n = series2ch.shape
		if(m<224):
			pad_size = (( 225 - m ) // 2 ) 
			series2ch = np.pad(series2ch,((0,0),(pad_size,pad_size),(0,0)),'minimum')
		if(n<224):
			pad_size = (( 225 - n ) // 2 ) 
			series2ch = np.pad(series2ch,((0,0),(0,0),(pad_size,pad_size)),'minimum')	
		
		series2ch = np.expand_dims(series2ch,1) # (50,1,224,224)
		series2ch = np.concatenate((series2ch,series2ch,series2ch),axis=1)
		#print(series2ch.shape)
		#print(series4ch.shape)
		series = np.concatenate((series4ch,series2ch),0) # (100,3,224,224)


		return (series, label)


class UKBB_LAX_Roll3(Dataset):
	"""
	UK Biobank cardiac MRI dataset
	LAX series 
	expands the data from using patient indexing to frame indexing
	(num_eg, num_frames, img) --> (num_eg*num_frames, num_frames, img)
	labels: (num_eg, num_frames) --> (num_eg*num_frames)
	Data is rolled such that the data starts with the frame_num corresponding to label
	
	Data from both the 2 chamber view and 4 chamber view are used
	the frames from the 2 sequences are alternated 

	"""
	def __init__(self,root_dir,labels, seed=123, mask = False, preprocess = False):
		self.root_dir = root_dir
		self.labels = labels
		self.preprocess = preprocess
		if(mask):
			self.list4ch = glob(root_dir+'/la_4ch_masked/*.npy') 
			self.list2ch = glob(root_dir+'/la_2ch_masked/*.npy') 
		else:
			self.list4ch = glob(root_dir+'/la_4ch/*.npy') 
			self.list2ch = glob(root_dir+'/la_2ch/*.npy')

		np.random.seed(seed)

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
		
		filename4ch = self.list4ch[p_idx]
		filename2ch = self.list2ch[p_idx]

		if(self.preprocess):
			input4ch = np.load(filename4ch)
			input2ch = np.load(filename2ch)

			temp4ch = np.zeros(input4ch.shape)
			series4ch = np.zeros(input4ch.shape)
			# min-max normalization ( to apply z -normalization? )
			for frame_num in range(series4ch.shape[0]):
				temp4ch[frame_num,:,:] = cv2.normalize(input4ch[frame_num,:,:], None, 0, 255, cv2.NORM_MINMAX)
			# histogram equalization
			temp4ch = np.uint8(temp4ch)
			clahe = cv2.createCLAHE(clipLimit=0.02)
			for frame_num in range(series4ch.shape[0]):
				series4ch[frame_num,:,:] = clahe.apply(temp4ch[frame_num,:,:])

			temp2ch = np.zeros(input2ch.shape)
			series2ch = np.zeros(input2ch.shape)
			# min-max normalization ( to apply z -normalization? )
			for frame_num in range(series4ch.shape[0]):
				temp2ch[frame_num,:,:] = cv2.normalize(input2ch[frame_num,:,:], None, 0, 255, cv2.NORM_MINMAX)
			# histogram equalization
			temp2ch = np.uint8(temp2ch)
			clahe = cv2.createCLAHE(clipLimit=0.02)
			for frame_num in range(series2ch.shape[0]):
				series2ch[frame_num,:,:] = clahe.apply(temp2ch[frame_num,:,:])
		else:
			series4ch = np.load(filename4ch)
			series2ch = np.load(filename2ch)

		series4ch = series4ch.astype(float) # type float64	
		series4ch = np.roll(series4ch,-frame_num,axis=0) 
		if (series4ch.shape[1]<series4ch.shape[2]):
			series4ch = series4ch.transpose([0,2,1])

		n_frames, m, n = series4ch.shape
		if(m<224):
			pad_size = (( 225 - m ) // 2 ) 
			series4ch = np.pad(series4ch,((0,0),(pad_size,pad_size),(0,0)),'minimum')
		if(n<224):
			pad_size = (( 225 - n ) // 2 ) 
			series4ch = np.pad(series4ch,((0,0),(0,0),(pad_size,pad_size)),'minimum')	
		
		#series4ch = np.expand_dims(series4ch,1) # (50,1,224,224)
		#series4ch = np.concatenate((series4ch,series4ch,series4ch),axis=1)
	
		series2ch = series2ch.astype(float) # type float64	
		series2ch = np.roll(series2ch,-frame_num,axis=0)
		if (series2ch.shape[1]<series2ch.shape[2]):
			series2ch = series2ch.transpose([0,2,1])

		n_frames, m, n = series2ch.shape
		if(m<224):
			pad_size = (( 225 - m ) // 2 ) 
			series2ch = np.pad(series2ch,((0,0),(pad_size,pad_size),(0,0)),'minimum')
		if(n<224):
			pad_size = (( 225 - n ) // 2 ) 
			series2ch = np.pad(series2ch,((0,0),(0,0),(pad_size,pad_size)),'minimum')	
		
		#series2ch = np.expand_dims(series2ch,1) # (50,1,224,224)
		#series2ch = np.concatenate((series2ch,series2ch,series2ch),axis=1)
	
		#series =np.stack((series4ch,series2ch),0) # (2, 50, 3, 224, 224)
		#series = series.transpose([1,0,2,3,4]) # ( 50, 2, 3, 224, 224)
		#n_frames, n_channel, m, n = series4ch.shape
		#series = np.reshape(series,(2*n_frames,3,m,n))

		series =np.stack((series4ch,series2ch),0) # ( 2, 50, 224, 224 )
		series = series.transpose([1,0,2,3]) # ( 50, 2, 224, 224 )
		n_frames, m, n = series4ch.shape
		series = np.reshape(series,(2*n_frames,m,n)) # ( 100, 224, 224 )

		series = np.expand_dims(series,1) # (100,1,224,224)
		series = np.concatenate((series,series,series),axis=1) # (100,3,224,224)
	
		return (series, label)
