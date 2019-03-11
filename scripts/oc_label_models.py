'''
Compares label model performance with vs without temporal modelling
'''
import sys
sys.path.append('../metal')
sys.path.append('../heart-MRI-pytorch')
sys.path.append('../data')
sys.path.append('../../sequential_ws')

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
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from metal.metrics import metric_score
from metal.label_model import LabelModel
from metal.label_model.baselines import MajorityLabelVoter
from metal.analysis import lf_summary, confusion_matrix
from DP.label_model import DPLabelModel, optimize


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


def train_model(args):

    #global args
    #args = parser.parse_args()

	hidden_size = 128 
	num_classes = 2
	encode_dim = 108 # using get_frm_output_size()

	if(torch.cuda.is_available()):
		device = torch.device('cuda:0')
		#device = 'cuda'
	else:
		device = 'cpu'
	
	#print(device)
	L,Y = load_labels(args) 

	# Label Model
	# labelling functions analysis
	print(lf_summary(L["dev"], Y = Y["dev"]))

	# majority vote of LFs
	mv = MajorityLabelVoter(seed=123)
	print('Majority Label Voter Metrics:')
	mv.score((L["dev"], Y["dev"]), metric=['accuracy','precision', 'recall', 'f1'])

	# training label model - no temporal modelling
	label_model = LabelModel(k=num_classes, seed=123)
	label_model.train_model(L["train"], Y["dev"], n_epochs = 500, log_train_every = 50)

	# evaluating label model
	print('Trained Label Model Metrics:')
	label_model.score((L["dev"], Y["dev"]), metric=['accuracy','precision', 'recall', 'f1'])

	# training label model without temporal modelling
	# naive model
	#print(L["train"].todense().shape) # (18850,5)
	#print(L["dev"].todense().shape) # (1500,5)
	#print(Y["dev"].shape) # (1500,)
	m_per_task = L["train"].todense().shape[1] # 5
	MRI_data_naive = {'Li_train': torch.LongTensor(np.array(L["train"].todense().astype(int))),
                'Li_dev': torch.LongTensor(np.array(L["dev"].todense())),
                'R_dev':Y["dev"]}

	MRI_data_naive['class_balance'] = torch.LongTensor([0.5,0.5]).to(device)

	# training naive model 
	naive_model = DPLabelModel(	m=m_per_task, 
								T=1,
								edges=[],
								coverage_sets=[[0,]]*m_per_task,
								mu_sharing=[[i,] for i in range(m_per_task)],
								phi_sharing=[],
								device=device,
								class_balance=MRI_data_naive['class_balance'], 
								seed=0)

	optimize(naive_model, L_hat=MRI_data_naive['Li_train'], num_iter=300, lr=1e-3, momentum=0.8, clamp=True, seed=0)

	# evaluating naive model 
	R_pred = naive_model.predict( MRI_data_naive['Li_dev'] ).data.numpy()
	R_pred = 2 - R_pred
	#print(R_pred)
	#print(MRI_data_naive['R_dev'])

	for metric in ['accuracy', 'f1', 'recall', 'precision']:
		score = metric_score(MRI_data_naive['R_dev'], R_pred, metric)
		print(f"{metric.capitalize()}: {score:.3f}")
			
	# training label model with temporal modelling
	# reshaping dataset
	num_frames = 50
	n_patients_train = round(L["train"].todense().shape[0]/num_frames) #(377)
	n_patients_dev = round(L["dev"].todense().shape[0]/num_frames) #(30)
	Ltrain = np.reshape(np.array(L["train"].todense()),(n_patients_train,num_frames,-1))
	Ldev = np.reshape(np.array(L["dev"].todense()),(n_patients_dev,num_frames,-1))
	Ydev = np.reshape(Y["dev"],(n_patients_dev,num_frames))
	# print(Ltrain.shape) # (377,50,5)
	#print(Ldev.shape) # (30,50,5)
	#print(Ydev.shape) # (30,50)

	# subsampling
	# selecting frames 3,13,23,33,43
	indices = np.linspace(2,42,5).astype(int)
	m_per_task = 5
	T = 5

	Ltrain_small = Ltrain[:,indices,:] # shape (377,5,5)
	Ldev_small = Ldev[:,indices,:] # shape (30,5,5)
	Ydev_small = Ydev[:,indices] # shape (30,5)

	Ltrain_small = np.reshape(Ltrain_small,((n_patients_train*T),m_per_task)) # shape (1885,5)
	Ldev_small = np.reshape(Ldev_small,((n_patients_dev*T),m_per_task)) # shape (150,5)
	Ydev_small = np.reshape(Ydev_small,((n_patients_dev*T),)) # shape (150,)

	MRI_data_temporal = {'Li_train': torch.LongTensor(Ltrain_small).view(n_patients_train, (m_per_task*T)), 
                    'Li_dev': torch.LongTensor(Ldev_small).view(n_patients_dev, (m_per_task*T)), 
                    'R_dev':torch.LongTensor(Ydev_small)[::T]*(2**T-1),
                    'm': m_per_task*T,
                    'T': T }

	MRI_data_temporal['class_balance'] = normalize((MRI_data_temporal['R_dev'].unsqueeze(1)==torch.arange(2**T, device=device).unsqueeze(0)).sum(0).float(), 
                                                dim=0, p=1)

	max_seed = 10
	temporal_models = [None,]*max_seed
	for seed in range(max_seed):
		markov_model = DPLabelModel(m=m_per_task*T, 
									T=T,
									edges=[(i,i+m_per_task) for i in range((T-1)*m_per_task)],
									coverage_sets=[[t,] for t in range(T) for _ in range(m_per_task)],
									mu_sharing=[[t*m_per_task+i for t in range(T)] for i in range(m_per_task)],
									phi_sharing=[[(t*m_per_task+i, (t+1)*m_per_task+i) for t in range(T-1)] for i in range(m_per_task)],
									device=device,
									class_balance=MRI_data_temporal['class_balance'],
									seed=seed)
		optimize(markov_model, L_hat=MRI_data_temporal['Li_train'], num_iter=1000, lr=1e-5, momentum=0.8, clamp=True, 
					verbose=False, seed=seed)
		temporal_models[seed] = markov_model

	for seed, model in enumerate(temporal_models):
		R_pred = model.predict(MRI_data_temporal['Li_dev'].cpu())
		F1 = metric_score(MRI_data_temporal['R_dev'].cpu()>0, R_pred.cpu()>0, 'f1')
		accuracy = metric_score(MRI_data_temporal['R_dev'].cpu(), R_pred.cpu(), 'accuracy')
		print(f"seed={seed}  accuracy={accuracy:.3f}  F1={F1:.3f}")

	

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


