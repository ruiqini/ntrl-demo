import matplotlib
import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch import Tensor
from torch.nn import Conv3d
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
#from EikoNet import database as db
from models import data_mlp as db
from models.metric_arm import model_network_metric as model_network
#from models import model_network_lowrank as model_network

#from models import model_network_fixgoal as model_network

from models.metric_arm import model_function_metric as model_function

import copy

import matplotlib
import matplotlib.pylab as plt

import pickle5 as pickle 

from timeit import default_timer as timer

torch.backends.cudnn.benchmark = True

import os
from datetime import datetime, timedelta


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class Model():
    def __init__(self, ModelPath, DataPath, dim, source, device='cpu'):

        # ======================= JSON Template =======================
        self.Params = {}
        self.Params['ModelPath'] = ModelPath
        self.Params['DataPath'] = DataPath
        self.dim = dim
        

        self.source = source

        current_time = datetime.utcnow()-timedelta(hours=5)
        
        #print(DataPath.split('/'))
        
        self.folder = self.Params['ModelPath']+"/"+DataPath.split('/')[-2]+'_'+current_time.strftime("%m_%d_%H_%M")
        # Pass the JSON information
        self.Params['Device'] = device
        self.Params['Pytorch Amp (bool)'] = False

        self.Params['Network'] = {}
        self.Params['Network']['Normlisation'] = 'OffsetMinMax'

        self.Params['Training'] = {}
        self.Params['Training']['Number of sample points'] = 2e5
        self.Params['Training']['Batch Size'] = 2000
        self.Params['Training']['Validation Percentage'] = 10
        self.Params['Training']['Number of Epochs'] = 5000
        self.Params['Training']['Resampling Bounds'] = [0.1, 0.9]
        self.Params['Training']['Print Every * Epoch'] = 1
        self.Params['Training']['Save Every * Epoch'] = 100
        self.Params['Training']['Learning Rate'] = 1e-3#5e-5
        self.Params['Training']['Random Distance Sampling'] = True
        self.Params['Training']['Use Scheduler (bool)'] = False

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss = []

    def train(self):
        
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        import shutil
        #import os

        # Specify the source and destination folders
        source_folder = './models/metric_arm'
        destination_folder = self.folder+'/models'
        os.makedirs(destination_folder, exist_ok=True)

        # Iterate over all the files in the source folder
        for filename in os.listdir(source_folder):
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            
            # Copy the file to the destination folder
            if os.path.isfile(source_file):
                shutil.copy2(source_file, destination_file)

        print("Files copied successfully!")
        # Initialising the network
        #self._init_network()
        self.B = 0.2*torch.normal(0,1,size=(128,self.dim))
        #torch.save(B, self.Params['ModelPath']+'/B.pt')
        freq_bands = 0.5**(torch.linspace(0, 8, 128)) 

        # bvals = 2**np.linspace(-1,1,128//3) #- 1
        # bvals = np.reshape(np.eye(3)*bvals[:,None,None], [len(bvals)*3, 3])
        # print(bvals)
        # #rot = np.array([[(2**.5)/2,-(2**.5)/2,0],[(2**.5)/2,(2**.5)/2,0],[0,0,1]])
        # #bvals = bvals @ rot.T
        # #rot = np.array([[1,0,0],[0,(2**.5)/2,-(2**.5)/2],[0,(2**.5)/2,(2**.5)/2]])
        # #bvals = bvals @ rot.T

        # print(bvals.shape)
        
        #self.B = torch.tensor(bvals).cuda().float()#freq_bands.unsqueeze(1).repeat(1,3)
        #print(self.B)
        self.network = model_network.NN(self.Params['Device'],self.dim, self.B)
        self.network.apply(self.network.init_weights)
        #self.network.float()
        self.network.to(self.Params['Device'])
        # Defining the optimization scheme

        self.function = model_function.Function(self.folder, self.Params['Device'],self.network,self.dim)

        print('train')
        
        #self.load('./Experiments/Gib/Model_Epoch_05000_ValLoss_6.403462e-03.pt')

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.Params['Training']['Learning Rate'],weight_decay=0.5)
        #self.optimizer = torch.optim.LBFGS(
        #    self.network.parameters(), lr=self.Params['Training']['Learning Rate'])
        if self.Params['Training']['Use Scheduler (bool)'] == True:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
            #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000,2000], gamma=0.5)
        # Creating a sampling dataset
        self.dataset = db.Database(self.Params['DataPath'])

        len_dataset = len(self.dataset)
        n_batches = int(len(self.dataset) /
                        int(self.Params['Training']['Batch Size']) + 1)
        training_start_time = time.time()

        # --------- Splitting the dataset into training and validation -------
        indices = list(range(int(len_dataset)))
        #validation_idx = np.random.choice(indices, size=int(
        #    len_dataset*(self.Params['Training']['Validation Percentage']/100)), replace=False)
        #train_idx = list(set(indices) - set(validation_idx))
        train_idx = list(set(indices))
        #validation_sampler = SubsetRandomSampler(validation_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        '''
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.Params['Training']['Batch Size']),
            num_workers = 0,
            shuffle=True)
        '''
        #'''
        dataloader = FastTensorDataLoader(self.dataset.data, 
                    batch_size=int(self.Params['Training']['Batch Size']), 
                    shuffle=True)
        speed = self.dataset.data[:,2*self.dim:]

        Lambda = torch.zeros_like(self.dataset.data[:,:2*self.dim]).cuda().detach()
        print(speed.min())
        #'''
        '''
        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.Params['Training']['Batch Size']),
            sampler=train_sampler,
        )
        '''
        weights = Tensor(torch.ones(len(self.dataset))).to(
                        torch.device(self.Params['Device']))
        PATH = self.Params['ModelPath']+'/check.pt'
        beta = 1.0
        prev_diff = 1.0
        current_diff = 1000.0
        step = -1000.0/4000.0
        #step = 1.0
        tt =time.time()

        current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
        current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
        #p=(torch.rand((5,6))-0.5).cuda()
        prev_state_queue = []
        prev_optimizer_queue = []
        for epoch in range(1, self.Params['Training']['Number of Epochs']+1):
            t_0=time.time()
            
            print_every = 1
            start_time = time.time()
            running_sample_count = 0
            total_train_loss = 0
            total_val_loss = 0
            total_diff=0
            '''
            if epoch%100==0:
                dataloader = FastTensorDataLoader(self.dataset.data, 
                    batch_size=int(self.Params['Training']['Batch Size']), 
                    shuffle=True)
            '''
            

            alpha = 0.8#0.95#0.95#min(max(0.5,0.5+0.5*step),1.05)
            alpha = 1.0#min(max(0.5,0.5+0.5*step),1)
            step+=1.0/4000/((int)(epoch/4000)+1.)
            gamma=0.001#max((4000.0-epoch)/4000.0/20,0.001)
            mu = 10

            prev_state_queue.append(current_state)
            prev_optimizer_queue.append(current_optimizer)
            if(len(prev_state_queue)>5):
                prev_state_queue.pop(0)
                prev_optimizer_queue.pop(0)
            
            current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
            current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
            
            aa = np.clip(1e-3*(1-(epoch-500)/1000.), a_min=5e-4, a_max=1e-3) 
            #print(aa)
            self.optimizer.param_groups[0]['lr']  = 5e-4#aa#np.clip(1e-3*(1-(epoch-8000)/1000.), a_min=5e-4, a_max=1e-3) 
            #self.optimizer.param_groups[0]['lr']  = np.clip(1e-3*(1-(epoch-500)/500.), a_min=5e-4, a_max=1e-3) 
            #1e-3 works
            prev_lr = self.optimizer.param_groups[0]['lr'] 
            t_1=time.time()
            #print(t_1-t_0)
            t_0=time.time()
            #print(prev)
            prev_diff = current_diff
            iter=0
            while True:
                total_train_loss = 0
                total_diff = 0
                #for i in range(10):
                ii = 0
                for i, wholedata in enumerate(dataloader,0):#train_loader_wei,dataloader
                    #print('----------------- Epoch {} - Batch {} --------------------'.format(epoch,i))
                    if ii>4:
                        break
                    ii = ii+1
                    t0 = time.time()
    
                    data = wholedata[0].to(self.Params['Device'])
                    #indexbatch = wholedata[1].to(self.Params['Device'])
                    #print(indexbatch)
                    #ind, indexbatch = data
                    #print(wholedata[1])
                    points = data[:,:2*self.dim]#.float()#.cuda()
                    speed = data[:,2*self.dim:2*self.dim+2]#.float()#.cuda()
                    normal = data[:,2*self.dim+2:]
                    #print(speed.shape)
                    speed = speed*speed*(2-speed)*(2-speed)

                    speed=alpha*speed+1-alpha

                    loss_value, loss_n, wv = self.function.Loss(points, speed, normal, beta, gamma, epoch)
                    
                    #Lambda[indexbatch,:] = Lamb
                    t1 = time.time()
                    #print(t1-t0)
                    
                    t0 = time.time()
                    loss_value.backward()

                    # Update parameters
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    total_train_loss += loss_value.item()
                    total_diff += loss_n.item()
                    # total_train_loss = 0
                    # total_diff = 0
                    # def closure():
                    #     self.optimizer.zero_grad()
                    #     loss_value, loss_n, wv = self.function.Loss(points, speed, normal, beta, gamma, epoch)
                        
                    #     total_train_loss = loss_value.item()#.clone().detach()
                    #     total_diff = loss_n.item()#.clone().detach()

                    #     loss_value.backward()
                    #     return loss_value
                    # self.optimizer.step(closure)

                    #print('')
                    #print(loss_value.shape)
                    
                    t1 = time.time()
                    #print(total_train_loss)

                    #print(t1-t0)
                    #print('')
                    #weights[indexbatch] = wv
                    
                    del points, speed, loss_value, loss_n#, Lamb#,indexbatch
                
                
                total_train_loss /= 4#len(dataloader)#dataloader train_loader
                total_diff /= 4#len(dataloader)#dataloader train_loader

                #total_train_loss /= len(dataloader)#dataloader train_loader
                #total_diff /= len(dataloader)#dataloader train_loader

                current_diff = total_diff
                diff_ratio = current_diff/prev_diff
        
                if (diff_ratio < 1.2 and diff_ratio > 0):#1.5
                    #self.optimizer.param_groups[0]['lr'] = prev_lr 
                    break
                else:
                    
                    iter+=1
                    with torch.no_grad():
                        random_number = random.randint(0, len(prev_state_queue)-1)
                        self.network.load_state_dict(prev_state_queue[random_number], strict=True)
                        self.optimizer.load_state_dict(prev_optimizer_queue[random_number])
   
                    print("RepeatEpoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                        epoch, total_diff, alpha))
                
                
            #'''
            self.total_train_loss.append(total_train_loss)
            
            beta = 1.0/total_diff
            
            t_1=time.time()
            #print(t_1-t_0)

            #del train_loader_wei, train_sampler_wei

            if self.Params['Training']['Use Scheduler (bool)'] == True:
                self.scheduler.step(total_train_loss)

            t_tmp = tt
            tt=time.time()
            #print(tt-t_tmp)
            #print('')
            if epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                with torch.no_grad():
                    #print("Epoch = {} -- Training loss = {:.4e} -- Validation loss = {:.4e}".format(
                    #    epoch, total_train_loss, total_val_loss))
                    print("Epoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                        epoch, total_diff, alpha))

            if (epoch % self.Params['Training']['Save Every * Epoch'] == 0) or (epoch == self.Params['Training']['Number of Epochs']) or (epoch == 1):
                self.function.plot(epoch,total_diff,alpha, self.source)
                with torch.no_grad():
                    self.save(epoch=epoch, val_loss=total_diff)
        
        #points = self.dataset.data[:,:2*self.dim]
        #T, w, Xp = self.network.out(points.cuda())
        #np.save(self.Params['DataPath']+'/value.npy',T.detach().cpu().numpy())

    def save(self, epoch='', val_loss=''):
        '''
            Saving a instance of the model
        '''
        torch.save({'epoch': epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'B_state_dict':self.B,
                    'train_loss': self.total_train_loss,
                    'val_loss': self.total_val_loss}, '{}/Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(self.folder, str(epoch).zfill(5), val_loss))

    def load(self, filepath):
        #B = torch.load(self.Params['ModelPath']+'/B.pt')
        
        checkpoint = torch.load(
            filepath, map_location=torch.device(self.Params['Device']))
        self.B = checkpoint['B_state_dict']

        self.network = model_network.NN(self.Params['Device'],self.dim,self.B)

        self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.network.to(torch.device(self.Params['Device']))
        self.network.float()
        self.network.eval()

        self.function = model_function.Function(self.folder, self.Params['Device'],self.network,self.dim)


        
    def load_pretrained_state_dict(self, state_dict):
        own_state=self.state_dict

