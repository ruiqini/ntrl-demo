import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable, grad
#import open3d as o3d

class _numpy2dataset(torch.utils.data.Dataset):
    def __init__(self, points, speed, normal):
        # Creating identical pairs
        points    = Variable(Tensor(points))
        speed  = Variable(Tensor(speed))
        normal  = Variable(Tensor(normal))
        self.data=torch.cat((points,speed,normal),dim=1)
        #self.grid  = Variable(Tensor(grid))

    def send_device(self,device):
        self.data    = self.data.to(device)

    def __getitem__(self, index):
        data = self.data[index]
        #print(index)
        return data, index
    def __len__(self):
        return self.data.shape[0]

def Database(PATH):
    
    #try:
    points = np.load('{}/sampled_points.npy'.format(PATH))#[:100000,:]
    speed = np.load('{}/speed.npy'.format(PATH))#[:100000,:]
    normal = np.load('{}/normal.npy'.format(PATH))#[:100000,:]
    #occupancies = np.unpackbits(np.load('{}/voxelized_point_cloud_128res_20000points.npz'.format(PATH))['compressed_occupancies'])
    #input = np.reshape(occupancies, (128,)*3)
    #grid = np.array(input, dtype=np.float32)
    #print(tau.min())
    #p0 = np.random.rand(100000,2)-0.5
    #s0 = sdf(p0)
    #p1 = np.random.rand(100000,2)-0.5
    #s1 = sdf(p1)
    #points = np.concatinate((p0,p1),axis=1)
    #speed = np.concatinate((s0,s1),axis=1)
    print(points.shape)
    print(speed.shape)
    print(normal.shape)
        #points[:,2:]=0
        #speed[:,1:]=1
    #except ValueError:
    #    print('Please specify a correct source path, or create a dataset')
    rows=points.shape[0]


    print(points.shape,speed.shape)
    #print(np.shape(grid))
    #print(XP.shape,YP.shape)
    database = _numpy2dataset(points,speed,normal)
    #database = _numpy2dataset(XP,YP)
    return database





