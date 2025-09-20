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
#from models import data_mlp as db
#from models import model_network_one as model_network
import igl 
import copy

import matplotlib
import matplotlib.pylab as plt

from timeit import default_timer as timer

#import torch_kdtree

torch.backends.cudnn.benchmark = True


class Function():
    def __init__(self, path, device, network, dim):

        # ======================= JSON Template =======================
        self.path = path
        self.device = device
        self.dim = dim

        self.network = network

        # Pass the JSON information
        #self.Params['Device'] = device

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss = []
        #input_file = "datasets/gibson/Cabin/mesh_z_up_scaled.off"
        #self.kdtree, self.v_obs, self.n_obs = self.pc_kdtree(input_file)

        self.alpha = 1.025
        limit = 0.5
        self.margin = limit/15.0
        self.offset = self.margin/10.0 
    
    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 

        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x                                                                                                    
    
    def Loss(self, points, Yobs, normal, beta, gamma, epoch):
        
        start=time.time()
        tau, w, Xp = self.network.out(points)
        dtau = self.gradient(tau, Xp)
        end=time.time()
        
        #print(end-start)

        start=time.time()
        
        
        #tau, dtau, Xp = self.network.out_grad(points)
        
        end=time.time()
        #print(end-start)
        #print(dtau)

        #print(end-start)
        #print('')
        #y-x
        #D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        D = torch.norm(Xp[:,self.dim:]-Xp[:,:self.dim], p=2, dim =1)
        
        
        DT0 = dtau[:,:self.dim]
        DT1 = dtau[:,self.dim:]
        
        
        S0 = torch.einsum('ij,ij->i', DT0, DT0)
        S1 = torch.einsum('ij,ij->i', DT1, DT1)

        td_weight = 1e-3
        with torch.no_grad():

            length0 = (0.02)/(Yobs[:,0]).unsqueeze(1)#5*torch.rand(Yobs.shape[0],1).cuda()
            Dir0 = length0*(DT0*Yobs[:,0].unsqueeze(1)**2).clone().detach()  
            #Dir1 = 0.03*(DT1/S1.unsqueeze(1)).clone().detach()  
            Xp_new0 = Xp.clone().detach()  
            
            Xp_new0[:,:self.dim] = Xp_new0[:,:self.dim] - Dir0

            tau_new0, w, Xp_new0 = self.network.out(Xp_new0)
            #tau_new1, w, Xp_new1 = self.network.out(Xp_new1)
            tau_new1 = length0#*1/Yobs[:,0].unsqueeze(1)
            del Xp_new0, Dir0#Xp_new1

        tau_loss0 = td_weight*((tau-(tau_new0+tau_new1))**2).squeeze()
        #(1.01-Yobs[:,0])*td_weight*
        #(1.4-Yobs[:,0])*

        with torch.no_grad():

            length1 = (0.02)/(Yobs[:,1]).unsqueeze(1)#5*torch.rand(Yobs.shape[0],1).cuda()
            Dir1 = length1*(DT1*Yobs[:,1].unsqueeze(1)**2).clone().detach()  
            Xp_new0 = Xp.clone().detach()  
            
            #Xp_new0[:,:self.dim] = Xp_new0[:,:self.dim] - Dir0
            Xp_new0[:,self.dim:] = Xp_new0[:,self.dim:] - Dir1
            #Xp_new[:,self.dim:]+=0.04*DT1/S1

            tau_new0, w, Xp_new0 = self.network.out(Xp_new0)
            #tau_new1, w, Xp_new1 = self.network.out(Xp_new1)
            tau_new1 = length1#*1/Yobs[:,1].unsqueeze(1)
            del Xp_new0, Dir1#Xp_new1

        tau_loss1 = td_weight*((tau-(tau_new0+tau_new1))**2).squeeze()
        
        where_d0 = (tau[:,0] < length0.squeeze())
        where_d1 = (tau[:,0] < length1.squeeze())
        tau_loss0[where_d0] = 0 
        tau_loss1[where_d1] = 0 

        tau_loss = tau_loss0+tau_loss1
        #'''

        Ypred0 = torch.sqrt(S0+1e-8)#torch.sqrt
        Ypred1 = torch.sqrt(S1+1e-8)#torch.sqrt


        Ypred0_visco = Ypred0
        Ypred1_visco = Ypred1

        sq_Ypred0 = (Ypred0_visco)#+gamma*lap0
        sq_Ypred1 = (Ypred1_visco)#+gamma*lap1


        sq_Yobs0 = (Yobs[:,0])#**2
        sq_Yobs1 = (Yobs[:,1])#**2

        #loss0 = (sq_Yobs0/sq_Ypred0+sq_Ypred0/sq_Yobs0)#**2#+gamma*lap0
        #loss1 = (sq_Yobs1/sq_Ypred1+sq_Ypred1/sq_Yobs1)#**2#+gamma*lap1
        l0 = ((sq_Yobs0*(sq_Ypred0)))
        l1 = ((sq_Yobs1*(sq_Ypred1)))
        
        l0_2 = (torch.sqrt(l0))#**(1/4)
        l1_2 = (torch.sqrt(l1))#**(1/4)    

        #w_num = w.clone().detach()
        loss_weight = 1e-2
        loss0 = loss_weight*(l0_2-1)**2  #/scale#+relu_loss0#**2#+gamma*lap0#**2
        loss1 = loss_weight*(l1_2-1)**2  #/scale#+relu_loss1#**2#+gamma*lap1#**2
        
        T = tau[:,0] #* torch.sqrt(T0)
        diff = loss0 + loss1 

        normal_weight = 1e-3

        normal0 = normal[:,:self.dim]
        normal1 = normal[:,self.dim:]
        #print(normal0)
        #print(DT0)
        n_loss0 = (1.01-Yobs[:,0].unsqueeze(1))*(Yobs[:,0].unsqueeze(1)*DT0+normal0)**2
        n_loss1 = (1.01-Yobs[:,1].unsqueeze(1))*(Yobs[:,1].unsqueeze(1)*DT1+normal1)**2
        #print(n_loss0.shape)
        #n_loss = normal_weight*torch.sum(n_loss0,dim=1)
        n_loss = normal_weight*(torch.sum(n_loss0,dim=1)+torch.sum(n_loss1,dim=1))
        
        #print(n_loss0)
        #T_num = T.clone.detach()+weight
        #
        #where = T>1
        #loss_td, loss_mpc = self.MPPIPlanner(Xp[where])
        #loss0 +
        #print(T)
        para = 0.5#max(0.5-epoch*0.001,0.5)
        #
        loss_n = (torch.sum((diff +n_loss +tau_loss)*torch.exp(-0.5*T)))/Yobs.shape[0]#*torch.exp(-para*T)
        
        loss = beta*loss_n #+ 1e-4*(reg_tau)
        
        return loss, loss_n, diff

    def TravelTimes(self, Xp):
     
        tau, w, coords = self.network.out(Xp)        

        TT = tau[:,0] #* torch.sqrt(T0)
            
        return TT
    
    def Tau(self, Xp):
        Xp = Xp.to(torch.device(self.device))
     
        tau, w, coords = self.network.out(Xp)

        return (tau-1)/100

    def Speed(self, Xp):

   

        Xp = Xp.to(torch.device(self.device))

        tau, w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        #Xp.requires_grad_()
        #tau, dtau, coords = self.network.out_grad(Xp)
        
        
        #D = Xp[:,self.dim:]-Xp[:,:self.dim]
        #T0 = torch.einsum('ij,ij->i', D, D)

        DT0 = dtau[:,:self.dim]

        #T3    = tau[:,0]**2



        #TT = LogTau * torch.sqrt(T0)

        #print(tau.shape)

        #print(T02.shape)
        #T1    = T0*torch.einsum('ij,ij->i', DT0, DT0)
        #T2    = -2*tau[:,0]*torch.einsum('ij,ij->i', DT0, D)
        
        
        S = torch.einsum('ij,ij->i', DT0, DT0)

        Ypred = 1/torch.sqrt(S)
        
        del Xp, tau, dtau#, T0#, T1, T2, T3
        return Ypred
    
    def StartSpeed(self, Xp):

        batch_size = 5000
        Ypred_list = []
        #print(Xp.shape[0])
        for i in range(math.ceil(Xp.shape[0]/batch_size)):

            Xp_tmp = Xp[i*batch_size:min((i+1)*batch_size,Xp.shape[0]),:]
            tau, w, Xp_tmp = self.network.out(Xp_tmp)
            dtau = self.gradient(tau, Xp_tmp)
            #Xp.requires_grad_()        
            
            D = Xp_tmp[:,self.dim:]-Xp_tmp[:,:self.dim]
            T0 = torch.einsum('ij,ij->i', D, D)

            DT0 = dtau[:,:self.dim]

            T3    = tau[:,0]**2

            T01    = T0*torch.einsum('ij,ij->i', DT0, DT0)
            T02    = -2*tau[:,0]*torch.einsum('ij,ij->i', DT0, D)
            
            
            S0 = (T01+T02+T3)
            

            Ypred = 1/torch.sqrt(S0)
            Ypred_list.append(Ypred.clone())
            del Xp_tmp, w, tau, dtau, D, DT0, T0, T01, T02, T3, S0, Ypred
        del Xp
        Ypred = torch.cat(Ypred_list,dim=0)
        del Ypred_list
        return Ypred
   
    def Gradient(self, Xp):
        #Xp = Xp.to(torch.device(self.device))
       
        #Xp.requires_grad_()
        
        #tau, dtau, coords = self.network.out_grad(Xp)
        #print(Xp.shape)
        tau, w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        
        #D = Xp[:,self.dim:]-Xp[:,:self.dim]
        #T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D)).view(-1,1)

        #A = T0*dtau[:,:self.dim]
        #B = tau/T0*D
        

        Ypred0 = -dtau[:,:self.dim]#-A+B
        #print(Ypred0.shape)
        Spred0 = torch.norm(Ypred0,dim=1).view(-1,1)
        Ypred0 = 1/Spred0**2*Ypred0

        Ypred1 = -dtau[:,self.dim:]#-A-B
        Spred1 = torch.norm(Ypred1,dim=1).view(-1,1)

        Ypred1 = 1/Spred1**2*Ypred1

        #print(Ypred0.shape)
        #print(Ypred1.shape)
        
        return torch.cat((Ypred0, Ypred1),dim=1)
    

    def pc_kdtree(self, input_file):
        v, f = igl.read_triangle_mesh(input_file)
        z = np.zeros(3)
        #n = igl.per_face_normals(v, f, z)
        n = igl.per_vertex_normals(v, f, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA)

        bary, FI, _ = igl.random_points_on_mesh(50000, v, f)

        face_verts = v[f[FI], :]
        v_obs = np.sum(bary[...,np.newaxis] * face_verts, axis=1)
        #v_obs = bary * v[f[FI,:]]

        face_norms = n[f[FI], :]
        #n_obs = n[FI,:]
        n_obs = np.sum(bary[...,np.newaxis] * face_norms, axis=1)

        v_obs = torch.tensor(v_obs, dtype=torch.float32, device='cuda')
        n_obs = torch.tensor(n_obs, dtype=torch.float32, device='cuda')

        kdtree = torch_kdtree.build_kd_tree(v_obs)
        return kdtree, v_obs, n_obs

    def speed_kdtree(self, query_points):
        if self.dim==2:
            query_points = torch.cat((query_points,torch.zeros((query_points.shape[0],1)).cuda()),dim=1)
        dists, inds = self.kdtree.query(query_points, nr_nns_searches=1)
        dists = dists.squeeze()
        inds = inds.squeeze()
        unsigned_distance = torch.sqrt(dists).view(-1)
        #print(closest_faces.shape)
        #print(normal_obs)
        #print(dists.shape)
        #print(inds.shape)
        normal_face = self.n_obs[inds,:].view(-1,3)
        #print(closest_points.shape)
        normal = query_points-self.v_obs[inds,:]
        normal = torch.nn.functional.normalize(normal,dim=1).view(-1,3)
        #print(v_obs[inds,:])
        #print(normal.shape)
        #print(normal_face.shape)
        dot = torch.einsum('ij,ij->i', normal, normal_face)
        where_d          =   (dot<np.sqrt(2)/2)
        #print(unsigned_distance)
        if unsigned_distance[where_d].shape[0]>0:
            unsigned_distance[where_d] *=-1
        unsigned_distance = unsigned_distance-self.offset
        #print(unsigned_distance)
        speed = torch.clamp(unsigned_distance/self.margin , self.offset/self.margin, 1)

        speed = speed*speed*(2-speed)*(2-speed)
        speed = self.alpha*speed+1-self.alpha
        return speed
    
    def GradientPlanner(self, Xp):
        XP = Xp.clone()
        dis=torch.norm(XP[:,self.dim:]-XP[:,:self.dim],dim=1)
        where = dis>0.03
        
        batch = XP.shape[0]
        loss_td = torch.zeros((batch)).cuda()
        loss_mpc = torch.zeros((batch)).cuda()
        if batch>0:
            steps = 5
            sample_num = 30
            horizon = 3

            batch_path_cost = torch.zeros((batch)).cuda()
            #path = []
            #print("Time:",womodel.function.TravelTimes(XP))
            #path.append(XP.clone())
            for iter in range(steps):
                #print(iter)

                if(XP[where].shape[0]==0):
                    break
                
                gradient = self.Gradient(XP[where].clone())

                XP[where,:self.dim] = XP[where,:self.dim] + 0.02*gradient[:,:self.dim]
                with torch.no_grad():
                    #print(XP[where,0:3].shape)
                    speed_where = self.speed_kdtree(XP[where,:self.dim])
                #path.append(XP.clone())
                batch_path_cost[where] += 0.02*1/speed_where
                #print(dP_prior[where].shape)
                
                dis=torch.norm(XP[:,self.dim:]-XP[:,:self.dim],dim=1)
                where = dis>0.03
                del dis
                #print(XP)
                #point0.append(XP[:,0:3].clone())
                #point1.append(XP[:,3:6].clone())
            
            with torch.no_grad():
                batch_terminal_cost = self.TravelTimes(XP).detach()    
                batch_path_cost = batch_path_cost.detach()  

            loss_td = (batch_terminal_cost+batch_path_cost-self.TravelTimes(Xp))**2
            #XP[:,3:6] = Xp[:,0:3] #recover init
            #loss_mpc = (self.TravelTimes(XP)-batch_terminal_cost)**2
        return loss_td, loss_mpc

    def MPPIPlanner(self, Xp):
        XP = Xp.clone()
        dis=torch.norm(XP[:,self.dim:]-XP[:,:self.dim],dim=1)
        where = dis>0.03
        
        batch = XP.shape[0]
        loss_td = torch.zeros((batch)).cuda()
        loss_mpc = torch.zeros((batch)).cuda()
        if batch>0:
            steps = 5
            sample_num = 30
            horizon = 3

            dP_prior = torch.zeros((batch,self.dim)).cuda()
            batch_path_cost = torch.zeros((batch)).cuda()
            #path = []
            #print("Time:",womodel.function.TravelTimes(XP))
            #path.append(XP.clone())
            for iter in range(steps):
                #print(iter)

                if(XP[where].shape[0]==0):
                    break
                
                XP_tmp = XP[where].clone()#[:,0:3]
                curr_batch = XP_tmp.shape[0]
                XP_tmp = XP_tmp.unsqueeze(1).unsqueeze(1).repeat(1,sample_num,horizon,1)

                cost_path = 0
                #dP_list = []
                dP = 0.02 * torch.normal(0,1,size=(curr_batch, sample_num, 1, self.dim),dtype=torch.float32, device='cuda') \
                    +0.01 * torch.normal(0,1,size=(curr_batch, sample_num, horizon, self.dim),dtype=torch.float32, device='cuda')
                #if iter>0:
                #    dP = 0.5*dP
                #print(dP.shape)
                #print(dP_prior[where].shape)
                dP = dP + dP_prior[where].unsqueeze(1).unsqueeze(1)
                #dP = 0.02 * torch.nn.functional.normalize(dP,dim=3)
                dP_norm = torch.norm(dP,dim=3)
                dP = dP/(torch.clamp(dP_norm.unsqueeze(3),min=0.02)/0.02)
                #print(dP)
                dP_cumsum = torch.cumsum(dP, dim=2)
                #dP0 = dP[:,:,0,:]
                #del dP
                #print(XP_tmp.shape)
                XP_tmp[...,0:self.dim] = XP_tmp[...,0:self.dim]+dP_cumsum
                del dP_cumsum

                XP_path = XP_tmp.view(curr_batch*sample_num*horizon,2*self.dim)
                #speed = womodel.function.StartSpeed(XP_path)
                with torch.no_grad():
                    speed = self.speed_kdtree(XP_path[:,:self.dim])
                
                del XP_path
                #print(speed)
                speed = speed.view(curr_batch, sample_num, horizon)
                #print(speed)
                #del XP_list
                #0.02*
                path_cost = torch.sum(torch.clamp(dP_norm,max=0.02)*1/speed,dim=2)
                #cost_path = 0.02*torch.sum(1/speed,dim=2)
                #print(cost_path.min())
                del speed#, dP_norm
                with torch.no_grad():
                    terminal_cost = self.TravelTimes(XP_tmp[:,:,-1,:].view(curr_batch*sample_num,2*self.dim))
                #print(terminal_cost.shape)
                del XP_tmp
                cost = path_cost+terminal_cost.view(curr_batch, sample_num)
                del path_cost, terminal_cost
                #print('T',womodel.function.TravelTimes(XP))
                #print(cost.min())
                #print(terminal_cost.shape)
                #print(cost.shape)
                weight = torch.softmax(-50*cost, dim=1)
                del cost
                #print(weight.unsqueeze(1).shape)
                #print(dP[:,:,0,:].shape)
                #print(dP.shape)
                #dP_prior = (weight.unsqueeze(0)@dP[:,:,0,:]) + 0.002*torch.normal(0,1,size=(batch, 1, 3),dtype=torch.float32, device='cuda')
                dP_prior[where] = torch.einsum('bij,bjk->bik', weight.unsqueeze(1), dP[:,:,0,:]).squeeze() #\
                #+ 0.001*torch.normal(0,1,size=(curr_batch, 3),dtype=torch.float32, device='cuda')
                del weight

                XP[where,0:self.dim] = dP_prior[where] + XP[where,0:self.dim]
                with torch.no_grad():
                    #print(XP[where,0:3].shape)
                    speed_where = self.speed_kdtree(XP[where,0:self.dim])
                #path.append(XP.clone())
                batch_path_cost[where] += torch.norm(dP_prior[where],dim=1)*1/speed_where
                #print(dP_prior[where].shape)
                
                dis=torch.norm(XP[:,self.dim:]-XP[:,:self.dim],dim=1)
                where = dis>0.03
                del dis
                #print(XP)
                #point0.append(XP[:,0:3].clone())
                #point1.append(XP[:,3:6].clone())
            
            with torch.no_grad():
                batch_terminal_cost = self.TravelTimes(XP).detach()    
                batch_path_cost = batch_path_cost.detach()  

            loss_td = (batch_terminal_cost+batch_path_cost-self.TravelTimes(Xp))**2
            XP[:,self.dim:] = Xp[:,:self.dim] #recover init
            loss_mpc = (self.TravelTimes(XP)-batch_terminal_cost)**2
        return loss_td, loss_mpc
    
    def plot(self,epoch,total_train_loss,alpha,source):
        limit = 0.5#0.5
        xmin     = [-limit,-limit]
        xmax     = [limit,limit]
        spacing=limit/40.0
        X,Y      = np.meshgrid(np.arange(xmin[0],xmax[0],spacing),np.arange(xmin[1],xmax[1],spacing))
        #dims_n = np.setdiff1d([0,1,2],dims)[0]

        size=81

        spacing = 2*limit/(size-1)#2.0#0.4
        X,Y  = np.meshgrid( np.arange(-limit,limit+0.1*spacing,spacing),
                            np.arange(-limit,limit+0.1*spacing,spacing))
        I,J  = np.meshgrid( np.arange(0,size,1),
                            np.arange(0,size,1))
        #print(X)
        #print(X.shape)
        #print(I.shape)
        V = np.zeros((size*size,3))
        V[:,0]=X.flatten()
        V[:,1]=Y.flatten()

        IDX = I*size+J

        F = np.zeros((2*(size-1)*(size-1),3))
        F[:(size-1)*(size-1),0]=IDX[:size-1,:size-1].flatten()
        F[:(size-1)*(size-1),1]=IDX[1:,:size-1].flatten()
        F[:(size-1)*(size-1),2]=IDX[:size-1,1:].flatten()
        F[(size-1)*(size-1):,0]=IDX[1:,1:].flatten()
        F[(size-1)*(size-1):,2]=IDX[1:,:size-1].flatten()
        F[(size-1)*(size-1):,1]=IDX[:size-1,1:].flatten()

        F=F.astype(int)

        Xsrc = source#[0]*self.dim
        #Xsrc[0] = -0.2
        #Xsrc[1] = -0.2
        #Xsrc[2] = -0.02
        #print(self.dim)
        #Xsrc=[-1.2, 0.4-0.5*np.pi, 1.4, 0.2-0.5*np.pi,-0.5,0.9]
        #-1.3, 0.6, 1.1, 0.2,-1.5,0.9
        scale = np.pi/0.5
        
        XP       = np.zeros((len(X.flatten()),2*self.dim))#*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
        XP[:,:self.dim] = Xsrc
        XP[:,self.dim:] = Xsrc

        XP=XP/scale

        XP[:,0]  = X.flatten()
        XP[:,1]  = Y.flatten()
        XP = Variable(Tensor(XP)).to(self.device)

        
        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)#*5
        tau = self.Tau(XP)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        S  = ss.to('cpu').data.numpy().reshape(X.shape)
        TAU = tau.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,S,vmin=0,vmax=1)
        ax.contour(X,Y,TT,np.arange(0,10,0.02), cmap='gist_heat', linewidths=0.5)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.path+"/plots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.jpg",bbox_inches='tight')
        plt.close(fig)
        #print(TT.flatten().shape)
        V[:,2] = TT.flatten()#0.2*
        igl.write_triangle_mesh(self.path+"/plots_TT"+str(epoch)+".obj", V, F) 


        XP       = np.zeros((len(X.flatten()),2*self.dim))#*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
        XP[:,:self.dim] = Xsrc
        XP[:,self.dim:] = Xsrc
        #XP=XP/scale
        XP[:,0]  = X.flatten()
        XP[:,1]  = Y.flatten()
        XP[:,2] = -0.1
        XP = Variable(Tensor(XP)).to(self.device)

        
        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)#*5
        tau = self.Tau(XP)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        S  = ss.to('cpu').data.numpy().reshape(X.shape)
        TAU = tau.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,S,vmin=0,vmax=1)
        ax.contour(X,Y,TT,np.arange(0,10,0.02), cmap='gist_heat', linewidths=0.5)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.path+"/tauplots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.jpg",bbox_inches='tight')

        plt.close(fig)

        V[:,2] = TT.flatten()
        igl.write_triangle_mesh(self.path+"/plots_TAU"+str(epoch)+".obj", V, F) 


         
