import os 
import glob
import numpy as np
from timeit import default_timer as timer
import igl
import traceback
import math
import torch
import pytorch_kinematics as pk

import math
import matplotlib.pyplot as plt

import torch_kdtree #import build_kd_tree

from torch_IK_UR5 import torch_IK_UR5, transformRobotParameter

from functorch import vmap, jacfwd

def rot_pose(x, a, b, c):

    rot_pose = torch.zeros((x.shape[0],4,4),dtype=torch.float32, device='cuda')
    cos_a = torch.cos(a)
    sin_a = torch.sin(a)
    cos_b = torch.cos(b)
    sin_b = torch.sin(b)
    cos_c = torch.cos(c)
    sin_c = torch.sin(c)

    rot_pose[:,0,0] = cos_b*cos_c
    rot_pose[:,0,1] = sin_a*sin_b*cos_c-cos_a*sin_c
    rot_pose[:,0,2] = cos_a*sin_b*cos_c+sin_a*sin_c

    rot_pose[:,1,0] = cos_b*sin_c
    rot_pose[:,1,1] = sin_a*sin_b*sin_c+cos_a*cos_c
    rot_pose[:,1,2] = cos_a*sin_b*sin_c-sin_a*cos_c

    rot_pose[:,2,0] = -sin_b
    rot_pose[:,2,1] = sin_a*cos_b
    rot_pose[:,2,2] = cos_a*cos_b

    rot_pose[:,:3,3] = x

    rot_pose[:,3,3] = 1
    return rot_pose

def build_chain():
    out_path = 'datasets/arm/UR5'
    end_effect = 'wrist_3_link'
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    chain = pk.build_serial_chain_from_urdf(
        open(out_path+'/'+"ur5e.urdf").read(), end_effect)
    chain = chain.to(dtype=dtype, device=d)
    th_batch = torch.rand(1, 6, device='cuda')#torch.tensor(sampled_points, device='cuda')
    tg_batch = chain.forward_kinematics(th_batch, end_only = False)
    mesh_list = []
    iter = 0
    for tg in tg_batch:
        #print(tg)
        if iter>1 and iter < 8:
            ball_list = np.load(out_path+'/meshes/sphere/sphere/'+tg+'.npy')
            mesh_list.append(torch.tensor(ball_list, dtype=torch.float32, device='cuda'))
        iter=iter+1
    return chain, mesh_list

def gradient(y, x, create_graph=True):                                                               
                                                                                  
    grad_y = torch.ones_like(y)                                                                 

    grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
    
    return grad_x  



def FK(sampled_points, chain, mesh_list):
    shape=sampled_points.shape
    pointsize = 0
    
    scale = np.pi/0.5
    
    th_batch = torch.tensor(sampled_points, requires_grad=True, device='cuda')#.cuda()
    del sampled_points
    #tg_batch = chain.forward_kinematics(th_batch, end_only = False)
    def chain_to_matrix(th_batch):

        tg_batch = chain.forward_kinematics(th_batch, end_only = False)
        iter = 0
        #pointsize = 0
        m_list = []
        for tg in tg_batch:
            if iter>1 and iter < 8:
                m = tg_batch[tg].get_matrix()
                m_list.append(m)
                del m
            iter=iter+1
        del tg_batch, th_batch
        #print(torch.cat(m_list, dim=0).shape)
        return torch.cat(m_list, dim=0)

    def output_and_jacobian_fn(th_batch):
        output = chain_to_matrix(th_batch)  # Get the output of the model
        jacobian = jacfwd(chain_to_matrix)(th_batch)  # Compute the Jacobian using forward-mode autodiff
        del th_batch
        return output, jacobian
    #print('th_batch',th_batch.shape)
    #matrix_list = chain_to_matrix( th_batch)
    matrix_list, jacobian = vmap(output_and_jacobian_fn)(th_batch)
    #print(matrix_list.shape, jacobian.shape)
    matrix_list = matrix_list.detach()
    jacobian = jacobian.detach()
    del th_batch

    torch.cuda.empty_cache()
    iter = 0
    #pointsize = 0   
    p_list = []
    gradient_p_list = []
    for iter in range(6):
            
        ball_list = mesh_list[iter]
        
        ones_column = torch.ones(ball_list.size(0), 1, device='cuda')
        nv = torch.cat((ball_list[:,:3], ones_column), dim=1)#.detach()

        m = matrix_list[:,iter,...]
        gradient_m = jacobian[:,iter,...].permute(0, 3, 1, 2)

        print(m.shape,nv.shape)

        p = torch.matmul(m[:],nv.T)
        p = torch.permute(p,(0,2,1))
        #print(p.shape, ball_list.shape)
        p[...,3] = ball_list[:,3]
        p_list.append(p)

        gradient_p = torch.matmul(gradient_m, nv.T)
        #print(gradient_p.shape)
        gradient_p_list.append(gradient_p)
        #if iter == 7:
        #    print(m)
        
        del m,p,nv,ones_column,ball_list,gradient_p,gradient_m
    del matrix_list, jacobian 
    torch.cuda.empty_cache()   
    return p_list, gradient_p_list

def arm_obstacle_distance(th_batch, chain, mesh_list, kdtree, v_obs):
    #import torch.autograd.profiler as profiler
    #chain, mesh_list = build_chain()
    whole_dis = torch.zeros((th_batch.shape[0]),dtype=torch.float32, device='cuda')
    whole_normal = torch.zeros((th_batch.shape[0],6),dtype=torch.float32, device='cuda')
    end_dis = []
    #print('all_shape',th_batch.shape)
    batch_size = 200000
    

    for batch_id in range(math.floor(th_batch.shape[0]/batch_size)+1):
        if batch_id*batch_size==th_batch.shape[0]:
            break
       
        sampled_points = th_batch[batch_id*batch_size:
                    min((batch_id+1)*batch_size,th_batch.shape[0]),:]
        batch_size = sampled_points.shape[0]
        
        p_list, gradient_p_list = FK(sampled_points, chain, mesh_list)
        del sampled_points

        torch.cuda.empty_cache()

        query_points = torch.cat(p_list, dim=1)
        query_points_grad = torch.cat(gradient_p_list, dim = 3)
        del p_list, gradient_p_list

        torch.cuda.empty_cache()

        query_points_grad = query_points_grad.permute(0,3,1,2)
        #print(query_points_grad.shape)

        query_points = torch.reshape(query_points, (-1, 4))
        query_points_grad = torch.reshape(query_points_grad, (-1, 6, 4))
        #print(query_points_grad.shape)

        dists, inds = kdtree.query(query_points[:,:3], nr_nns_searches=1)
        dists = dists.squeeze()
        inds = inds.squeeze()
        distance = torch.sqrt(dists)-query_points[:,3]

        del dists
        #sign_distance = torch.sign(unsigned_distance)
        #print(closest_points.shape)

        #distance.backward()
        #print(sampled_points.grad)

        normal = query_points[:,:3]-v_obs[inds,:]

        #print(f"Initial memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")


        del inds, query_points
        #print(normal.shape)
        normal = torch.bmm(query_points_grad[...,:3],normal.unsqueeze(2)).squeeze(2)
        #print(normal.shape)

        #del query_points_grad

        torch.cuda.empty_cache()

        distance = distance.reshape(batch_size, -1)
        normal = normal.reshape(batch_size, -1, normal.shape[-1])
        #print(normal.shape)
        
        #min_distance, min_ind = torch.min(distance, dim=1)
        arg_min = torch.argmin(distance, dim=1, keepdim=True)
        #distance = distance.detach().cpu().numpy()
        #print(arg_min.shape)
        #min_distance = distance[arg_min]
        #min_normal = normal[arg_min, :]

        min_distance = torch.gather(distance, 1, arg_min).squeeze(1)
        min_normal = torch.gather(normal, 1, (arg_min.unsqueeze(2)).expand(-1, -1, 6)).squeeze(1)
        #print('min_distance',min_distance.shape)
        del distance, normal, arg_min
        #print('min_normal',min_normal.shape)
        #print(whole_distance)
        whole_dis[batch_id*batch_size:  min((batch_id+1)*batch_size,th_batch.shape[0])] = min_distance
        whole_normal[batch_id*batch_size:  min((batch_id+1)*batch_size,th_batch.shape[0]),:] = min_normal
        #whole_dis.append(min_distance)
        #whole_normal.append(min_normal)

        #print(torch.cuda.memory_summary())

        #print(torch.cuda.memory_summary(device=None, abbreviated=False))


        del min_distance#, min_normal 

        torch.cuda.empty_cache()
    #print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    #print('del')
    del th_batch

    #print(whole_dis.shape, whole_normal.shape)
    #del chain, mesh_list

    # memory_in_bytes = whole_normal.element_size() * whole_normal.nelement()

    # # Convert to MB
    # memory_in_mb = memory_in_bytes / 1e6

    # print(f"Tensor memory usage: {memory_in_mb:.2f} MB")
    torch.cuda.empty_cache()

    #stats = torch.cuda.memory_stats()
    #print(stats)
    #print('whole_normal',whole_normal.shape)
    #print('whole_normal',whole_normal.shape)
    #print('whole_dis',whole_dis.shape)
    # whole_normal = whole_normal/torch.norm(whole_normal, p=2, dim=1, keepdim=True)
    # where = ~torch.any(whole_normal.isnan(),dim=1)
    # whole_normal = whole_normal[where]
    # whole_dis = whole_dis[where]
    # whole_th_batch = th_batch[where]
    
    return whole_dis, whole_normal

def arm_append_list(X_list, Y_list, N_list,
                    chain, mesh_list, kdtree, v_obs,
                    numsamples, dim, offset, margin):
    
    OutsideSize = numsamples + 2
    WholeSize = 0

    scale = math.pi/0.5

    while OutsideSize > 0:
        '''
        P  = torch.rand((15*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        dP = torch.rand((15*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((15*numsamples,1),dtype=torch.float32, device='cuda'))*np.sqrt(3)
        nP = P + torch.nn.functional.normalize(dP,dim=1)*rL
        #nP = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        
        PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
        

        x0 = P[PointsInside,:]
        x1 = nP[PointsInside,:]
        '''

        P  = torch.rand((int(2*numsamples),3),dtype=torch.float32, device='cuda')
        P[:,0]=(P[:,0]*1.2+0.2)
        P[:,1]=(P[:,1]-0.5)*1.2
        P[:,2]=P[:,2]*1.2-0.1

        x0 = P

        a0 = -0.5*math.pi+(torch.rand((x0.shape[0],1),dtype=torch.float32, device='cuda')-0.5)*0.6*math.pi#.squeeze()
        b0 = (torch.rand((x0.shape[0],1),dtype=torch.float32, device='cuda')-0.5)*0.6*math.pi#.squeeze()
        c0 = -0.5*math.pi+(torch.rand((x0.shape[0],1),dtype=torch.float32, device='cuda')-0.5)*0.6*math.pi#.squeeze()

        end_pose0 = rot_pose(x0,a0.squeeze(),b0.squeeze(),c0.squeeze()) #end_pose0 @
        print(end_pose0.shape)
        torch_ik = torch_IK_UR5(end_pose0.shape[0])
        torch_ik.setJointLimits(-math.pi, math.pi)
        t0 = torch_ik.solveIK(end_pose0)
        x0 = torch.reshape(t0,(t0.shape[0]*t0.shape[1],t0.shape[2]))
        del end_pose0, t0, torch_ik, a0, b0, c0

        x0 = x0/scale

        dP = torch.rand((x0.shape[0],dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((x0.shape[0],1),dtype=torch.float32, device='cuda'))*0.5#np.sqrt(2)
        nP = x0 + torch.nn.functional.normalize(dP,dim=1)*rL
        #nP = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        del P, dP, rL
        PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
        

        x0 = x0[PointsInside,:]
        x1 = nP[PointsInside,:]

        #print(x0.shape[0])
        if(x0.shape[0]<=1):
            continue

        th_batch0 = scale*x0
        #print(th_batch.shape)
        obs_distance0, normal0 = arm_obstacle_distance(th_batch0, chain, mesh_list, kdtree, v_obs) #- 0.01
        #print(torch.min(obs_distance0))
        #print(torch.max(obs_distance0))
        obs_distance0 = obs_distance0-0.01

        where_d          =  (obs_distance0 > 0) & (obs_distance0 < margin) #\
                            #& (torch.norm(normal0,dim=1) > 0.1)
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = obs_distance0[where_d]
        n0 = normal0[where_d]

        th_batch1 = scale*x1
        obs_distance1, normal1 = arm_obstacle_distance(th_batch1, chain, mesh_list, kdtree, v_obs) #- 0.01
        obs_distance1 = obs_distance1-0.01

        # where_d          =    (torch.norm(normal1,dim=1) > 0.1)
        # x0 = x0[where_d]
        # x1 = x1[where_d]
        # y0 = y0[where_d]
        # y1 = obs_distance1[where_d]
        # n0 = n0[where_d]
        # n1 = normal1[where_d]

        y1 = obs_distance1
        n1 = normal1
        

        del th_batch0, th_batch1, obs_distance0, obs_distance1, normal0, normal1
        
        print('WholeSize',WholeSize)
        #print(x1.shape)
        #print(y0.shape)
        #print(y1.shape)
        n0 = n0/torch.norm(n0,dim=1,keepdim=True)
        n1 = n1/torch.norm(n1,dim=1,keepdim=True)
        where0 = ~torch.any(n0.isnan(),dim=1)
        where1 = ~torch.any(n1.isnan(),dim=1)
        x0 = x0[where0&where1]
        x1 = x1[where0&where1]
        y0 = y0[where0&where1]
        y1 = y1[where0&where1]
        n0 = n0[where0&where1]
        n1 = n1[where0&where1]
    # 

        x = torch.cat((x0,x1),1)
        y = torch.cat((y0.unsqueeze(1),y1.unsqueeze(1)),1)
        n = torch.cat((n0,n1),1)
        del x0,x1,y0,y1,n0,n1

        X_list.append(x)
        Y_list.append(y)
        N_list.append(n)

        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]

        del x,y,n
        
        if(WholeSize > numsamples):
            break
    return X_list, Y_list, N_list

def arm_rand_sample_bound_points(numsamples, dim, 
                                 v_obs, offset, margin):
    numsamples = int(numsamples)

    chain, mesh_list = build_chain()

    bb_max = np.array([[0.6,0.6,0.6]])#v_obs.max(axis=0, keepdims=True)
    bb_min = np.array([[-0.6,-0.6,-0.6]])#v_obs.min(axis=0, keepdims=True)

    bb_max = v_obs.max(axis=0, keepdims=True)
    bb_min = v_obs.min(axis=0, keepdims=True)
    
    bb_max = torch.tensor(bb_max, dtype=torch.float32, device='cuda')[0]
    bb_min = torch.tensor(bb_min, dtype=torch.float32, device='cuda')[0]
    #print(bb_max)
    
    v_obs = torch.tensor(v_obs, dtype=torch.float32, device='cuda')
    #n_obs = torch.tensor(n_obs, dtype=torch.float32, device='cuda')

    kdtree = torch_kdtree.build_kd_tree(v_obs)

    X_list = []
    Y_list = []
    N_list = []

    X_list, Y_list, N_list = arm_append_list(X_list, Y_list, N_list,
                                    chain, mesh_list, kdtree, v_obs,
                                    numsamples, dim, offset, margin)
  
    X = torch.cat(X_list,0)[:numsamples]
    Y = torch.cat(Y_list,0)[:numsamples]    
    N = torch.cat(N_list,0)[:numsamples]    

    sampled_points = X.detach().cpu().numpy()
    distance = Y.detach().cpu().numpy()
    normal = N.detach().cpu().numpy()

    
    distance0 = distance[:,0]#-0.005
    distance1 = distance[:,1]#-0.005
    speed  = np.zeros((distance.shape[0],2))
    speed[:,0] = np.clip(distance0 , a_min = offset, a_max = margin)/margin
    speed[:,1] = np.clip(distance1 , a_min = offset, a_max = margin)/margin

    return sampled_points, speed, normal

def point_obstacle_distance(query_points, kdtree, v_obs, normal_obs):


    dists, inds = kdtree.query(query_points, nr_nns_searches=1)
    dists = dists.squeeze()
    inds = inds.squeeze()
    unsigned_distance = torch.sqrt(dists)
    #print(closest_faces.shape)
    #print(normal_obs)
    #print(dists.shape)
    #print(inds.shape)
    normal_face = normal_obs[inds,:]
    #print(closest_points.shape)
    normal = query_points-v_obs[inds,:]
    normal = torch.nn.functional.normalize(normal,dim=1)

    dot = torch.einsum('ij,ij->i', normal, normal_face)
    return unsigned_distance, dot, normal 

def point_append_list(X_list,Y_list, N_list,
                      kdtree,  v_obs, normal_obs,
                      bb_max,bb_min,
                        numsamples, dim, offset, margin):
    
    OutsideSize = numsamples + 2
    WholeSize = 0

    while OutsideSize > 0:
        P  = 2*(torch.rand((20*numsamples,3),dtype=torch.float32, device='cuda')-0.5)
        P[:,0]=P[:,0]*bb_max[0]
        P[:,1]=P[:,1]*bb_max[1]
        P[:,2]=P[:,2]*bb_max[2]
        
        dP = torch.rand((20*numsamples,3),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((20*numsamples,1),dtype=torch.float32, device='cuda'))*np.sqrt(dim)
        
        if dim==2:
            P[:,2]=0
            dP[:,2]=0
        nP = P + torch.nn.functional.normalize(dP,dim=1)*(rL)#+0.01)
        #nP = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        #nP  = 2*(torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5)
        #nP[:,0]=nP[:,0]*bb_max[0]
        #nP[:,1]=nP[:,1]*bb_max[1]
        #nP[:,2]=nP[:,2]*bb_max[2]
        #PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
        PointsInside =  (nP[:,0] <= bb_max[0]) & (nP[:,0] >= bb_min[0]) &\
                       (nP[:,1] <= bb_max[1]) & (nP[:,1] >= bb_min[1]) &\
                       (nP[:,2] <= bb_max[2]) & (nP[:,2] >= bb_min[2])
        #'''
        PointsInside =   (nP[:,0] <= bb_max[0]) & (nP[:,0] >= bb_min[0]) &\
                         (nP[:,1] <= bb_max[1]) & (nP[:,1] >= bb_min[1]) &\
                         (nP[:,2] <= bb_max[2]) & (nP[:,2] >= bb_min[2]) &\
                         (P[:,0] <= bb_max[0]) & (P[:,0] >= bb_min[0]) &\
                         (P[:,1] <= bb_max[1]) & (P[:,1] >= bb_min[1]) &\
                         (P[:,2] <= bb_max[2]) & (P[:,2] >= bb_min[2])
        #'''

        x0 = P[PointsInside,:]
        x1 = nP[PointsInside,:]

        #PointsInside = (torch.norm(nP,dim=1) <= 0.49) & (torch.norm(P,dim=1) <= 0.49) &\
        #                (torch.norm(nP,dim=1) >= 0.31) & (torch.norm(P,dim=1) >= 0.31)
        
        #x0 = P[PointsInside,:]
        #x1 = nP[PointsInside,:]

        '''
        '''

        #print(x0.shape[0])
        if(x0.shape[0]<=1):
            continue
        #print(len(PointsOutside))
        

        obs_distance0, dot0, normal0 = point_obstacle_distance(x0, kdtree, v_obs, normal_obs)
        
        where_d          =   (dot0<np.sqrt(2)/2)
        obs_distance0[where_d] *=-1
        normal0[where_d] *=-1
        
        obs_distance0 = obs_distance0-offset
        where_d          =   (obs_distance0 < margin) & (obs_distance0 > offset) 
        #where_d          =   (obs_distance0 < 2*margin) & (obs_distance0 > -offset) 
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = obs_distance0[where_d]
        n0 = normal0[where_d]

        obs_distance1, dot1, normal1 = point_obstacle_distance(x1, kdtree, v_obs, normal_obs)
        where_d          =   (dot1<np.sqrt(2)/2)
        obs_distance1[where_d] *=-1
        normal1[where_d] *=-1
        
        obs_distance1 = obs_distance1-offset
        
        where_d          =   (obs_distance1 > -offset) 
        
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = y0[where_d]
        y1 = obs_distance1[where_d]
        n0 = n0[where_d]
        n1 = normal1[where_d]

  

        print(x0.shape)

        x = torch.cat((x0[:,:dim],x1[:,:dim]),1)
        y = torch.cat((y0.unsqueeze(1),y1.unsqueeze(1)),1)
        n = torch.cat((n0[:,:dim],n1[:,:dim]),1)

        X_list.append(x)
        Y_list.append(y)
        N_list.append(n)
        
        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]
        
        if(WholeSize > numsamples):
            break
    return X_list, Y_list, N_list

def point_rand_sample_bound_points(numsamples, dim, 
                                    v_obs, n_obs,
                                    offset, margin):
    numsamples = int(numsamples)

    bb_max = np.array([[0.6,0.6,0.6]])#v_obs.max(axis=0, keepdims=True)
    bb_min = np.array([[-0.6,-0.6,-0.6]])#v_obs.min(axis=0, keepdims=True)

    bb_max = v_obs.max(axis=0, keepdims=True)
    bb_min = v_obs.min(axis=0, keepdims=True)
    
    bb_max = torch.tensor(bb_max, dtype=torch.float32, device='cuda')[0]
    bb_min = torch.tensor(bb_min, dtype=torch.float32, device='cuda')[0]
    #print(bb_max)
    
    v_obs = torch.tensor(v_obs, dtype=torch.float32, device='cuda')
    n_obs = torch.tensor(n_obs, dtype=torch.float32, device='cuda')

    kdtree = torch_kdtree.build_kd_tree(v_obs)

    
    X_list = []
    Y_list = []
    N_list = []
    
    X_list, Y_list, N_list = point_append_list(X_list, Y_list, N_list, kdtree, v_obs, n_obs,
                                bb_max, bb_min, numsamples, dim, offset, margin)
   
    X = torch.cat(X_list,0)[:numsamples]
    Y = torch.cat(Y_list,0)[:numsamples]
    N = torch.cat(N_list,0)[:numsamples]

    sampled_points = X.detach().cpu().numpy()
    distance = Y.detach().cpu().numpy()
    normal = N.detach().cpu().numpy()
    
    distance0 = distance[:,0] 
    distance1 = distance[:,1] 
    speed  = np.zeros((distance.shape[0],2))
    speed[:,0] = np.clip(distance0/margin , a_min = offset/margin, a_max = 1)
    speed[:,1] = np.clip(distance1/margin , a_min = offset/margin, a_max = 1)
    
    return sampled_points, speed, normal

def sample_speed(path, numsamples, dim):
    
    try:

        global out_path
        out_path = os.path.dirname(path)
        #print(out_path)
        global path_name 
        path_name = os.path.splitext(os.path.basename(out_path))[0]
        print('pp',path)
        global task_name 
        task_name = out_path.split('/')[2]#os.path.splitext(os.path.basename(out_path),'/')
        if task_name=='arm':
            #dim = np.loadtxt(out_path+'/dim')
            global end_effect
            with open(out_path+'/dim') as f:
                iter = 0
                for line in f:
                    data = line.split()
                    if iter==0:
                        dim = int(data[0])
                    else:
                        end_effect = data[0]
                        print(end_effect)
                    iter=iter+1
        file_name = os.path.splitext(os.path.basename(path))[0]
        input_file = os.path.join(out_path,file_name + '_scaled.off')
        out_file = out_path + '/sampled_points.npy'

        print(input_file)
        if os.path.exists(out_file):
            print(f'Exists: {out_file}')
            #return
   
        #out_file = out_path + '/boundary_{}_samples.npz'.format( sigma)


        limit = 0.5
        xmin=[-limit]*dim
        xmax=[limit]*dim
        velocity_max = 1
        
        
        margin = limit/10.0
        offset = margin/10.0 

        v, f = igl.read_triangle_mesh(input_file)
        z = np.zeros(3)
        #n = igl.per_face_normals(v, f, z)




        v_obs = v


        start = timer()
        
        sampled_points, speed, normal = arm_rand_sample_bound_points(numsamples, dim, 
                                 v_obs, offset, margin)
        
        end = timer()

        #S0, I0, C0 = igl.signed_distance(sampled_points[:,:3], v, f, return_normals=False)
        #S1, I1, C1 = igl.signed_distance(sampled_points[:,3:], v, f, return_normals=False)
        #where = (S0>0) & (S1>0)

        #sampled_points=sampled_points[where]
        #speed=speed[where]
        #normal=normal[where]

        print(sampled_points.shape)

        # sampled_points=sampled_points[:numsamples]
        # speed=speed[:numsamples]
        # normal=normal[:numsamples]

        print(end-start)

        B = np.random.normal(0, 1, size=(3, 128))

        np.save('{}/sampled_points'.format(out_path),sampled_points)
        np.save('{}/speed'.format(out_path),speed)
        np.save('{}/normal'.format(out_path),normal)
        np.save('{}/B'.format(out_path),B)
    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))
    
