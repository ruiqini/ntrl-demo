import os 
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

        # centers = box_list[:, :3]  # Shape (10, 3), these are x, y, z centers of each box
        # sizes = box_list[:, 3:] 
        # box_min = centers - sizes / 2  # Min corner of the boxes
        # box_max = centers + sizes / 2  # Max corner of the boxes

        # def is_outside_box(samples, box_min, box_max):
        #     # Check if samples are outside all the boxes
        #     outside = (samples.unsqueeze(1) < box_min).any(dim=-1) | (samples.unsqueeze(1) > box_max).any(dim=-1)
        #     #print(outside.shape)
        #     return outside.all(dim=1)
        # where0 = is_outside_box(x0, box_min, box_max)
        # where1 = is_outside_box(x1, box_min, box_max)
        # x0 = x0[where0&where1]
        # x1 = x1[where0&where1]
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
        
        obs_distance0 = obs_distance0#-offset
        where_d          =   (obs_distance0 < margin) & (obs_distance0 > offset) 
        #where_d          =   (obs_distance0 < 2*margin) & (obs_distance0 > -offset) 
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = obs_distance0[where_d]
        #y0 = obs_distance0
        n0 = normal0[where_d]

        obs_distance1, dot1, normal1 = point_obstacle_distance(x1, kdtree, v_obs, normal_obs)
        where_d          =   (dot1<np.sqrt(2)/2)
        obs_distance1[where_d] *=-1
        normal1[where_d] *=-1
        
        obs_distance1 = obs_distance1#-offset
        
        where_d          =   (obs_distance1 > -offset) 
        
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = y0[where_d]
        y1 = obs_distance1[where_d]
        n0 = n0[where_d]
        n1 = normal1[where_d]

        #where_d          =   (obs_distance1 < 1.1*margin) & (obs_distance1 > offset) 
        #x0 = x0[where_d]
        #x1 = x1[where_d]
        #y0 = y0[where_d]
        #y1 = obs_distance1[where_d]

        print(x0.shape)
        #print(x1.shape)
        #print(y0.shape)
        #print(y1.shape)
        #y0 = torch.minimum(y0,-x0[:,2])
        #y1 = torch.minimum(y1,-x1[:,2])

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

    #bb_max = np.array([[0.6,0.6,0.6]])#v_obs.max(axis=0, keepdims=True)
    #bb_min = np.array([[-0.6,-0.6,-0.6]])#v_obs.min(axis=0, keepdims=True)

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
        
        if task_name=='test':
            margin = limit/20.0
            offset = margin/10.0 
        elif task_name=='gibson':
            margin = limit/15.0
            offset = margin/10.0 
        else:
            margin = limit/15.0
            offset = margin/10.0 

        # ind = out_path.split('/')[3]
        # all_box_list = np.load('datasets/c3d/all_box_list.npy')
        # box_list = all_box_list[int(ind)]/40
        # box_list = torch.tensor(box_list, dtype=torch.float32, device='cuda')
        # print(ind,box_list)
        v, f = igl.read_triangle_mesh(input_file)
        z = np.zeros(3)
        n = igl.per_vertex_normals(v, f, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA)
        #n = igl.per_face_normals(v, f, z)


        bary, FI, _ = igl.random_points_on_mesh(100000, v, f)

        face_verts = v[f[FI], :]
        v_obs = np.sum(bary[...,np.newaxis] * face_verts, axis=1)

        face_norms = n[f[FI], :]
        #n_obs = n[FI,:]
        n_obs = np.sum(bary[...,np.newaxis] * face_norms, axis=1)

        start = timer()
        
        sampled_points, speed, normal = point_rand_sample_bound_points(numsamples, dim, 
                    v_obs, n_obs, offset, margin)

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
    
