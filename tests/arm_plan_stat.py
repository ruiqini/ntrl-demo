import sys

sys.path.append('.')
from models.metric_arm import model_test_metric as md
import torch
import os 
import numpy as np
import matplotlib.pylab as plt
import torch
from torch import Tensor
from torch.autograd import Variable, grad


from timeit import default_timer as timer
import math
import igl
from glob import glob
import random

import pytorch_kinematics as pk

#import torch_kdtree #import build_kd_tree

def Viz_line_pc(centers, radii, colors, env_path):
    import trimesh
    meshpath = env_path
    v, f = igl.read_triangle_mesh(meshpath)

    #v[:,2] = v[:,2]-0.3
    bb_max = v.max(axis=0, keepdims=True)
    bb_min = v.min(axis=0, keepdims=True)
    print('bbox', bb_max, bb_min)
    
    #mesh = trimesh.load(meshpath)
    #mesh.visual.face_colors = [200, 192, 207, 150]
    z_values = v[:, 2]
    x_values = v[:, 0]

    combined_values = (z_values + x_values) / 2

    # Normalize the combined values between 0 and 1 for the colormap
    combined_normalized = (combined_values - combined_values.min()) / (combined_values.max() - combined_values.min())

    # Use a colormap from matplotlib (e.g., 'viridis', 'plasma', etc.)
    cmap = plt.get_cmap('viridis')  # Choose 'viridis', 'jet', etc.
    pc_colors = cmap(combined_normalized) 

    # Convert colors to the range [0, 255] for Trimesh
    pc_colors = (pc_colors * 255).astype(np.uint8)
    point_cloud = trimesh.points.PointCloud(vertices=v, colors=pc_colors)
    
    # Create a Scene and add the point cloud
    scene = trimesh.Scene()
    scene.add_geometry(point_cloud)
    # Define line segments for X (red), Y (green), and Z (blue) axes
    axis_length = 1.0
    x_axis = trimesh.load_path(np.array([[0, 0, 0], [axis_length, 0, 0]]))
    y_axis = trimesh.load_path(np.array([[0, 0, 0], [0, axis_length, 0]]))
    z_axis = trimesh.load_path(np.array([[0, 0, 0], [0, 0, axis_length]]))
    x_axis.colors = [[255, 0, 0, 255]]
    y_axis.colors = [[0, 255, 0, 255]]
    z_axis.colors = [[0, 0, 255, 255]]
    scene.add_geometry([ x_axis, y_axis, z_axis])
    # Define camera positions
    
    # Define a plane
    # height = 0.3
    # size = 7
    # center = np.array([-5, 0, height])
    # plane_vertices =[
    #     center + np.array([-size, -size, 0]),
    #     center + np.array([size, -size, 0]),
    #     center + np.array([size, size, 0]),
    #     center + np.array([-size, size, 0])
    # ]
    # plane_faces = [
    #     [0, 1, 2],
    #     [0, 2, 3],
    #     [0, 3, 2],
    #     [0, 2, 1]
    # ]
    # plane_mesh = trimesh.Trimesh(plane_vertices, plane_faces, process=False)
    # plane_mesh.visual.face_colors = [100, 100, 255, 100]
    # add points cloud
    if True:
        #start_points = query_points
        #trimesh create sphere
        print(centers.shape)
        vertices = centers.reshape(-1, 3)
        edges = np.array([[i*centers.shape[1] + j, i*centers.shape[1] + j + 1] for i in range(centers.shape[0]) for j in range(centers.shape[1] - 1)])

        # Create a single Path3D object with all points and edges
        path = trimesh.load_path(vertices[edges])
        #path.visual.vertex_colors = colors.reshape(-1, 4).astype(np.int64)
        
        #print(query_points[0], query_points[-1])
        #point_cloud = trimesh.PointCloud(end_points, end_colors)

        #start_points = query_points
        #trimesh create sphere
        spheres = []

        # Create spheres with colors
        for i, (center, radius) in enumerate(zip(centers.reshape(-1, 3), radii.reshape(-1, 1))):
            # Create the sphere (either uv_sphere or icosphere)
            sphere = trimesh.creation.icosphere(radius=radius)
            
            # Translate the sphere to the correct center
            sphere.apply_translation(center)
            
            # Assign vertex colors (RGBA) for the current sphere
            vertex_colors = np.tile(colors[i], (sphere.vertices.shape[0], 1))  # Repeat color for each vertex
            sphere.visual.vertex_colors = vertex_colors
            
            # Append to the list of spheres
            spheres.append(sphere)

        # Concatenate all the spheres into a single mesh
        combined_spheres = trimesh.util.concatenate(spheres)

        scene.add_geometry([path, combined_spheres])
    
    scene.show()

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

def FK(sampled_points, chain, mesh_list):
    shape=sampled_points.shape
    pointsize = 0
    
    scale = np.pi/0.5
    
    th_batch = torch.tensor(sampled_points, requires_grad=True, device='cuda')#.cuda()

    tg_batch = chain.forward_kinematics(th_batch, end_only = False)

    iter = 0
    #pointsize = 0
    p_list = []
    for tg in tg_batch:
        if iter>1 and iter < 8:
            
            ball_list = mesh_list[iter-2]#np.load(out_path_+'/meshes/sphere/'+tg+'.npy')
            
            ones_column = torch.ones(ball_list.size(0), 1, device='cuda')
            nv = torch.cat((ball_list[:,:3], ones_column), dim=1)

            m = tg_batch[tg].get_matrix()
            p = torch.matmul(m[:],nv.T)
            p = torch.permute(p,(0,2,1))
            p[...,3] = ball_list[:,3]
            p_list.append(p)
            ##if iter == 7:
            #    print(m)
            
            del m,p,nv
        iter=iter+1
    return p_list

def MPPI(womodel, XP):
    steps = 200
    sample_num = 50
    horizon = 5

    dP_prior = torch.zeros((1,6)).cuda()

    point0=[]

    point0.append(XP[:,0:6].clone())

    for iter in range(steps):
        #print(iter)
        XP_tmp = XP.clone()#[:,0:3]
        #print(XP_tmp)
        XP_tmp = XP_tmp.unsqueeze(0).repeat(sample_num,horizon,1)
        dP_list = []
        cost_path = 0
        XP_list = []
        #dP_list = []
        dP = 0.015 * torch.normal(0,1,size=(sample_num, 1, 6),dtype=torch.float32, device='cuda') \
            +0.015 * torch.normal(0,1,size=(sample_num, horizon, 6),dtype=torch.float32, device='cuda')
        #dP = 0.02 * torch.nn.functional.normalize(dP + dP_prior,dim=2)
        dP = dP + 2*dP_prior
        dP_norm = torch.norm(dP,dim=2,keepdim=True)
        dP = dP/(torch.clamp(dP_norm,min=0.015)/0.015)
        #print(dP)
        dP_cumsum = torch.cumsum(dP, dim=1)
        #print(XP_tmp.shape)
        XP_tmp[...,0:6] = XP_tmp[...,0:6]+dP_cumsum
        
        indices = [0, -1]

        cost = womodel.function.TravelTimes(XP_tmp[:,indices,:].reshape(-1,12))
        
        cost = cost.reshape(-1,2)
        cost = 10*cost[:,0] + cost[:,1]#torch.sum(cost.reshape(-1,2),dim=1)#
        
        
        weight = torch.softmax(-50*cost, dim=0)
        
        dP_prior = (weight@dP[:,0,:]) 

        XP[:,0:6] = dP_prior + XP[:,0:6]

        #print(XP.shape)
        dis=torch.norm(XP[:,6:12]-XP[:,0:6])
        #print(XP)
        point0.append(XP[:,0:6].clone())
        
        if(dis<0.01):
            break

    point0.append(XP[:,6:12].clone())
    return point0, iter

modelPath = './Experiments/UR5'

meshname = 'Auburn'#
#meshname = 'Spotswood'
dataPath = './datasets/arm/'+ meshname
#dataPath = './datasets/new/'

womodel    = md.Model(modelPath, dataPath, 6, [0, 0.0, 0.0,0, 0.0, 0.0], device='cuda')
pt='./Experiments/UR5/arm_09_19_01_14/Model_Epoch_01700_ValLoss_4.005805e-03.pt'
print(pt)
womodel.load(pt)#
womodel.network.eval()

#dataPath = './datasets/Gib'
paths = dataPath
scale = math.pi/0.5




XP=torch.tensor([[0.00,0.0,0.0,-0.00,0.00,-0.00,
                        0.2, -0.5, -1.2, 0.5*np.pi,0.5*np.pi,0.0]]).cuda()
XP=torch.tensor([[-0.2, -0.5, -0.35, 0.2*np.pi,0.5*np.pi,0.0,
                  0.00,0.0,0.0,-0.00,0.00,-0.00]]).cuda()
XP=torch.tensor([[0.4, -0.5, -0.35, 0.3*np.pi,0.5*np.pi,0.0,
                     0.2, -0.7, -0.9, 0.2*np.pi,0.7*np.pi,0.0]]).cuda()
XP=torch.tensor([[0.2, -0.5, -1.2, 0.5*np.pi,0.5*np.pi,0.0,
                     -0.2, -0.5, -0.35, 0.2*np.pi,0.5*np.pi,0.0]]).cuda()

XP=torch.tensor([[0.2, -0.7, -1.0, 0.5*np.pi,0.5*np.pi,0.0,
                        -0.2, -0.5, -0.35, 0.2*np.pi,0.5*np.pi,0.0]]).cuda()
    

   
BASE=torch.tensor([[0, -0.5*np.pi, 0.0, -0.5*np.pi,0.0,0.0,
                        0, -0.5*np.pi, 0.0, -0.5*np.pi,0.0,0.0]]).cuda()
#XP = start_goal
XP = XP+BASE #Variable(Tensor(XP)).to('cuda').unsqueeze(0)
XP = XP/scale

for ii in range(5):
    
    start = timer()
    with torch.no_grad():
        point, iter = MPPI(womodel, XP.clone())

    end = timer()

    print('Time:', end-start)
if iter == 199:
    print('Failed')
    #continue

query_points = torch.cat(point).to('cpu').data.numpy()#np.asarray(point)


chain, mesh_list = build_chain()

p_list = FK(query_points*scale, chain, mesh_list)



points = torch.cat(p_list,dim=1)

#np.save('Evaluations/Arm/ur5_points_list.npy', query_points*scale)


color1 = np.random.randint(256, size=(1, 4))
color1[0,0] = 50
color1[0,1] = 50
color1[0,2] = 200
color1[0,3] = 100
color2 = np.random.randint(256, size=(1, 4))
color2[0,0] = 200
color2[0,1] = 50
color2[0,2] = 50
color2[0,3] = 100
n_steps = points.shape[0]
interpolation = np.linspace(0, 1, n_steps)
colors = np.outer(1 - interpolation, color1) + np.outer(interpolation, color2)
colors = np.expand_dims(colors, axis=1)
colors = np.repeat(colors, repeats=points.shape[1], axis=1)

#colors[:int(colors.shape[0]/2),...] = color1
#colors[int(colors.shape[0]/2):,...] = color2

'''

'''
#points = points.view(-1,4)
points = points.detach().cpu().numpy()
print(points.shape)
file_path = 'datasets/arm/'
mesh_name = 'realpc_scaled.off'

path = file_path + 'UR5' + '/' + mesh_name

centers = points[...,0:3]
radii = points[...,3]

Viz_line_pc(centers, radii, colors.reshape(-1, 4).astype(np.int64), path)

