import sys
sys.path.append('.')
from models.metric_arm import model_train_metric as md#_newout_sqrtlog _newout_log2
from os import path
import numpy as np
modelPath = './Experiments/UR5'         
dataPath = './datasets/arm/UR5'

model    = md.Model(modelPath, dataPath, 6, [-1.2, 0.4-0.5*np.pi, 1.4, 0.2-0.5*np.pi,-0.5,0.9],device='cuda:0')

model.train()