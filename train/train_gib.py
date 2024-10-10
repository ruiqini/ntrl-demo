import sys
sys.path.append('.')
from models.metric import model_train_metric as md
from os import path
from glob import glob

modelPath = './Experiments/Gib'
#dataPath = './datasets/gibson/Auburn'
dataPath = './datasets/gibson/Spotswood'



#model    = md.Model(modelPath, dataPath, 3, [0, 0.3,-0.03],device='cuda:0')
model    = md.Model(modelPath, dataPath, 3, [-0.15, 0.1,0.1],device='cuda:0')

model.train()


