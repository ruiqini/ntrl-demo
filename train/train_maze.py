import sys
sys.path.append('.')
from models.metric import model_train_metric as md#_newout_sqrtlog _newout_log2
#from models import model_train as md
from os import path

modelPath = './Experiments/Maze'

dataPath = './datasets/test/maze'


# #model    = md.Model(modelPath, dataPath, 2,[-0.0,-0.0], device='cuda:0')

#model    = md.Model(modelPath, dataPath, 2,[-0.45,-0.45], device='cuda:0')
# #model    = md.Model(modelPath, dataPath, 2,[0.03,-0.45], device='cuda:0')

#dataPath = './datasets/test/maze5'

model    = md.Model(modelPath, dataPath, 2,[-0.0,-0.0], device='cuda:0')

#model    = md.Model(modelPath, dataPath, 2,[-0.0,-0.0], device='cuda:0')
#model    = md.Model(modelPath, dataPath, 2,[0.03,-0.45], device='cuda:0')

model.train()


