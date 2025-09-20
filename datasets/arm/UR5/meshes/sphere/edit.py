import numpy as np

name = 'forearm_link'
link = np.load(name+'.npy')
#sort link by x
print(link)
link = link[(-link[:,0]).argsort()]
print(link)
np.save('sphere/'+name+'.npy',link)