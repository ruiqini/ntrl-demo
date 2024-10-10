## About
This is a minimal example. 

## Setup
1. git clone this repo
2. run `docker build -t ntrl:demo .` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run -u $(id -u):$(id -g) --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/home/n/Eikonal_Planning/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/home/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -ti --rm ntrl:demo` to start the docker container.
4. Find `torch_kdtree` and install
5. run `python dataprocessing/preprocess.py --config configs/gibson.txt ` to sample training data
6. run `python train/train_gib.py` to start the training. 
