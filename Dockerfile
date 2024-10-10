# Start from the iGibson base image or another if more appropriate
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
#FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
#RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install pickle5
RUN pip install libigl
RUN pip install pytorch_kinematics
RUN pip install configargparse
#RUN pip install pointnext



WORKDIR /workspace
