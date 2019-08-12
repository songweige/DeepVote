# Rectify the Static Assumption of Multi-View Stereo 3D Mapping with DeepVote Model

## Pre-requisites
* Python 2.7
* PyTorch
* Numpy
* Scipy

## Method
Binocular(Pairwise) Stereo Matching (BSM) algorithms assume the objects are the same in two views which doesn't actually hold for satellite images. 
Due to the time discrepancy among different views, there could be great illumination and appearance differences, which leads to 
poor accuracy in the fused DSM. The core assumption of our method is that based the content of the area, 
some of the inaccuracy could be retified.

## Network Architecture

The network takes as input a set of images with two channels: color, and height which is calculated by BSM. 
It aims at rectify the inaccuracy caused by the violation of the static assumption of BSM.
The base model contains a series of convolutional blocks which are applied element-wise to extract local features. A global mean pooling 
operation is employed in the end to calculate the final rectification. The final DSM is obtained as the aggregation of the retification and 
the average of the input height maps. Each convolutional block includes several convolutional permutation equivalence layers with leakyReLU 
activation function and BatchNorm layers.

![Network Architecture](images/model.pdf)
