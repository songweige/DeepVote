# Rectify the Static Assumption of Multi-View Stereo 3D Mapping with DeepVote Model

The DeepVote model aims at rectifying the inaccuracy caused by the violation of the static assumption of Binocular(Pairwise) Stereo Matching (BSM) algorithms, which are widely used in Multi-View Stereo 3D mapping of the satellite images.

## Pre-requisites
* Python 2.7
* PyTorch
* Numpy
* Scipy

## Network Architecture

The network takes as input a set of the arbitrary number of images with two channels: color, and height which is calculated by BSM. 
The base model contains a series of convolutional blocks applied elementwise to extract local features. A global mean pooling 
operation is employed in the end to calculate the final rectification. The final DSM is obtained as the aggregation of the retification and 
the average of the input height maps. Each convolutional block includes several convolutional permutation equivalence layers with leakyReLU 
activation function and BatchNorm layers. The whole aarchitecture is shown as below.

![Network Architecture](image/model.jpg)

## Preparation 

First, create the enviroment with Anaconda. Installing Pytorch with the other versions of CUDA can be found at [Pytorch document](https://pytorch.org/get-started/previous-versions/). Here Pytorch 3.1.0 and CUDA 9.0 are used:
```
  mkdir DeepVote DeepVote/data DeepVote/results
  cd DeepVote
  git clone git@github.com:SongweiGe/DeepVote.git
  conda create -n DeepVote python=2.7
  conda activate DeepVote
  conda install scipy, pytorch=0.3.1 cuda90 -c pytorch
```

Then download the data from the link https://drive.google.com/open?id=1_2XN9GNBW7458o_4nrzaQ838UEoxUU7V to directory `cd data/`. The data are processed from [IARPA Multi-View Stereo 3D Mapping Challenge](https://www.iarpa.gov/challenges/3dchallenge.html). The input height and color images are stored in `data/MVS` and the ground truth DSM are stored in `data/DSM`. The KML files used to crop the areas are stored in `data/kml`.

## Usage

### Training
```
python train_DeepVote.py -m mean2 -o ../results/mean2/ -g 0 -r True > logs/mean2_res_fold1-4.txt
```

### Inference

```
python infer.py -g 1 -o ../results/ -m base -r True -if 0 -ie 399 -n plain_res -i ../data/MVS/dem_6_18 -np 1
```

### Visualization

```
python plot_diff.py -i ../results/plain_res/reconstruction/dem_6_18_fold0_399_pair10.npy -g ../data/DSM/dem_6_18.npy
```
