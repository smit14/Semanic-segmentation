# Semanic-segmentation
The folder contains code for the pixel-wise classification of a road-scene image using different models like FCN32, FCN16, FCN8, Convolution-Deconvolution Model. The whole code is developed using TensorFlow library in python. This also contains unpooling operation in TensorFlow.

# Objective
Real-time pixel wise segmentation of Road scene images using Deconvolution network
with bispline upsampling

# Description
Semantic segmentation is a pixel-wise classification of an image where each pixel belongs to one of the
classes like car, road, pedestrians etc in case of road scene. Real-time semantic segmentation is an active
topic of research and very crucial for self-driving cars. Convolution - Deconvolution architecture has
been used to increase the accuracy of the task but it requires to store indices from pooling operation
which results in higher latency. We experimented different ways to do deconvolution process which
requires less memory and hence less inference time. This included replacing the unpooling method by
bi-spline upsampling and modifying deconvolution network to reduce the parameters involved.

# Tools and Libraries
* Tensorflow
* Numpy
* Matplotlib
