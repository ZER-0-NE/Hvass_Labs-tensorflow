import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import keras

data = MNIST()

#conv1
filter_size1 = 5
num_filters1 = 16

#conv2
filter_size2 = 5
num_filters2 = 26

#fully-connected
fc_size = 128

