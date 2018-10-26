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

img_size = data.img_size
img_size_flat = data.img_size_flat #1-D array of data values
img_shape = data.img_shape
num_classes = data.num_classes
num_channels = data.num_channels #no. of color channels for images

def plot_images(images, cls_true, cls_pred = None):
	assert len(images) == len(cls_true) == 9
	fig, axes = plt.subplots(3,3)
	fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

	for x, y in enumerate(axes.flat):
		y.imshow(images[x].reshape(img_shape), cmap = 'binary')

		if cls_pred is None:
			xlabel = "True {0}".format(cls_true[x])
		else:
			xlabel = "True {0}, Pred: {1}".format(cls_true[x], cls_pred[x])

		y.set_xlabel(xlabel)

		y.set_xticks([])
		y.set_yticks([])

	plt.show()

images = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
plot_images(images = images, cls_true = cls_true)

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))
def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape = [length]))

def new_conv_layer(input, #previous layer input
					num_input_channels, #of previous layer
					filter_size, #width and height of each filter
					num_filters,
					use_pooling=True) #2x2 max-pooling
	
	shape = [filter_size, filter_size, num_input_channels, num_filters] #shape of filter-weights for convolution
	wieghts = new_weights(shape = shape)
	biases = new_biases(length = num_filters)

	'''
	The strides are set as 1 in all dimensions, first and last stride are labelled as 1
	first being for image-number and last for input-channel. padding =same => size of
	input and output image are same.
	'''
	layer = tf.nn.conv2d(input = input,
							filter = weights,
							strides = [1,1,1,1],
							padding = 'SAME')
	



