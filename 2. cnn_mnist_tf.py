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





