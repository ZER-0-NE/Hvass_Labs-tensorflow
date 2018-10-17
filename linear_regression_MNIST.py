import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mnist import MNIST


data = MNIST()

img_size_flat = data.img_size_flat
img_shape = data.img_shape
num_classes = data.num_classes

#plotting images over 3x3 grid
def plot_img(images, cls_true, cls_pred = None):
	assert len(images) == len(cls_true) == 9

	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

	for i, ax in enumerate(axes.flat):
		ax.imshow(images[i].reshape(img_shape), cmap = 'binary')

		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "False {0} Pred: {1}".format(cls_true[i], cls_pred[i])
		ax.set_xlabel(xlabel)

		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()

#plotting sample images
images = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
plot_img(images, cls_true = cls_true)

