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

#defining input for tesorflow graph
x = tf.placeholder(tf.float32, [None, img_size_flat]) # arbitrary no. of images in which each image is of size img_size_flat
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

#defining variables to be optimized
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))
'''
defining the model ---> x*weights + biases is forward propagated in each iteration and 
weights and biases are optimized
'''
logits = tf.matmul(x, weights) + biases
'''
The above operation results in a matrix of size num_images(which we defined as None earlier)
and num_classes. 
We then normalize the matrix because the values may be too large or too small for our 
interpretation
'''
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis = 1)

#Optimizing the cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,
														labels = y_true)
'''
The cross-entropy is a performance measure used in classification. The cross-entropy 
is a continuous function that is always positive and if the predicted output of the 
model exactly matches the desired output then the cross-entropy equals zero. 
The goal of optimization is therefore to minimize the cross-entropy so it gets as 
close to zero as possible by changing the weights and biases of the model.
'''
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#creating tensorflow session

session = tf.Session()
session.run(tf.global_variables_initializer())
batch_size = 12

def optimize(num_iterations):
	for x in range(num_iterations):
		x_batch, y_true_batch, _ = data.random_batch(batch_size = batch_size)
		'''
		x_batch holds a batch of images and y_true_batch holds the true labels for the images.
		Put the batch into a dict with proper names for placeholder variables in Tensorflow 
		Graph.
		'''
		feed_dict_train = {x: x_batch,
							y_true: y_true_batch}
		session.run(optimizer, feed_dict = feed_dict_train)

feed_dict_test = {x: data.x_test,
				y_true: data.y_test,
				y_true_cls: data.y_test_cls}

def print_accuracy():
	acc = session.run(accuracy, feed_dict = feed_dict_test)
	print("Accuracy on test set: {0:.1%}".format(acc))

