import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
This script shows activation functions and their implementations
'''
x = tf.placeholder(dtype=tf.float32, shape=(None,2), name = "x")
w = tf.Variable(tf.ones((2,1)), dtype = tf.float32)
b = tf.Variable(tf.ones(1), dtype=tf.float32)

# "Input times weights, add Bias and Activate"
z = tf.add(tf.matmul(x,w), b)

init = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)
	data = {x: [[0.25, 0.15]]}
	result_z = session.run(z, feed_dict=data)
	print("Value of z is {}".format(result_z[0][0]))

# Sigmoid
def sigmoid_val(z):
	return (1/ (1 + np.exp(-z)))
out_def = sigmoid_val(result_z[0][0])
out = tf.sigmoid(z)
with tf.Session() as session:
	session.run(init)
	data = {x: [[0.25, 0.15]]}
	result = session.run(out, feed_dict=data)

print("tf.sigmoid: {} and sigmoid_func: {}".format(out_def, result[0][0]))

# Plotting sigmoid
data_points = np.linspace(-5, 5, 200)
y_sigmoid = tf.sigmoid(data_points)
with tf.Session() as session:
	y_sigmoid = session.run(y_sigmoid)

plt.figure(figsize = (10,6))
plt.plot(data_points, y_sigmoid, c="blue", label="sigmoid")
plt.ylim((-0.2, 1.2))
plt.legend(loc="best")
plt.show()


# Relu
def relu_val(z):
	if z < 0:
		return 0
	else:
		return z

out_def = relu_val(result_z[0][0])
out = tf.nn.relu(z)
with tf.Session() as session:
	session.run(init)
	data = {x: [[0.25, 0.15]]}
	result = session.run(out, feed_dict=data)

print("tf.nn.relu: {} and relu_func: {}".format(out_def, result[0][0]))

# Plotting relu
data_points = np.linspace(-5, 5, 200)
y_relu = tf.nn.relu(data_points)
with tf.Session() as session:
	y_relu = session.run(y_relu)

plt.figure(figsize = (10,6))
plt.plot(data_points, y_relu, c="blue", label="relu")
plt.ylim((-0.2, 1.2))
plt.legend(loc="best")
plt.show()

# tanh
def tanh_val(z):
	return (2/(1+np.exp(-2*z)) -1)

out_def = tanh_val(result_z[0][0])
out = tf.tanh(z)
with tf.Session() as session:
	session.run(init)
	data = {x: [[0.25, 0.15]]}
	result = session.run(out, feed_dict=data)

print("tf.tanh: {} and tanh_func: {}".format(out_def, result[0][0]))

# Plotting tanh
data_points = np.linspace(-5, 5, 200)
y_tanh = tf.tanh(data_points)
with tf.Session() as session:
	y_tanh = session.run(y_tanh)

plt.figure(figsize = (10,6))
plt.plot(data_points, y_tanh, c="blue", label="tanh")
plt.ylim((-0.2, 1.2))
plt.legend(loc="best")
plt.show()
