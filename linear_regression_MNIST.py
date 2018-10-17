import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mnist import MNIST


data = MNIST(data_dir = "/data/dir")

