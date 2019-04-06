import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from read_translate_image import get_image
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)

imag_index = 500

#test_image
#imag = mnist.train.images[imag_index]
pred = mnist.train.labels[imag_index]
#imag = np.array(imag, dtype='float32')
imag = get_image()
img = imag.reshape((28, 28))



#input_data
img_size = 28
n_classes = 10
batch_size = 100
n_epochs = 20

#conv_1
filter_size_1 = 5
num_filter_1 = 16

#max_pool_1
filter_size_m1 = 2

#conv_2
filter_size_2 = 5
num_filter_2 = 36

#max_pool_2
filter_size_m2 = 2

#full_conn
fc_num_features = 128
n_feature = 7*7*36

#Weights and Biases
weights={'weight_conv1':tf.Variable(tf.random_normal([filter_size_1,filter_size_1,1,num_filter_1])),
             'weight_conv2':tf.Variable(tf.random_normal([filter_size_2,filter_size_2,num_filter_1,num_filter_2])),
             'weight_fully_conn1':tf.Variable(tf.random_normal([n_feature, fc_num_features])),
             'weight_fully_conn2':tf.Variable(tf.random_normal([fc_num_features, n_classes]))
             }

biases={'biases_conv1':tf.Variable(tf.constant(0.05, shape=[num_filter_1])),
            'biases_conv2':tf.Variable(tf.constant(0.05, shape=[num_filter_2])),
            'biases_fully_conn1':tf.Variable(tf.constant(0.05, shape=[fc_num_features])),
            'biases_fully_conn2':tf.Variable(tf.constant(0.05, shape=[n_classes]))
            }

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, 'F:/python/trained_data/mnist_data/data.ckpt')

#x = tf.placeholder(tf.float32, shape=[None, img_size*img_size], name = 'x')

#y = tf.placeholder(tf.float32, shape=[None, n_classes], name = 'y')

#Layers of the neural network
def conv_layer(image, weights, biases):
    return tf.nn.conv2d(input=image,
                        filter=weights,
                        strides=[1,1,1,1],
                        padding='SAME')+biases
def max_pool(image):
    return tf.nn.max_pool(value=image,
                          ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='SAME')
def relu(image):
    return tf.nn.relu(image)
def fully_connected_layer(image, weights, biases):
    return tf.matmul(image,weights)+biases
#neural_network_model
def neural_nework_model(x):

    x = tf.reshape(x, [-1, img_size, img_size, 1])
    conv1 = conv_layer(x, weights['weight_conv1'], biases['biases_conv1'])
    max1 = max_pool(conv1)
    relu1 = relu(max1)
    conv2 = conv_layer(relu1, weights['weight_conv2'], biases['biases_conv2'])
    max2 = max_pool(conv2)
    relu2 = relu(max2)

    #reshape for the fully connected layer
    fc = tf.reshape(relu2, [-1, n_feature])
    
    fcl = fully_connected_layer(fc, weights['weight_fully_conn1'], biases['biases_fully_conn1'])
    relu3 = relu(fcl)
    fcl2 = fully_connected_layer(relu3, weights['weight_fully_conn2'], biases['biases_fully_conn2'])   

    output = tf.nn.softmax(fcl2)
    
    return tf.argmax(output, dimension=1)

print('Prediction: ',sess.run(neural_nework_model(imag)))
predict = tf.convert_to_tensor(pred, dtype=tf.float32)
print('Actual value: ',sess.run(tf.argmax(predict)))
plt.imshow(img, cmap = 'gray')
plt.show()
