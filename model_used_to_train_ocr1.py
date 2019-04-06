import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)


#input_data
img_size = 28
n_classes = 10
batch_size = 100
n_epochs = 3

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
    
x = tf.placeholder(tf.float32, shape=[None, img_size*img_size], name = 'x')

y = tf.placeholder(tf.float32, shape=[None, n_classes], name = 'y')

saver = tf.train.Saver()

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
def neural_nework_model(x, testing):

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
    if testing:
        return tf.argmax(output, dimension=1)

    return output

def train_neural_network(x, y):

    model = neural_nework_model(x, False)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = model,
                                                            labels = y)

    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    
    with tf.Session() as ses:
        ses.run(tf.initialize_all_variables())
        #print('old weights: ',ses.run(weights['weight_conv1']))

        for epoch in range(n_epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples/batch_size)):

                x_set, y_set = mnist.train.next_batch(batch_size)
                _, c = ses.run([optimizer, cost], feed_dict={x: x_set,
                                                             y: y_set})

                epoch_loss += c
            print('Epoch: ', epoch, ' Loss: ', epoch_loss)
        print(ses.run(weights['weight_conv1']))
        save_path = saver.save(ses, 'F:/python/trained_data/mnist_data/data.ckpt')
        #For testing the changes
        acc = accuracy.eval({x:mnist.test.images,
                         y:mnist.test.labels})
        #print('new weights: ',ses.run(weights['weight_conv1']))

        print('Accuracy: ', acc)
    
train_neural_network(x, y)



















