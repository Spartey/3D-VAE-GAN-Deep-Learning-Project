import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

tf.reset_default_graph()

# Create interactive session
sess = tf.InteractiveSession()

# Discriminator Net
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

D_W1 = tf.get_variable('D_W1', shape = [784, 128], initializer=tf.contrib.layers.xavier_initializer())
D_b1 = tf.Variable(tf.zeros(shape = [128]), name = 'D_b1')

D_W2 = tf.get_variable('D_W2', shape = [128, 1], initializer=tf.contrib.layers.xavier_initializer())
D_b2 = tf.Variable(tf.zeros(shape = [1]), name = 'D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

G_W1 = tf.get_variable('G_W1', shape = [100, 128], initializer=tf.contrib.layers.xavier_initializer())
G_b1 = tf.Variable(tf.zeros(shape = [128]), name = 'G_b1')

G_W2 = tf.get_variable('G_W2', shape = [128, 784], initializer=tf.contrib.layers.xavier_initializer())
G_b2 = tf.Variable(tf.zeros(shape = [784]), name = 'G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]


# Training Iteration
N = 50000
# Training steps for Discriminator
K = 1
# Mini batch size
batch_size = 100
# dimensionality of latent representation
Z_dim = 100


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    
    return D_prob, D_logit


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))


# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdamOptimizer(0.0001).minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer(0.00015).minimize(G_loss, var_list=theta_G)

# Init all tf variables
sess.run(tf.global_variables_initializer())

for itr in range(N):
    for step in range(K):
        X_batch, _ = mnist.train.next_batch(batch_size)
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_batch, Z: sample_Z(batch_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})


# test on random z
check = sess.run(G_sample, feed_dict={Z: sample_Z(1, Z_dim)})
plt.imshow(np.reshape(check, [28,28]), cmap='Greys_r')
