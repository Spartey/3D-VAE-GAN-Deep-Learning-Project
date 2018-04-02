import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

tf.reset_default_graph()

batch_size = 64
dec_in_channels = 1
n_latent = 8
reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = int(49*dec_in_channels/2)

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def encoder(X_in, keep_prob=0.5):
    with tf.variable_scope("encoder", reuse=None):
        x = tf.reshape(X_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(x, filters=64, kernel_size=5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)
        x = tf.layers.conv2d(x, filters=64, kernel_size=5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)
        x = tf.layers.conv2d(x, filters=64, kernel_size=5, strides=1, padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        
        z_mu = tf.layers.dense(x, units=n_latent)
        z_sig = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z = z_mu + tf.multiply(epsilon, tf.exp(z_sig))
        
        return z, z_mu, z_sig
    
def decoder(z_in, keep_prob=0.5):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(z_in, units=inputs_decoder)
        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 5, strides = 1, padding = 'same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)
        x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 5, strides = 2, padding = 'same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)
        x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 5, strides = 2, padding = 'same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])
        
        return img

with tf.variable_scope("loss", reuse=None):
    X_ph = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28, 1], name = "X_ph")
    keep_prob_ph = tf.placeholder(dtype = tf.float32, shape = (), name = 'keep_prob_ph')
    X_flat_ph = tf.reshape(X_ph, shape = [-1, 28*28])
    sampled_z, z_mu, z_sig = encoder(X_ph, keep_prob_ph)
    dec = decoder(sampled_z, keep_prob_ph)

unreshaped = tf.reshape(dec, [-1, 28*28])
reconstruction_loss = tf.reduce_sum(tf.squared_difference(unreshaped, X_flat_ph), 1)
KL_divergence = -0.5 * tf.reduce_sum(1.0 + 2.0 * z_sig - z_mu**2 - tf.exp(2.0 * z_sig), 1)
loss = tf.reduce_mean(reconstruction_loss + KL_divergence)

optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

from tqdm import tqdm

for i in tqdm(range(1000)):
    batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
    batch = np.array(batch).reshape((len(batch),28,28,1))
    sess.run(optimizer, feed_dict = {X_ph: batch, keep_prob_ph: 0.8})
    # print(i)
    if not i % 200:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, reconstruction_loss, KL_divergence, z_mu, z_sig], feed_dict = {X_ph: batch, keep_prob_ph: 1.0})
        plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
        plt.show()
        plt.imshow(d[0].reshape([28,28]), cmap='gray')
        plt.show()
        print(i, ls, np.mean(i_ls), np.mean(d_ls))