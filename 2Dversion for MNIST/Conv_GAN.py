import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from scipy.misc import imsave
import os


class GAN:

	def __init__(self, params):

		self.params = params

		self.input = tf.placeholder(tf.float32, shape=[None, 784], name="Inputs")
		input = tf.reshape(self.input, [-1, 28, 28, 1])

		with tf.variable_scope("discriminator"):
			output_discriminate = layers.conv2d(input, num_outputs=16, kernel_size=3, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
			output_discriminate = layers.conv2d(output_discriminate, num_outputs=32, kernel_size=3, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
			output_discriminate = layers.conv2d(output_discriminate, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
			output_discriminate = layers.flatten(output_discriminate)
			output_discriminate = layers.fully_connected(output_discriminate, num_outputs=1, activation_fn=None)

		self.D_out = output_discriminate


		with tf.variable_scope("generator"):

			samples = tf.random_normal([self.params.minibatch_size,self.params.n_z], 0, 1, dtype=tf.float32)
			output_generate = layers.fully_connected(samples, num_outputs=7*7*32, activation_fn=None)
			output_generate = tf.reshape(output_generate, [-1, 7, 7, 32])
			output_generate = layers.conv2d_transpose(output_generate, num_outputs=16, kernel_size=3, stride=2)
			output_generate = layers.conv2d_transpose(output_generate, num_outputs=1, kernel_size=3, stride=2, activation_fn=tf.nn.sigmoid)

		self.G_out = output_generate

		with tf.variable_scope("discriminator", reuse=True):
			output_discriminate = layers.conv2d(self.G_out, num_outputs=16, kernel_size=3, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
			output_discriminate = layers.conv2d(output_discriminate, num_outputs=32, kernel_size=3, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
			output_discriminate = layers.conv2d(output_discriminate, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
			output_discriminate = layers.flatten(output_discriminate)
			output_discriminate = layers.fully_connected(output_discriminate, num_outputs=1, activation_fn=None)

		self.DG_out = output_discriminate

		D_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
		G_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")

		# self._D_loss = tf.losses.sigmoid_cross_entropy(self.D_out, tf.ones(tf.shape(self.D_out))) + tf.losses.sigmoid_cross_entropy(1-self.DG_out, tf.ones(tf.shape(self.DG_out)))
		# self._G_loss = tf.losses.sigmoid_cross_entropy(self.DG_out, tf.ones(tf.shape(self.DG_out)))
		D_real = self.D_out
		D_fake = self.DG_out
		self._D_loss = -(tf.reduce_mean(D_real) - tf.reduce_mean(0.25*D_fake**2 + D_fake))
		self._G_loss = -tf.reduce_mean(0.25*D_fake**2 + D_fake)
		

		self._train_D = tf.train.AdamOptimizer(learning_rate=self.params.lr).minimize(self._D_loss, var_list=D_params)
		self._train_G = tf.train.AdamOptimizer(learning_rate=self.params.lr).minimize(self._G_loss, var_list=G_params)

	@property
	def loss(self):
		return self._D_loss + self._G_loss

	@property
	def optimize_discriminator(self):
		return self._train_D

	@property
	def optimize_generator(self):
		return self._train_G

	@property
	def prediction(self):
		return self.G_out

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img

def train(params):

	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	n_samples = mnist.train.num_examples
	n_batches = n_samples//params.minibatch_size

	if os.path.isdir("results"):
		pass
	else:
		os.makedirs("results")

	with tf.Session() as sess:

		gan = GAN(params)
		sess.run(tf.global_variables_initializer())
		for i in range(params.epochs):
			for j in range(n_batches):
				batch = mnist.train.next_batch(params.minibatch_size)[0]
				sess.run(gan.optimize_discriminator, feed_dict={gan.input : batch})
				if j%params.k_steps==0:
					sess.run(gan.optimize_generator)
				if j==(n_batches-1):
					G_loss, D_loss, loss = sess.run([gan._G_loss, gan._D_loss, gan.loss], feed_dict={gan.input : batch})
					print "Epoch : " + str(i) + " complete."
					print "G_Loss :" + str(G_loss)
					print "D_Loss :" + str(D_loss)
					print "Loss :" + str(loss)
					print "#######################"
					generated_images = sess.run(gan.prediction)
					generated_images = generated_images.reshape([params.minibatch_size, 28, 28])
					imsave("results/"+str(i)+".jpg", generated_images[0])

					# imsave("results/"+str(i)+".jpg", merge(generated_images[:params.minibatch_size],[8,8]))

if __name__=='__main__':
	flags = tf.app.flags
	flags.DEFINE_float("lr", 1e-4, "Learning rate for GAN")
	flags.DEFINE_integer("epochs", 100000, "Epochs for training")
	flags.DEFINE_integer("k_steps", 10, "Train Generator")
	flags.DEFINE_integer("minibatch_size", 64, "1Mini-batch size for training")
	flags.DEFINE_integer("n_z", 20, "Latent space dimension")
	params = flags.FLAGS

	train(params)

