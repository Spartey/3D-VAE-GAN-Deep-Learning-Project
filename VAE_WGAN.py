import os, time, itertools
import numpy as np
import matplotlib
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

matplotlib.use('Agg')
import matplotlib.pyplot as plt


batch_size = 100
lr = 5e-5
train_epoch = 20
n_latent = 8
alpha_1 = 5
alpha_2 = 5e-4


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)


def encoder(x, keep_prob=0.5, isTrain=True):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = lrelu(conv1, 0.2)
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        conv5 = tf.layers.conv2d(lrelu4, 16, [4, 4], strides=(1, 1), padding='valid')
        x = tf.contrib.layers.flatten(conv5)
        z_mu = tf.layers.dense(x, units=n_latent)
        z_sig = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = z_mu + tf.multiply(epsilon, tf.exp(z_sig))

        return z, z_mu, z_sig


def generator(x, isTrain=True):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv5)

        return o


def discriminator(x, isTrain=True):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
        o = tf.nn.sigmoid(conv5)

        return o, conv5


def show_result(num_epoch, fixed_x_, show=False, save=False, path='result.png'):
    test_images = sess.run(G_z, {x: fixed_x_, isTrain: False, keep_prob: 1})
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid * size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (64, 64)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters

# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

# variables : input
x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
keep_prob = tf.placeholder(dtype=tf.float32)
isTrain = tf.placeholder(dtype=tf.bool)


# networks : encoder
z, z_mu, z_sig = encoder(x, keep_prob, isTrain)
z = tf.reshape(z, (-1, 1, 1, 8))
# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain)

# loss for each network

reconstruction_loss = tf.reduce_sum(tf.squared_difference(tf.reshape(G_z, (-1, 64*64)), tf.reshape(x, (-1, 64*64))), 1)
KL_divergence = -0.5 * tf.reduce_sum(1.0 + 2.0 * z_sig - z_mu**2 - tf.exp(2.0 * z_sig), 1)
mean_KL = tf.reduce_sum(KL_divergence)
mean_recon = tf.reduce_sum(reconstruction_loss)

VAE_loss = tf.reduce_mean(alpha_1 * KL_divergence + alpha_2 * reconstruction_loss)

D_loss_real = tf.reduce_mean(D_real_logits)
D_loss_fake = tf.reduce_mean(D_fake_logits)
D_loss = D_loss_real - D_loss_fake
G_loss = -tf.reduce_mean(D_fake_logits)
sub_loss = G_loss + VAE_loss

tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss', G_loss)

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]
E_vars = [var for var in T_vars if var.name.startswith('encoder')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.RMSPropOptimizer(lr).minimize(-D_loss, var_list=D_vars)
    G_optim = tf.train.RMSPropOptimizer(lr).minimize(sub_loss, var_list=G_vars + E_vars)
    for var in D_vars:
        tf.clip_by_value(var, -0.01, 0.01)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

logger = tf.summary.FileWriter('./graphs', sess.graph)
merged = tf.summary.merge_all()

# MNIST resize and normalization
train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

# results save folder
root = 'VAE_WGAN_results/'
if not os.path.isdir(root):
    os.mkdir(root)

# training-loop
num = 0
for epoch in range(train_epoch):
    epoch_start_time = time.time()
    fixed_x = train_set[:25]
    for iter in range(mnist.train.num_examples // batch_size):
        # update discriminator
        x_ = train_set[iter * batch_size:(iter + 1) * batch_size]
        for _ in range(4):
            sess.run(D_optim, feed_dict={x: x_, keep_prob: 0.8, isTrain: True})
        
        loss_d_, loss_g_, _VAE_loss, _KL_divergence, _reconstruction_loss, summary, _, _ = \
            sess.run([D_loss, G_loss, VAE_loss, mean_KL, mean_recon, merged, D_optim, G_optim],
                                                   {x: x_, keep_prob: 0.8, isTrain: True})
        if num % 10 == 0:
            print("batch:", num)
            fixed_p = root + str(num + 1) + '.png'
            print("D Loss:", loss_d_)
            print("G Loss:", loss_g_)
            print("VAE loss:", _VAE_loss)
            print("KL divergence:", _KL_divergence)
            print("reconstruction_loss:", _reconstruction_loss)
            print("###########")
            if num % 100 == 0:
                show_result((num + 1), fixed_x, save=True, path=fixed_p)
        num += 1

sess.close()
