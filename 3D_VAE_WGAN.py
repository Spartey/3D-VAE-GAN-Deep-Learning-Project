import os, time, itertools
import numpy as np
import matplotlib
import tensorflow as tf
import getTrain

# matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 100
D_lr = 5e-5
G_lr = 1e-4
train_epoch = 20
n_latent = 8
alpha_1 = 5
alpha_2 = 5e-4


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)


def encoder(x, keep_prob=0.5, isTrain=True):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):  # 64 * 64 * 4
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')  # 32 * 32 * 128
        lrelu1 = tf.nn.elu(conv1)

        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')  # 16 * 16 *256
        lrelu2 = tf.nn.elu(tf.layers.batch_normalization(conv2, training=isTrain))

        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')  # 8 * 8 * 512
        lrelu3 = tf.nn.elu(tf.layers.batch_normalization(conv3, training=isTrain))

        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')  # 4 * 4 * 1024
        lrelu4 = tf.nn.elu(tf.layers.batch_normalization(conv4, training=isTrain))

        conv5 = tf.layers.conv2d(lrelu4, 32, [4, 4], strides=(1, 1), padding='valid')  # 1 * 1 * 32
        lrelu5 = tf.nn.elu(tf.layers.batch_normalization(conv5, training=isTrain))

        x = tf.nn.dropout(lrelu5, keep_prob)
        x = tf.contrib.layers.flatten(x)
        z_mu = tf.layers.dense(x, units=n_latent)
        z_sig = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = z_mu + tf.multiply(epsilon, tf.exp(z_sig))

        return z, z_mu, z_sig


def generator(x, isTrain=True):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        # 1st hidden layer
        conv1 = tf.layers.conv3d_transpose(x, 256, [2, 2, 2], strides=(1, 1, 1), padding='valid', use_bias=False)  # (-1, 2, 2, 2, 256)
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv3d_transpose(lrelu1, 128, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False) # (-1, 4, 4, 4, 128)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv3d_transpose(lrelu2, 64, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False)  # (-1, 8, 8, 8, 64)
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv3d_transpose(lrelu3, 32, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False)  # (-1, 16, 16, 16, 32)
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv3d_transpose(lrelu4, 1, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False)  # (-1, 32, 32, 32, 1)
        o = tf.nn.tanh(conv5)

        return o


def discriminator(x, isTrain=True):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):  # (-1, 32, 32,, 32, 1)
        # 1st hidden layer
        conv1 = tf.layers.conv3d(x, 128, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False)  # (-1, 16, 16, 16, 128)
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv3d(lrelu1, 256, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False)  # (-1, 8, 8, 8, 256)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv3d(lrelu2, 512, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False)  # (-1, 4, 4, 4, 512)
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # output layer
        conv4 = tf.layers.conv3d(lrelu3, 1, [4, 4, 4], strides=(1, 1, 1), padding='valid', use_bias=False)
        o = tf.nn.sigmoid(conv4)

        return o, conv4


# variables : input
x_image = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
x_3D = tf.placeholder(tf.float32, shape=(None, 32, 32, 32, 1))

keep_prob = tf.placeholder(dtype=tf.float32)
isTrain = tf.placeholder(dtype=tf.bool)

# networks : encoder
z, z_mu, z_sig = encoder(x_image, keep_prob, isTrain)
z = tf.reshape(z, (-1, 1, 1, 1, n_latent))
# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x_3D, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain)

# loss for each network

reconstruction_loss = tf.reduce_sum(tf.squared_difference(tf.reshape(G_z, (-1, 32 * 32 * 32)), tf.reshape(x_3D, (-1, 32 * 32 * 32))),
                                    1)
KL_divergence = -0.5 * tf.reduce_sum(1.0 + 2.0 * z_sig - z_mu ** 2 - tf.exp(2.0 * z_sig), 1)
mean_KL = tf.reduce_sum(KL_divergence)
mean_recon = tf.reduce_sum(reconstruction_loss)

VAE_loss = tf.reduce_mean(alpha_1 * KL_divergence + alpha_2 * reconstruction_loss)


D_loss_real = tf.reduce_mean(D_real_logits)
D_loss_fake = tf.reduce_mean(D_fake_logits)
D_loss = D_loss_real - D_loss_fake
G_loss = -tf.reduce_mean(D_fake_logits)
# sub_loss = G_loss + VAE_loss

tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss', G_loss)

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]
E_vars = [var for var in T_vars if var.name.startswith('encoder')]

clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in D_vars]


# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.RMSPropOptimizer(D_lr).minimize(-D_loss, var_list=D_vars)
    G_optim = tf.train.RMSPropOptimizer(G_lr).minimize(G_loss, var_list=G_vars)
    E_optim = tf.train.AdamOptimizer(G_lr).minimize(VAE_loss, var_list=E_vars)
    # E_optim = tf.train.RMSPropOptimizer(lr).minimize(VAE_loss, var_list=E_vars)


# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

logger = tf.summary.FileWriter('./graphs', sess.graph)
merged = tf.summary.merge_all()

# results save folder
root = 'VAE_WGAN_results/'
if not os.path.isdir(root):
    os.mkdir(root)

# load dataset
image_path = "./grey-office-chair-image/"
model_path = "./office-chair-model/"
dataset = getTrain.getData(image_path, model_path)

# training-loop
num = 0
for batch in range(100):
    x_im, x_3d = dataset.get_batch(10)
    for _ in range(4):
        sess.run(D_optim, feed_dict={x_image: x_im, x_3D: x_3d, keep_prob: 0.8, isTrain: True})
        sess.run(clip)
    loss_d_, loss_g_, _VAE_loss, _KL_divergence, _reconstruction_loss, summary, _, _, _ = \
        sess.run([D_loss, G_loss, VAE_loss, mean_KL, mean_recon, merged, D_optim, G_optim, E_optim],
                 {x_image: x_im, x_3D: x_3d, keep_prob: 0.8, isTrain: True})
    sess.run(clip)
    print("D Loss:", loss_d_)
    print("G Loss:", loss_g_)
    print("VAE loss:", _VAE_loss)
    print("KL divergence:", _KL_divergence)
    print("reconstruction_loss:", _reconstruction_loss)
    print("###########")
sess.close()


