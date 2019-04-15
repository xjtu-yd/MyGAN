import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from utils import dense, deconv2d, bn, conv_cond_concat, lrelu, conv2d
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)


def transpose_conv2d(x, w, strides, output):
    return tf.nn.conv2d_transpose(x, w, strides=strides, output_shape=output, padding="SAME")


def generator(image, label, is_training=True, reuse=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse) as scope:
        # batch_size = image.get_shape().as_list()[0]
        batch_size = tf.shape(image)[0]
        image = tf.concat([image, label], 1)  # [bz,zdim+10]
        net = tf.nn.relu(bn(dense(image, 1024, name='g_fc1'), is_training, name='g_bn1'))
        net = tf.nn.relu(bn(dense(net, 128 * 7 * 7, name='g_fc2'), is_training, name='g_bn2'))
        net = tf.reshape(net, [-1, 7, 7, 128])

        output_shape = tf.stack([batch_size, 7, 7, 64])

        # [bz, 14, 14, 64]
        net = tf.nn.relu(
            bn(deconv2d(output_shape, net, 64, 4, 4, 2, 2, padding='SAME', name='g_dc3'), is_training, name='g_bn3'))

        output_shape = tf.stack([batch_size, 14, 14, 1])
        # [bz, 28, 28, 1]
        out = tf.nn.sigmoid(deconv2d(output_shape, net, 1, 4, 4, 2, 2, padding='SAME', name='g_dc4'))
        return out


def load(checkpoint_dir, saver):
    import re
    print(" [*] Reading checkpoints...")
    # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def save(saver, checkpoint_dir, step, sess, model_name):
    checkpoint_dir = os.path.join(checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name + '.model'), global_step=step)


def discriminator(input_image, label, is_training=True, reuse=None):

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        # batch_size = input_image.get_shape().as_list()[0]
        batch_size = tf.shape(input_image)[0]
        ydim = tf.shape(label)[-1]
        y = tf.reshape(label, [batch_size, 1, 1, ydim])
        x = conv_cond_concat(input_image, y)  # [bz, 28, 28, 11]
        # [bz, 14, 14, 64]
        net = lrelu(conv2d(11,x, 64, 4, 4, 2, 2, padding="SAME", name='d_conv1'), name='d_l1')
        # [bz, 7, 7, 128]
        net = lrelu(bn(conv2d(64, net, 128, 4, 4, 2, 2, padding="SAME", name='d_conv2'), is_training, name='d_bn2'),
                    name='d_l2')
        net = tf.reshape(net, [batch_size, 7 * 7 * 128])
        # [bz, 1024]
        net = lrelu(bn(dense(net, 1024, name='d_fc3'), is_training, name='d_bn3'), name='d_l3')
        # [bz, 1]
        yd = dense(net, 1, name='D_dense')
        yc = dense(net, 10, name='C_dense')
        return yd, net, yc


def get_data(g_dim, batch_size):
    g_noise = np.random.normal(0, 1, size=[batch_size, g_dim])
    real_image_batchx, real_image_batchy = mnist.train.next_batch(batch_size)
    # print(real_image_batchx.shape)
    real_image_batchx = real_image_batchx.reshape([-1, 28, 28, 1])
    real_image_batchx = np.transpose(real_image_batchx, (0, 3, 1, 2))
    # print(real_image_batchx.shape)
    real_image_batch = real_image_batchx.reshape([batch_size, 28, 28, 1])
    real_image_batchy = np.expand_dims(real_image_batchy, -2);

    real_image_batchx2, real_image_batchy2 = mnist.train.next_batch(batch_size)
    real_image_batchx2 = real_image_batchx2.reshape([-1, 28, 28, 1])
    real_image_batchx2 = np.transpose(real_image_batchx2, (0, 3, 1, 2))
    real_image_batch2 = real_image_batchx2.reshape([batch_size, 28, 28, 1])
    real_image_batchy2 = np.expand_dims(real_image_batchy2, -2);

    real_image_batchx3, real_image_batchy3 = mnist.train.next_batch(batch_size)
    real_image_batchx3 = real_image_batchx3.reshape([-1, 28, 28, 1])
    real_image_batchx3 = np.transpose(real_image_batchx3, (0, 3, 1, 2))
    real_image_batch3 = real_image_batchx3.reshape([batch_size, 28, 28, 1])
    real_image_batchy3 = np.expand_dims(real_image_batchy3, -2)

    real_image_batchx4, real_image_batchy4 = mnist.train.next_batch(batch_size)
    real_image_batchx4 = real_image_batchx4.reshape([-1, 28, 28, 1])
    real_image_batchx4 = np.transpose(real_image_batchx4, (0, 3, 1, 2))
    real_image_batch4 = real_image_batchx4.reshape([batch_size, 28, 28, 1])
    real_image_batchy4 = np.expand_dims(real_image_batchy4, -2)

    real_image_batch = np.concatenate((real_image_batchx, real_image_batchx2), axis=1)
    real_image_batch2 = np.concatenate((real_image_batchx3, real_image_batchx4), axis=1)
    real_image_batch = np.concatenate((real_image_batch, real_image_batch2), axis=1)
    real_image_batch = np.transpose(real_image_batch, (0, 2, 3, 1))

    real_image_batchy = np.concatenate((real_image_batchy, real_image_batchy2), axis=1)
    real_image_batchy2 = np.concatenate((real_image_batchy3, real_image_batchy4), axis=1)
    real_image_batchy = np.concatenate((real_image_batchy, real_image_batchy2), axis=1)
    real_image_batchy = np.transpose(real_image_batchy, (0, 2, 1))

    return real_image_batch, real_image_batchy


batch_size = 64
g_dim = 100

g_placeholder = tf.placeholder(tf.float32, [None, g_dim], name="g_placeholder")
d_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1], name="d_placeholder")
both_label = tf.placeholder(tf.float32, [None, 10], name="both_label")

G = generator(g_placeholder, both_label, is_training=True)
D_Gr, D_conv_real, D_cls_real = discriminator(d_placeholder, both_label, is_training=True)
D_Gf, D_conv_fake, D_cls_fake = discriminator(G, both_label, is_training=True, reuse=True)


D_loss = tf.reduce_mean(D_Gr - D_Gf)
G_loss = tf.reduce_mean(D_Gf)

# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Gr, labels=tf.ones_like(D_Gr)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Gf, labels=tf.zeros_like(D_Gf)))
# D_loss = D_loss_fake + D_loss_real
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Gf, labels=tf.ones_like(D_Gf)))

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

d_trainner = tf.train.RMSPropOptimizer(0.0001).minimize(D_loss, var_list=d_vars)
g_trainner = tf.train.RMSPropOptimizer(0.0001).minimize(G_loss, var_list=g_vars)

d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

epoch_num = 300

# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True #allocate dynamically
# sess = tf.Session(config = config)

sess = tf.Session()
tf.get_variable_scope().reuse_variables()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
tf.summary.scalar('Genertor_loss', G_loss)
tf.summary.scalar('Discriminator_loss', D_loss)

images_for_tensorboard = generator(g_placeholder, both_label, is_training=False)
tf.summary.image('Generated_images', images_for_tensorboard, 100)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)
checkpoint_dir = "./checkpoint"

could_load, checkpoint_counter = load(checkpoint_dir, saver)
if could_load:
    start_epoch = (int)(checkpoint_counter / 1000)
    start_batch_id = checkpoint_counter - start_epoch * 1000
    counter = checkpoint_counter
    print(" [*] Load SUCCESS")
else:
    start_epoch = 0
    start_batch_id = 0
    counter = 1
    print(" [!] Load failed...")

for g in range(start_epoch, 300000):

    g_noise = np.random.normal(0, 1, size=[batch_size, g_dim])
    real_image_batchx, real_image_batchy = mnist.train.next_batch(batch_size)
    real_image_batch = real_image_batchx.reshape([-1, 28, 28, 1])
    real_image_batchy = real_image_batchy.reshape([-1, 10])
    # print(real_image_batch.shape)
    # print(real_image_batchy.shape)

    sess.run(d_clip)
    _, dLoss, __, gLoss = sess.run([d_trainner, D_loss, g_trainner, G_loss],
                                   feed_dict={g_placeholder: g_noise, d_placeholder: real_image_batch,
                                              both_label: real_image_batchy})

    if g % 1000 == 0:
        print(str(g) + " dLoss:" + str(dLoss) + "gLoss:" + str(gLoss))
        # print(real_image_batchy.shape)
        summary = sess.run(merged,
                           {g_placeholder: g_noise, d_placeholder: real_image_batch, both_label: real_image_batchy})
        writer.add_summary(summary, g)

        real_image_batchy = real_image_batchy[0]
        real_image_batchy = real_image_batchy.reshape([-1, 10])
        # print(real_image_batchy.shape)
        g_noise2 = np.random.normal(0, 1, size=[1, g_dim])
        generated_images = generator(g_noise2, real_image_batchy, is_training=False, reuse=True)
        # print(real_image_batchy.shape)
        image = sess.run(generated_images, feed_dict={g_placeholder: g_noise2, both_label: real_image_batchy})
        # image = np.transpose(image, (0, 3, 1, 2))
        for i in range(1):
            # tmp = np.expand_dims(image[0][i],axis=-1)
            # print(tmp.shape)
            plt.imshow(image[0].reshape([28, 28]), cmap='Greys')
            plt.savefig("./samples/filename" + str(g) + "_" + str(i) + ".png")
        # image_add = 0.25 * image[0][0][:][:] + 0.25 * image[0][1][:][:] + 0.25 * image[0][2][:][:] + 0.25 * image[0][3][:][:]

        # plt.imshow(image_add.reshape([28, 28]), cmap='Greys')
        # plt.savefig("./samples/filename" + str(g) + "_add" + ".png")

        # image = np.transpose(image, (0, 2, 3, 1))
        im = image[0].reshape([-1, 28, 28, 1])
        result = discriminator(d_placeholder, real_image_batchy, is_training=False, reuse=True)
        # print(real_image_batchy.shape)
        estimate = sess.run(result, {d_placeholder: im, both_label: real_image_batchy})
        print("Estimate:", estimate[0])
        # saver.save(sess, '/wgan_' + str(g) + '.cptk')
        save(saver, checkpoint_dir, g, sess, str(g))
