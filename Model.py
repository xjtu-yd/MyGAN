import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/")


def generator(image, image_dim, reuse_variable = None):
    a = 7
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variable) as scope:
        w1 = tf.get_variable(name="g_w1", shape=[image_dim, 224*224*1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable("g_b1", [224*224], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g1 = tf.matmul(tf.cast(image, tf.float32), w1) + b1
        g1 = tf.reshape(g1, [-1, 224,224,1]) # [image_dim, 8,8,128]
        g1 = tf.nn.relu(g1)
        #print(g1.shape)

        w2 = tf.get_variable("g_w2", [3, 3, 1,64], tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable("g_b2", [64], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))

        g2 = tf.nn.conv2d(g1, w2, strides=[1, 2, 2, 1], padding="SAME")  # [image_dim, 112, 112, 128]
        g2 = g2 + b2
        #print(g2.shape)
        #g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5)
        g2 = tf.nn.relu(g2)

        w3 = tf.get_variable("g_w3", [3, 3, 64, 32], tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        b3 = tf.get_variable("g_b3", [32], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g3 = tf.nn.conv2d(g2, w3, strides=[1, 2, 2, 1], padding="SAME")  # [image_dim, 56, 56, 32]
        g3 = g3 + b3
        # g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5)
        g3 = tf.nn.relu(g3)
        #print(g3.shape)
        #g3 = tf.transpose(g3, (0, 3, 1, 2))

        w4 = tf.get_variable("g_w4", [3, 3, 32, 4], tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b4 = tf.get_variable("g_b4", [4], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g4 = tf.nn.conv2d(g3, w4, strides=[1, 2, 2, 1], padding="SAME")  # [image_dim, 28, 28, 4]
        g4 = g4 + b4
        # g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5)

        g4 = tf.nn.sigmoid(g4)


        return g4

def load(checkpoint_dir, saver):
    import re
    print(" [*] Reading checkpoints...")
    #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

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

def discriminator(input_image, reuse = None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        w1 = tf.get_variable("d_w1", [5, 5, 4, 100], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable("d_b1", [100], initializer=tf.constant_initializer(0.0))
        d1 = tf.nn.conv2d(input_image, w1, strides=[1, 2, 2, 1], padding="SAME")
        d1 = d1 + b1
        #d1 = tf.layers.batch_normalization(d1)
        d1 = tf.nn.relu(d1)

        w2 = tf.get_variable("d_w2", [5, 5, 100, 50], tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable("d_b2", [50], tf.float32, initializer=tf.constant_initializer(0.0))
        d2 = tf.nn.conv2d(d1, w2, strides=[1, 2, 2, 1], padding="SAME")
        d2 = d2 + b2
        #d2 = tf.layers.batch_normalization(d2, training=True, )
        d2 = tf.nn.relu(d2)

        w3 = tf.get_variable("d_w3", [7*7*50 , 1024], tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b3 = tf.get_variable("d_b3", [1024], tf.float32, initializer=tf.constant_initializer(0.0))
        d3 = tf.reshape(d2, [-1, 7*7*50])
        d3 = tf.matmul(d3, w3)
        d3 = d3 + b3
        d3 = tf.nn.relu(d3)

        w4 = tf.get_variable("d_w4", [1024, 1], tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b4 = tf.get_variable("d_b4", [1], tf.float32, initializer=tf.constant_initializer(0.0))
        d4 = tf.matmul(d3, w4)
        d4 = d4 + b4

        return d4


batch_size = 64
g_dim = 100

g_placeholder = tf.placeholder(tf.float32, [None, g_dim], name="g_placeholder")
d_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 4], name = "d_placeholder")

G = generator(g_placeholder, g_dim)
D_Gf = discriminator(G)
D_Gr = discriminator(d_placeholder, True)

D_loss = tf.reduce_mean(D_Gr - D_Gf)
G_loss = tf.reduce_mean(D_Gf)

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

d_trainner = tf.train.RMSPropOptimizer(0.0001).minimize(D_loss, var_list=d_vars)
g_trainner = tf.train.RMSPropOptimizer(0.0001).minimize(G_loss, var_list=g_vars)

d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

epoch_num = 300

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)

tf.get_variable_scope().reuse_variables()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
tf.summary.scalar('Genertor_loss', G_loss)
tf.summary.scalar('Discriminator_loss', D_loss)

images_for_tensorboard = generator(g_placeholder, g_dim)
tf.summary.image('Generated_images', images_for_tensorboard, 100)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for g in range(start_epoch, 100000):
    g_noise = np.random.normal(0, 1, size = [batch_size, g_dim])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    real_image_batch = np.transpose(real_image_batch, (0, 3, 1, 2))
    real_image_batch = np.concatenate((real_image_batch, real_image_batch), axis=1)
    real_image_batch = np.concatenate((real_image_batch, real_image_batch), axis=1)
    real_image_batch = np.transpose(real_image_batch, (0, 2, 3, 1))
    sess.run(d_clip)
    _, dLoss, __, gLoss = sess.run([d_trainner, D_loss, g_trainner, G_loss], feed_dict={g_placeholder:g_noise, d_placeholder:real_image_batch})

    if g % 1000 == 0 :
        print(str(g) + " dLoss:" + str(dLoss) + "gLoss:" + str(gLoss))
        summary = sess.run(merged, {g_placeholder:g_noise, d_placeholder:real_image_batch})
        writer.add_summary(summary, g)

        g_noise2 = np.random.normal(0, 1, size=[1, g_dim])
        generated_images = generator(g_noise2, g_dim, True)
        image = sess.run(generated_images, feed_dict={g_placeholder:g_noise2})
        image = np.transpose(image, (0, 3, 1, 2))
        for i in range(4):
            #tmp = np.expand_dims(image[0][i],axis=-1)
            #print(tmp.shape)
            plt.imshow(image[0][i], cmap='Greys')
            plt.savefig("./samples/filename" + str(g) + "_" + str(i) + ".png")
        image_add = 0.25 * image[0][0][:][:] + 0.25 * image[0][1][:][:] + 0.25 * image[0][2][:][:] + 0.25 * image[0][3][:][:]
        image_add = sigmoid(image_add)

        plt.imshow(image_add.reshape([28, 28]), cmap='Greys')
        plt.savefig("./samples/filename" + str(g) + "_add" + ".png")

        image = np.transpose(image, (0, 2, 3, 1))
        im = image[0].reshape([1, 28, 28, 4])
        result = discriminator(d_placeholder, True)
        estimate = sess.run(result, {d_placeholder: im})
        print("Estimate:", estimate)
        #saver.save(sess, '/wgan_' + str(g) + '.cptk')
        save(saver, checkpoint_dir, g, sess, str(g))

