import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

def transpose_conv2d(x, w, strides , output):
    return tf.nn.conv2d_transpose(x, w, strides=strides, output_shape=output,padding="SAME")

def generator(image, image_dim, label, reuse_variable = None):
    #label = np.array(label)
    #print(label.shape)
    ch = 1
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variable) as scope:

        w1 = tf.get_variable(name="g_w1", shape=[image_dim, 280*280*ch], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable("g_b1", [280*280*ch], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g1 = tf.matmul(tf.cast(image, tf.float32), w1) + b1
        g1 = tf.reshape(g1, [-1, 280,280, ch]) # [image_dim, 8,8,128]
        g1 = tf.nn.relu(g1)
        #print(g1.shape)


        #print(label.shape)

        label = tf.cast(tf.reshape(label, shape=[-1, 1, 1, 10]), tf.float32)  # 转换数据类型
        label = tf.tile(label, [1, 280, 280, 1])  # 进行张量扩张，对c的第二维和第三维分别重复x_init.shape[1], x_init.shape[2]次,第一维和最后一维保持不变。
       # print(label.shape)
        label = tf.nn.relu(label)
        #print(g1.shape)
        g1 = tf.concat([g1, label], axis=-1)  # 连接，将原始数据加入标签。

        #print(g1.shape)


        w2 = tf.get_variable("g_w2", [3, 3, 11,64], tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable("g_b2", [64], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))

        g2 = tf.nn.conv2d(g1, w2, strides=[1, 2, 2, 1], padding="SAME")  # [image_dim, 112, 112, 128]
        g2 = g2 + b2
        #print(g2.shape)
        #g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5)
        g2 = tf.nn.relu(g2)



        w3 = tf.get_variable("g_w3", [3, 3, 64,32], tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        b3 = tf.get_variable("g_b3", [32], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))

       # batch_size = tf.shape(image)[0]
       # g3 = transpose_conv2d(g2, w3, strides=[1, 2, 2, 1], output = tf.stack([batch_size, 14, 14, 32]))
       # print(g3.shape)
        g3 = tf.nn.conv2d(g2, w3, strides=[1, 5, 5, 1], padding="SAME")  # [image_dim, 56, 56, 32]
        g3 = g3 + b3
        # g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5)
        g3 = tf.nn.relu(g3)
        #print(g3.shape)
        #g3 = tf.transpose(g3, (0, 3, 1, 2))

        w4 = tf.get_variable("g_w4", [3, 3, 32,ch], tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b4 = tf.get_variable("g_b4", [ch], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        #g4 = transpose_conv2d(g3, w4, strides=[1,2,2,1], output=tf.stack([batch_size, 28, 28 , 4]))
       # print(g4.shape)
        g4 = tf.nn.conv2d(g3, w4, strides=[1,1, 1, 1], padding="SAME")  # [image_dim, 28, 28, 4]
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

def discriminator(input_image, label, reuse = None):
    ch = 1
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

        w1 = tf.get_variable("d_w1", [5, 5, ch, 100], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable("d_b1", [100], initializer=tf.constant_initializer(0.0))
        d1 = tf.nn.conv2d(input_image, w1, strides=[1, 2, 2, 1], padding="SAME")
        d1 = d1 + b1
        d1 = tf.nn.relu(d1)
        batch_size = tf.shape(input_image)[0]

        #shp = label.shape.as_list()
        #print(shp)
        label = tf.cast(tf.reshape(label, [-1, 1, 1, 10]), tf.float32)
        #label = tf.cast(np.reshape(label, shape=tf.stack([batch_size, 1, 10, 4])), tf.float32)
        label = tf.tile(label, [1, 14, 14, 10])
       # label = tf.reshape(label, shape=[-1, 14, 14, 100])
        label = tf.nn.relu(label)

        d1 = tf.concat([d1, label], axis=-1)


        w2 = tf.get_variable("d_w2", [5, 5, 200, 50], tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable("d_b2", [50], tf.float32, initializer=tf.constant_initializer(0.0))
        d2 = tf.nn.conv2d(d1, w2, strides=[1, 2, 2, 1], padding="SAME")
        d2 = d2 + b2
        d2 = tf.layers.batch_normalization(d2, training=True, )
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

        d4 = tf.sigmoid(d4)
        return d4




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
d_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1], name = "d_placeholder")
both_label = tf.placeholder(tf.float32, [None, 10], name = "both_label")

G = generator(g_placeholder, g_dim, both_label)
D_Gf = discriminator(G, both_label)
D_Gr = discriminator(d_placeholder, both_label, True)

D_loss = tf.reduce_mean(D_Gr - D_Gf)
G_loss = tf.reduce_mean(D_Gf)

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

images_for_tensorboard = generator(g_placeholder, g_dim, both_label)
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





for g in range(start_epoch, 300000):

    g_noise = np.random.normal(0, 1, size=[batch_size, g_dim])
    real_image_batchx, real_image_batchy = mnist.train.next_batch(batch_size)
    real_image_batch = real_image_batchx.reshape([-1, 28, 28, 1])
    real_image_batchy = real_image_batchy.reshape([-1,10])
   # print(real_image_batch.shape)
   # print(real_image_batchy.shape)

    sess.run(d_clip)
    _, dLoss, __, gLoss = sess.run([d_trainner, D_loss, g_trainner, G_loss], feed_dict={g_placeholder:g_noise, d_placeholder:real_image_batch, both_label:real_image_batchy})

    if g % 1000 == 0 :
        print(str(g) + " dLoss:" + str(dLoss) + "gLoss:" + str(gLoss))
        #print(real_image_batchy.shape)
        summary = sess.run(merged, {g_placeholder:g_noise, d_placeholder:real_image_batch,both_label:real_image_batchy})
        writer.add_summary(summary, g)

        real_image_batchy = real_image_batchy[0]
        real_image_batchy = real_image_batchy.reshape([-1, 10])
        #print(real_image_batchy.shape)
        g_noise2 = np.random.normal(0, 1, size=[1, g_dim])
        generated_images = generator(g_noise2, g_dim, real_image_batchy, True)
        #print(real_image_batchy.shape)
        image = sess.run(generated_images, feed_dict={g_placeholder:g_noise2, both_label:real_image_batchy})
        #image = np.transpose(image, (0, 3, 1, 2))
        for i in range(1):
            #tmp = np.expand_dims(image[0][i],axis=-1)
            #print(tmp.shape)
            plt.imshow(image[0].reshape([28, 28]), cmap='Greys')
            plt.savefig("./samples/filename" + str(g) + "_" + str(i) + ".png")
        #image_add = 0.25 * image[0][0][:][:] + 0.25 * image[0][1][:][:] + 0.25 * image[0][2][:][:] + 0.25 * image[0][3][:][:]

        #plt.imshow(image_add.reshape([28, 28]), cmap='Greys')
        #plt.savefig("./samples/filename" + str(g) + "_add" + ".png")

        #image = np.transpose(image, (0, 2, 3, 1))
        im = image[0].reshape([-1, 28, 28, 1])
        result = discriminator(d_placeholder,real_image_batchy, True)
        #print(real_image_batchy.shape)
        estimate = sess.run(result, {d_placeholder: im,both_label:real_image_batchy})
        print("Estimate:", estimate)
        #saver.save(sess, '/wgan_' + str(g) + '.cptk')
        save(saver, checkpoint_dir, g, sess, str(g))

