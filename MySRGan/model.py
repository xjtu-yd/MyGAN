from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataloader import *
from utils import *
import collections
import datetime


# Definition of the generator
def generator(gen_inputs, gen_output_channels, reuse=False, is_training=False, num_resblock=16):
    # The Bx residual blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
            # net = batchnorm(net, FLAGS.is_training)
            net = prelu_tf(net)
            net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
            # net = batchnorm(net, FLAGS.is_training)
            net = net + inputs

        return net

    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(gen_inputs, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, num_resblock + 1, 1):
            name_scope = 'resblock_%d' % (i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv2(net, 3, 64, 1, use_bias=False, scope='conv')
            net = batchnorm(net, is_training)

        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            net = conv2(net, 9, gen_output_channels, 1, scope='conv')

    return net


# Definition of the discriminator
def discriminator(dis_inputs, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            net = batchnorm(net, FLAGS.is_training)
            net = lrelu(net, 0.2)

        return net

    with tf.device('/gpu:0'):
        with tf.variable_scope('discriminator_unit'):
            # The input layer
            with tf.variable_scope('input_stage'):
                net = conv2(dis_inputs, 3, 64, 1, scope='conv')
                net = lrelu(net, 0.2)
            # print(net.get_shape())
            # The discriminator block part
            # block 1
            net = discriminator_block(net, 64, 3, 2, 'disblock_1')

            # block 2
            net = discriminator_block(net, 128, 3, 1, 'disblock_2')

            # block 3
            net = discriminator_block(net, 128, 3, 2, 'disblock_3')

            # block 4
            net = discriminator_block(net, 256, 3, 1, 'disblock_4')

            # block 5
            net = discriminator_block(net, 256, 3, 2, 'disblock_5')

            # block 6
            net = discriminator_block(net, 512, 3, 1, 'disblock_6')

            # block_7
            net = discriminator_block(net, 512, 3, 2, 'disblock_7')
            # print(net.get_shape())
            # The dense layer 1
            with tf.variable_scope('dense_layer_1'):
                # print(net.get_shape())
                # net = slim.flatten(net)
                # print(net.get_shape())
                net = denselayer(net, 1024)
                net = lrelu(net, 0.2)

            # The dense layer 2
            with tf.variable_scope('dense_layer_2'):
                net = denselayer(net, 1)
                net = tf.nn.sigmoid(net)

    return net


# VGG19 net
def vgg_19_activate(inputs,
                    num_classes=1000,
                    is_training=False,
                    dropout_keep_prob=0.5,
                    spatial_squeeze=True,
                    scope='vgg_19',
                    reuse=False,
                    fc_conv_padding='VALID'):
    """Oxford Net VGG 19-Layers version E Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
      fc_conv_padding: the type of padding to use for the fully connected layer
        that is implemented as a convolutional layer. Use 'SAME' padding if you
        are applying the network in a fully convolutional manner and want to
        get a prediction map downsampled by a factor of 32 as an output. Otherwise,
        the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
    Returns:
      the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2', reuse=reuse)  # 重复卷积层2次
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool5')  # 最大池化
            # Use conv2d instead of fully_connected layers.
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)  # 集合转换为字典

            return net, end_points


# VGG19 net
def vgg_19(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           reuse=False,
           fc_conv_padding='VALID'):
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = inputs
            for i in range(2):
                net = slim.conv2d(net, 64, [3, 3], scope='conv1/conv1_%d' % (i + 1), activation_fn=None)
                net = tf.nn.relu(net, name=scope + 'relu1_%d' % (i + 1))
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            for i in range(2):
                net = slim.conv2d(net, 128, [3, 3], scope='conv2/conv2_%d' % (i + 1), activation_fn=None)
                net = tf.nn.relu(net, name='relu2_%d' % (i + 1))
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            for i in range(4):
                net = slim.conv2d(net, 256, [3, 3], scope='conv3/conv3_%d' % (i + 1), activation_fn=None)
                net = tf.nn.relu(net, name='relu3_%d' % (i + 1))
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            for i in range(4):
                net = slim.conv2d(net, 512, [3, 3], scope='conv4/conv4_%d' % (i + 1), activation_fn=None)
                net = tf.nn.relu(net, name='relu4_%d' % (i + 1))
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            for i in range(4):
                net = slim.conv2d(net, 512, [3, 3], scope='conv5/conv5_%d' % (i + 1), activation_fn=None)
                net = tf.nn.relu(net, name='relu5_%d' % (i + 1))
            net = slim.max_pool2d(net, [2, 2], scope='pool5')  # 最大池化
            # Use conv2d instead of fully_connected layers.
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)  # 集合转换为字典

            return net, end_points


def VGG19_slim_activate(input, type, reuse, scope):
    # Define the feature to extract according to the type of perceptual
    if type == 'VGG54':
        target_layer = scope + 'vgg_19/conv5/conv5_4'
    elif type == 'VGG22':
        target_layer = scope + 'vgg_19/conv2/conv2_2'
    else:
        raise NotImplementedError('Unknown perceptual type')
    _, output = vgg_19(input, is_training=False, reuse=reuse)
    output = output[target_layer]

    return output


def VGG19_slim(input, reuse=False, scope=''):
    # Define the feature to extract according to the type of perceptual
    target_layer = 'vgg_19/conv5/conv5_4'
    _, output = vgg_19(input, is_training=False, reuse=reuse)
    # print(output.keys)
    output = output[target_layer]
    return _, output


# Define the whole network architecture
def SRGAN(test_LR, inputs, targets, global_step, FLAGS):
    # Define the container of the parameter
    Network = collections.namedtuple('Network', 'gen_train, \
        d_train, discrim_real_output, discrim_fake_output, discrim_loss, \
        gen_loss, adversarial_loss, content_loss, gen_output, global_step, d_clip, \
        learning_rate, inputs, targets, test_gen')

    # Build the generator part
    with tf.variable_scope('generator'):
        output_channel = targets.shape[-1]
        gen_output = generator(inputs, output_channel, reuse=False, is_training=FLAGS.is_training, num_resblock=FLAGS.num_resblock)

        test_gen = generator(test_LR, 3, reuse=True, is_training=False)
        #test_gen.set_shape([1, None, None, 3])
        # gen_output.set_shape([FLAGS.batch_size, FLAGS.crop_size * 4, FLAGS.crop_size * 4, 3])

    # Build the fake discriminator
    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=False):
            discrim_fake_output = discriminator(gen_output, FLAGS=FLAGS)

    # Build the real discriminator
    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator', reuse=True):
            discrim_real_output = discriminator(targets, FLAGS=FLAGS)

    # Use the VGG54 feature
    with tf.name_scope('vgg19_1') as scope:
        _, extracted_feature_gen = VGG19_slim(gen_output, reuse=False, scope=scope)
    with tf.name_scope('vgg19_2') as scope:
        _, extracted_feature_target = VGG19_slim(targets, reuse=True, scope=scope)

    # Calculating the generator loss
    with tf.variable_scope('generator_loss'):
        # Content loss
        with tf.variable_scope('content_loss'):
            # Compute the euclidean distance between the two features
            # diff = tf.subtract(extracted_feature_gen, extracted_feature_target)
            diff = extracted_feature_gen, extracted_feature_target
            if FLAGS.perceptual_mode == 'MSE':
                content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))  # 按通道累加后求均值
            else:
                content_loss = FLAGS.vgg_scaling * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

        with tf.variable_scope('adversarial_loss'):
            # adversarial_loss = tf.reduce_mean(-tf.log(discrim_fake_output + FLAGS.EPS))
            adversarial_loss = tf.reduce_mean(discrim_fake_output + FLAGS.EPS)

        gen_loss = content_loss + (FLAGS.ratio) * adversarial_loss
        # print(adversarial_loss.get_shape())
        # print(content_loss.get_shape())

    # Calculating the discriminator loss
    with tf.variable_scope('discriminator_loss'):
        # discrim_fake_loss = tf.log(1 - discrim_fake_output + FLAGS.EPS)
        # discrim_real_loss = tf.log(discrim_real_output + FLAGS.EPS)
        #
        # discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))
        discrim_loss = tf.reduce_mean(discrim_real_output - discrim_fake_output)

    # Define the learning rate and global step
    with tf.variable_scope('get_learning_rate_and_global_step'):
        # global_step = tf.contrib.framework.get_or_create_global_step()  # 获取目前模型训练到达的全局步数
        global_step = tf.assign(global_step, global_step + 1)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                                   staircase=FLAGS.stair)  # each decay_step later mul a decat_rate to learning rate
        # incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('dicriminator_train'):
        discrim_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        discrim_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta).minimize(loss=discrim_loss,
                                                                                             var_list=discrim_tvars)
        d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in discrim_tvars]
        # discrim_grads_and_vars = discrim_optimizer.compute_gradients(discrim_loss, discrim_tvars)
        # discrim_train = discrim_optimizer.apply_gradients(discrim_grads_and_vars)

    with tf.variable_scope('generator_train'):
        # Need to wait discriminator to perform train step
        with tf.control_dependencies([discrim_optimizer] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta).minimize(loss=gen_loss,
                                                                                             var_list=gen_tvars)
            # gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            # gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

    # [ToDo] If we do not use moving average on loss??
    # exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    # update_loss = exp_averager.apply([discrim_loss, adversarial_loss, content_loss])

    return Network(
        gen_train=gen_optimizer,
        d_train=discrim_optimizer,
        discrim_real_output=discrim_real_output,
        discrim_fake_output=discrim_fake_output,
        discrim_loss=discrim_loss,
        gen_loss=gen_loss,
        adversarial_loss=adversarial_loss,
        content_loss=content_loss,
        gen_output=gen_output,
        global_step=global_step,
        d_clip=d_clip,
        learning_rate=learning_rate,
        inputs=inputs,
        targets=targets,
        test_gen=test_gen,
    )


def train(FLAGS):
    # Load data for training and testing
    # ToDo Add online downscaling
    data_LR_list, data_HR_list = get_date_list(FLAGS)
    print('Data count = %d' % (len(data_HR_list)))

    input_LR = tf.placeholder(tf.float32, [None, FLAGS.crop_size, FLAGS.crop_size, 3], name="input_LR")
    input_HR = tf.placeholder(tf.float32, [None, FLAGS.crop_size * 4, FLAGS.crop_size * 4, 3], name="input_HR")
    global_step = tf.Variable(0, trainable=False)
    test_LR = tf.placeholder(tf.float32, [1, None, None, 3], name="test_LR")

    test_gen = generator(test_LR, 3, reuse=False, is_training=False)

    # Connect to the network
    Net = SRGAN(test_LR, input_LR, input_HR, global_step, FLAGS)

    print('Finish building the network!!!')

    targets = deprocess(Net.targets)
    outputs = deprocess(Net.gen_output)
    # Convert back to uint8
    converted_inputs = tf.image.convert_image_dtype(Net.inputs, dtype=tf.uint8, saturate=True)  # 转为0~255
    converted_targets = tf.image.convert_image_dtype(targets, dtype=tf.uint8, saturate=True)
    converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    psnr = calculate_psnr_train(converted_targets, converted_outputs)



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        tf.summary.scalar('global_step', Net.global_step)
        tf.summary.scalar('Genertor_loss', Net.gen_loss)
        tf.summary.scalar('Discriminator_loss', Net.discrim_loss)
        tf.summary.scalar('adversarial_loss', Net.adversarial_loss)
        tf.summary.scalar('content_loss', Net.content_loss)
        tf.summary.scalar('PSNR', psnr)
        tf.summary.scalar('learning_rate', Net.learning_rate)

        tf.summary.image('input_summary', converted_inputs, 100)
        tf.summary.image('target_summary', converted_targets, 100)
        tf.summary.image('outputs_summary', converted_outputs, 100)

        merged = tf.summary.merge_all()
        logdir = FLAGS.logs_dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)

        # load vgg_19 model
        vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
        vgg_restore = tf.train.Saver(vgg_var_list)
        vgg_restore.restore(sess, FLAGS.vgg_ckpt)

        count = 0

        if FLAGS.checkpoint:
            print('Loading model from the checkpoint...')
            checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if checkpoint is None:
                print("There is no checkpoint.")
            else:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                count = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
                saver.restore(sess, checkpoint)
                print('Loading model successed!')
        count = 0
        test_img = read_image('./data/test/LR/img_004.png')
        test_img = np.expand_dims(test_img, axis=0)
        for i in range(count, FLAGS.max_epoch):
            step_per_epoch = math.ceil(len(data_HR_list) / FLAGS.batch_size)
            data_shuffle(data_LR_list, data_HR_list, i)
            for j in range(step_per_epoch):
                data_LR, data_HR = get_data_next_batch(data_LR_list, data_HR_list, j, FLAGS.batch_size, FLAGS)
                sess.run(Net.d_clip)
                _, gen_loss, _, d_loss, g_output, PSNR, g_step, lr = sess.run(
                    [Net.gen_train, Net.gen_loss, Net.d_train, Net.discrim_loss, Net.gen_output, psnr, Net.global_step,
                     Net.learning_rate],
                    feed_dict={input_LR: data_LR, input_HR: data_HR, test_LR: test_img})

                print("Epoch: " + str(i) + "/" + str(FLAGS.max_epoch) + "\t" + "step: " + str(j) + "\t"
                      + "g_loss: " + str(gen_loss) + "\t" + "d_loss: " + str(d_loss))
                print("Global step: " + str(g_step) + "\t" + "learning rate: " + str(lr)
                      + "\t" + "PSNR: " + str(PSNR))
                print(" ")
                # print((np.array(data_LR)).shape)
                # print((np.array(g_output)).shape)
                if g_step % FLAGS.summary_freq == 0:
                    summary = sess.run(merged, feed_dict={input_LR: data_LR, input_HR: data_HR})
                    writer.add_summary(summary, g_step)
                    list_LR = os.listdir(FLAGS.test_LR_dir)
                    for t in list_LR:
                        path = os.path.join('./data/test/LR/', t)
                        img = read_image(path)
                        # print(img.shape)
                        img = np.expand_dims(img, axis=0)
                        #print(img.shape)
                        _, gen_loss, _, d_loss, g_output, PSNR, g_step, lr, gen = sess.run(
                            [Net.gen_train, Net.gen_loss, Net.d_train, Net.discrim_loss, Net.gen_output, psnr,
                             Net.global_step, Net.learning_rate, Net.test_gen],
                            feed_dict={input_LR: data_LR, input_HR: data_HR, test_LR: img})
                        #gen = sess.run([test_gen], feed_dict={test_LR: img})

                        # print(type(test_gen))
                        #print((np.array(gen)).shape)
                        save_image(gen, t, FLAGS.sample_dir, g_step, FLAGS.test_HR_dir)

                    if g_step % FLAGS.save_freq == 0:
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'MySR.ckpt'), global_step=g_step)
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'MySR.model'), global_step=g_step)
