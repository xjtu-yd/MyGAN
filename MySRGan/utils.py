from metrics.PSNR_SSIM import *
from metrics.lpips_tf import *
import os
import cv2
import collections
import random
import tensorflow.contrib.slim as slim

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def read_image(fname):
    image = cv2.imread(fname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def preprocess(image):
    # [0, 1] => [-1, 1]
    return image * 2 - 1


def deprocess(image):
    return (image + 1) / 2


def calculate_psnr_ssim_dir(folder_GT, folder_Gen):
    return calculate_dir(folder_GT, folder_Gen)


def calculate_psnr_ssim_dir_train(folder_GT, folder_Gen):
    ndims = folder_GT.shape.ndims
    PSNR = []
    SSIM = []
    if ndims == 4:
        num = folder_GT.shape[0]
        for i in range(num):
            psnr, ssim = calculate_single_train(folder_GT[i], folder_Gen[i])
            PSNR.append(psnr)
            SSIM.append(ssim)
    else:
        return 0., 0.
    return sum(PSNR) / len(PSNR), sum(SSIM) / len(SSIM)


def calculate_lpips_train(folder_GT, folder_Gen):
    ndims = folder_GT.shape.ndims
    LPIPS = []
    if ndims == 4:
        num = folder_GT.shape[0]
        for i in range(num):
            L = test_lpips_single(folder_GT[i], folder_Gen[i])
            LPIPS.append(L)
    else:
        return 0., 0.
    return sum(LPIPS) / len(LPIPS)


def calculate_psnr_ssim_single(img1, img2):
    return calculate_single(img1, img2)


def calcuate_lpips_dir(session, floder_GT, floder_GEN, image0_ph, image1_ph, distance_t):
    return calcuate_dir(session, floder_GT, floder_GEN, image0_ph, image1_ph, distance_t)


def calcuate_lpips_single(session, img1, img2, image0_ph, image1_ph, distance_t):
    return calcuate_single(session, img1, img2, image0_ph, image1_ph, distance_t)


def check_folder_all(args):
    check_folder(args.logs_dir)
    check_folder(args.result_dir)
    check_folder(args.checkpoint_dir)
    check_folder(args.sample_dir)


def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def data_loader(args):
    # with tf.device('/cpu:0'):
    # Define the returned data batches
    Data = collections.namedtuple('Data', 'paths_LR, paths_HR, inputs, targets, image_count, steps_per_epoch')

    # Check the input directory
    if (args.input_dir_LR == 'None') or (args.input_dir_HR == 'None'):
        raise ValueError('Input directory is not provided')

    if (not os.path.exists(args.input_dir_LR)) or (not os.path.exists(args.input_dir_HR)):
        raise ValueError('Input directory not found')

    image_list_HR = os.listdir(args.input_dir_HR)
    image_list_LR = os.listdir(args.input_dir_LR)
    if len(image_list_LR) == 0:
        raise Exception('No png files in the input directory')

    image_list_HR_temp = sorted(image_list_HR)
    image_list_LR_temp = sorted(image_list_LR)

    image_list_LR = [os.path.join(args.input_dir_LR, _) for _ in image_list_LR_temp]
    image_list_HR = [os.path.join(args.input_dir_HR, _) for _ in image_list_HR_temp]

    image_list_LR_tensor = tf.convert_to_tensor(image_list_LR, dtype=tf.string)
    image_list_HR_tensor = tf.convert_to_tensor(image_list_HR, dtype=tf.string)  # 生成一个op

    with tf.variable_scope('load_image'):
        output = tf.train.slice_input_producer([image_list_LR_tensor, image_list_HR_tensor],
                                               shuffle=False, capacity=args.name_queue_capacity)  # 文件名队列

        # Reading and decode the images
        reader = tf.WholeFileReader(name='image_reader')
        image_LR = tf.read_file(output[0])  # 读取图片
        # print(image_LR)
        image_HR = tf.read_file(output[1])
        input_image_LR = tf.image.decode_png(image_LR, channels=3)  # 解码png格式图片,得到uint8格式，范围0~255
        input_image_HR = tf.image.decode_png(image_HR, channels=3)
        input_image_LR = tf.image.convert_image_dtype(input_image_LR, dtype=tf.float32)  # 转换数据类型，【0,1】之间的float
        input_image_HR = tf.image.convert_image_dtype(input_image_HR, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(input_image_LR)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):  # 在执行下面语句之前先执行断言
            input_image_LR = tf.identity(input_image_LR)  # 创建一个计算图中的节点
            input_image_HR = tf.identity(input_image_HR)

        # Normalize the low resolution image to [0, 1], high resolution to [-1, 1]
        a_image = preprocessLR(input_image_LR)
        b_image = preprocess(input_image_HR)

        inputs, targets = [a_image, b_image]

    # The data augmentation part
    with tf.name_scope('data_preprocessing'):
        with tf.name_scope('random_crop'):
            # Check whether perform crop
            if (args.random_crop is True) and args.mode == 'train':
                print('[Config] Use random crop')
                # Set the shape of the input image. the target will have 4X size
                input_size = tf.shape(inputs)
                target_size = tf.shape(targets)
                offset_w = tf.cast(
                    tf.floor(tf.random_uniform([], 0, tf.cast(input_size[1], tf.float32) - args.crop_size)),
                    dtype=tf.int32)
                offset_h = tf.cast(
                    tf.floor(tf.random_uniform([], 0, tf.cast(input_size[0], tf.float32) - args.crop_size)),
                    dtype=tf.int32)

                # if args.task == 'MySRGAN' or args.task == 'SRResnet':
                inputs = tf.image.crop_to_bounding_box(inputs, offset_h, offset_w, args.crop_size,
                                                       args.crop_size)  # 从左上角（offset_h, offset_w）点开始裁剪大小为（ args.crop_size,args.crop_size）的图像
                targets = tf.image.crop_to_bounding_box(targets, offset_h * 4, offset_w * 4, args.crop_size * 4,
                                                        args.crop_size * 4)

            # Do not perform crop
            else:
                inputs = tf.identity(inputs)
                targets = tf.identity(targets)

        with tf.variable_scope('random_flip'):
            # Check for random flip:
            if (args.flip is True) and (args.mode == 'train'):
                print('[Config] Use random flip')
                # Produce the decision of random flip
                decision = tf.random_uniform([], 0, 1, dtype=tf.float32)  # 产生一个【0,1】范围的随机数，决定是否对图像进行翻转

                input_images = random_flip(inputs, decision)
                target_images = random_flip(targets, decision)
            else:
                input_images = tf.identity(inputs)
                target_images = tf.identity(targets)

        input_images.set_shape([args.crop_size, args.crop_size, 3])
        target_images.set_shape([args.crop_size * 4, args.crop_size * 4, 3])  # 设定图片形状

    if args.mode == 'train':  # 打乱队列之后的数据取batch
        paths_LR_batch, paths_HR_batch, inputs_batch, targets_batch = tf.train.shuffle_batch(
            [output[0], output[1], input_images, target_images],
            batch_size=args.batch_size, capacity=args.image_queue_capacity + 4 * args.batch_size,
            min_after_dequeue=args.image_queue_capacity, num_threads=args.queue_thread)
    else:
        paths_LR_batch, paths_HR_batch, inputs_batch, targets_batch = tf.train.batch(
            [output[0], output[1], input_images, target_images],
            batch_size=args.batch_size, num_threads=args.queue_thread, allow_smaller_final_batch=True)

    steps_per_epoch = int(math.ceil(len(image_list_LR) / args.batch_size))
    inputs_batch.set_shape([args.batch_size, args.crop_size, args.crop_size, 3])
    targets_batch.set_shape([args.batch_size, args.crop_size * 4, args.crop_size * 4, 3])

    return Data(
        paths_LR=paths_LR_batch,
        paths_HR=paths_HR_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        image_count=len(image_list_LR),
        steps_per_epoch=steps_per_epoch
    )


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def read_img(path):
    # read image by cv2 or from lmdb
    # return: Numpy float32, HWC, BGR, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                               biases_initializer=None)


# Define our tensorflow version PRelu
def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5  # <=0

    return pos + neg


# Define our Lrelu
def lrelu(inputs, alpha):
    return tf.nn.leaky_relu(inputs, alpha=alpha, name=None)
    # return keras.layers.LeakyReLU(alpha=alpha).call(inputs)


def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                           scale=False, fused=True, is_training=is_training)


# Our dense layer
def denselayer(inputs, output_size):
    output = tf.layers.dense(inputs, output_size, activation=None,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    return output


def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)


# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output


def save_image(img, name, path, step, HR_path):
    fullpath = os.path.join(path, str(step) + '-' + name)
    img = deprocess(img)
    img = img * 255
    img = img.astype(np.uint8)
    image = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
    cv2.imwrite(fullpath, image)
    img_hr = read_image(os.path.join(HR_path, name))
    calculate_psnr_ssim_single(img_hr * 255, img[0])

