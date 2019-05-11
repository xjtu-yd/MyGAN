from model import *


def load_image(fname):
    image = cv2.imread(fname)
    print(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image)
    return image.astype(np.float32) / 255.0


def test_psnr_ssim():
    path1 = './data/eveluate_H'
    path2 = './result/eveluate_gen'
    calculate_psnr_ssim_dir(path1, path2)


def test_lpips():
    path1 = './data/DIV2K_valid_HR/'  # HR
    path2 = './data/ESRGAN_vaild_res/'  # gen
    # calculate_dir(path1, path2)

    image0_ph = tf.placeholder(tf.float32, name='image0_ph')
    image1_ph = tf.placeholder(tf.float32, name='image1_ph')
    distance_t = lpips(image0_ph, image1_ph, model='net-lin', net='alex')
    with tf.Session() as session:
        calcuate_lpips_dir(session, path1, path2, image0_ph, image1_ph, distance_t)
        # variable_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        # print(variable_names)


def test_lpips_single(img1, img2):
    image0_ph = tf.placeholder(tf.float32, name='image0_ph')
    image1_ph = tf.placeholder(tf.float32, name='image1_ph')
    distance_t = lpips(image0_ph, image1_ph, model='net-lin', net='alex')
    with tf.Session() as session:
        calcuate_lpips_single(session, img1, img2, image0_ph, image1_ph, distance_t)


def test_vgg_19():
    img1 = load_image('./data/eveluate_H/img_001.png')
    # print(img1.shape)
    img = np.expand_dims(img1, axis=0)
    # print(img.shape)
    _, VS = VGG19_slim(img, 'VGG54')
    # _, VS = vgg_19(img, is_training=True)
    with tf.Session() as sess:
        a = tf.trainable_variables()
        print(a)
        sess.run(tf.global_variables_initializer())
        _, res = sess.run([_, VS])
        print(res)
        a = tf.trainable_variables()
        print(a)


def test_save_image():
    img1 = load_image('./data/eveluate_H/img_001.png')
    results = np.expand_dims(img1, axis=0)
    plt.imshow(results[0])
    plt.savefig("./samples/" + str(10) + "_" + str(5) + ".png")
# for i in range(30):
#     n = random.randint(0,1)
#     print(n)

# test_save_image()

# test_lpips()
# test_psnr_ssim()
img1 = load_image('./data/eveluate_H/img_001.png')
img2 = img1
# img2 = load_image('./result/eveluate_gen/img_001_rlt.png')
# calculate_psnr_ssim_single(img1 * 255., img2 * 255.)
test_lpips_single(img1, img2)

"""
  1 - img_001                  . 	LPIPS: 0.208597
  2 - img_002                  . 	LPIPS: 0.277586
  3 - img_003                  . 	LPIPS: 0.256999
  4 - img_005                  . 	LPIPS: 0.139589
Average: LPIPS: 0.220693
PSNR: 18.200703 dB, 	SSIM: 0.367645
LPIPS: 0.199259

"""

'''
Testing RGB channels.
  1 - img_001                  . 	PSNR: 18.230109 dB, 	SSIM: 0.369997
  2 - img_002                  . 	PSNR: 22.141513 dB, 	SSIM: 0.643817
  3 - img_003                  . 	PSNR: 20.713821 dB, 	SSIM: 0.471611
  4 - img_005                  . 	PSNR: 19.464722 dB, 	SSIM: 0.612903
Average: PSNR: 20.137541 dB, SSIM: 0.524582
'''
