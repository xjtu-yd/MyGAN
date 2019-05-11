from utils import *




def crop_image(image_LR, image_HR, args):
    input_size = image_LR.shape
    offset_w = random.randint(0, input_size[1] - args.crop_size)
    offset_h = random.randint(0, input_size[0] - args.crop_size)

    image_LR = image_LR[offset_h:offset_h + args.crop_size, offset_w:offset_w + args.crop_size]
    offset_w = offset_w * 4
    offset_h = offset_h * 4
    image_HR = image_HR[offset_h:offset_h + args.crop_size * 4, offset_w:offset_w + args.crop_size * 4]

    return image_LR, image_HR


def flip_image(image_LR, image_HR):
    n = random.randint(0, 3)
    if n == 0:
        image_LR = image_LR
        image_HR = image_HR
    elif n == 1:  # 水平翻转
        image_LR = cv2.flip(image_LR, 1)
        image_HR = cv2.flip(image_HR, 1)
    elif n == 2:  # 垂直翻转
        image_LR = cv2.flip(image_LR, 0)
        image_HR = cv2.flip(image_HR, 0)
    else:  # 水平垂直翻转
        image_LR = cv2.flip(image_LR, -1)
        image_HR = cv2.flip(image_HR, -1)

    return image_LR, image_HR


def get_data_next_batch(data_LR_list, data_HR_list, step, batch_size, args):
    num = len(data_HR_list)
    start = step * batch_size
    end = (step + 1) * batch_size

    batch_LR = None
    batch_HR = None

    for t in range(start, end):
        i = t
        if end >= num:
            i = math.floor(random.uniform(0, num))
        LR = read_image(data_LR_list[i])
        HR = read_image(data_HR_list[i])
        HR = preprocess(HR)
        if args.random_crop:
            LR, HR = crop_image(LR, HR, args)

        if args.flip:
            LR, HR = flip_image(LR, HR)

        LR = np.expand_dims(LR, axis=0)
        HR = np.expand_dims(HR, axis=0)
        if batch_LR is None:
            batch_HR = HR
            batch_LR = LR
        else:
            batch_LR = np.concatenate((batch_LR, LR), axis=0)
            batch_HR = np.concatenate((batch_HR, HR), axis=0)

    return batch_LR, batch_HR

    # print(LR.shape)
    # print(HR.shape)
    # print(batch_LR.shape)
    # print(batch_HR.shape)

    # LR = LR * 255.
    # LR = LR.astype(np.uint8)
    # image = cv2.cvtColor(LR[0], cv2.COLOR_BGR2RGB)
    # cv2.imwrite('./rrr.png', image)
    #
    # HR = deprocess(HR)
    # HR = HR * 255.
    # HR = HR.astype(np.uint8)
    # image = cv2.cvtColor(HR[0], cv2.COLOR_BGR2RGB)
    # cv2.imwrite('./rrrH.png', image)


def get_date_list(args):
    # Check the input directory
    if (args.input_dir_LR == 'None') or (args.input_dir_HR == 'None'):
        raise ValueError('Input directory is not provided')

    if (not os.path.exists(args.input_dir_LR)) or (not os.path.exists(args.input_dir_HR)):
        raise ValueError('Input directory not found')

    image_list_HR = os.listdir(args.input_dir_HR)
    image_list_LR = os.listdir(args.input_dir_LR)

    if len(image_list_LR) == 0:
        raise Exception('No png files in the input directory')

    if len(image_list_LR) != len(image_list_HR):
        raise ValueError('The number of images is different')

    image_list_HR_temp = sorted(image_list_HR)
    image_list_LR_temp = sorted(image_list_LR)

    image_list_LR = [os.path.join(args.input_dir_LR, _) for _ in image_list_LR_temp]
    image_list_HR = [os.path.join(args.input_dir_HR, _) for _ in image_list_HR_temp]

    return image_list_LR, image_list_HR


def data_shuffle(data_LR_list, data_HR_list, seed=5):
    random.seed(seed)
    random.shuffle(data_LR_list)
    random.seed(seed)
    random.shuffle(data_HR_list)
