import argparse
from model import *
from dataloader import *



def get_parse():
    desc = "Tensorflow implementation of MYSRGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='The mode of the model train, test.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size of the model train, test.')
    parser.add_argument('--random_crop', type=bool, default=True, help='Whether perform the random crop')
    parser.add_argument('--flip', type=bool, default=True, help='Whether random flip data augmentation is applied')
    parser.add_argument('--train_HR_dir', type=str, default='./data/train/HR/',
                        help='the directory of training HR images')
    parser.add_argument('--train_LR_dir', type=str, default='./data/train/LR/',
                        help='the directory of training LR images')
    parser.add_argument('--test_HR_dir', type=str, default='./data/test/HR/',
                        help='the directory of testing HR images')
    parser.add_argument('--test_LR_dir', type=str, default='./data/test/LR/',
                        help='the directory of testing LR images')

    parser.add_argument('--logs_dir', type=str, default='./logs', help='the directory of logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='the directory of checkpoint')
    parser.add_argument('--checkpoint', type=bool, default=False, help='the directory of checkpoint')
    parser.add_argument('--result_dir', type=str, default='./result/image_gen/',
                        help='the directory to save the generated images if you test')
    parser.add_argument('--sample_dir', type=str, default='./samples/',
                        help='the directory to save the generated images when training')
    parser.add_argument('--lpips_model', choices=['net-lin', 'net'], default='net-lin',
                        help='the model of lpips ,net-lin or net')
    parser.add_argument('--lpips_net', choices=['alex', 'vgg'], default='alex', help='the net of lpips, alex, or vgg')
    parser.add_argument('--test_item', type=str, default='hello', help='test parser')

    # The parameters of read data
    parser.add_argument('--name_queue_capacity', type=int, default=2048,
                        help='The capacity of the filename queue (suggest large to ensure enough random shuffle.')
    parser.add_argument('--image_queue_capacity', type=int, default=2048,
                        help='The capacity of the image queue (suggest large to ensure enough random shuffle.')
    parser.add_argument('--queue_thread', type=int, default=10,
                        help='The threads of the queue (More threads can speedup the training process.')
    parser.add_argument('--crop_size', type=int, default=24, help='The crop size of the training image')
    # parser.add_argument('--HR_size', type=int, default=96, help='HR images size')
    parser.add_argument('--scale', type=int, default=4, help='the scale of HR to LR')

    # train
    parser.add_argument('--perceptual_mode', type=str, default='VGG54', help='which decides the loss feature map')
    parser.add_argument('--vgg_scaling', type=float, default=0.0061, help='see name')
    parser.add_argument('--EPS', type=float, default=1e-10, help='refuse divie 0')
    parser.add_argument('--ratio', type=float, default=0.001, help='The ratio of adversarial loss')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The learning rate for the network')
    parser.add_argument('--decay_step', type=int, default=5000, help='The steps needed to decay the learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='The decay rate of each decay step')
    parser.add_argument('--stair', type=bool, default=False,
                        help='Whether perform staircase decay. True => decay in discrete interval.')  # false denotes update learning rate each step,true is decay_step to update
    parser.add_argument('--beta', type=float, default=0.9, help='The beta1 parameter for the Adam optimizer')
    parser.add_argument('--max_epoch', type=int, default=50000, help='The max epoch for the training')
    parser.add_argument('--max_iter', type=int, default=1000000, help='The max iteration of the training')
    parser.add_argument('--display_freq', type=int, default=50, help='The diplay frequency of the training process')
    parser.add_argument('--summary_freq', type=int, default=100, help='The frequency of writing summary and test images')
    parser.add_argument('--save_freq', type=int, default=500, help='The frequency of saving checkpoint')
    parser.add_argument('--vgg_ckpt', type=str, default='./vgg19/vgg_19.ckpt',
                        help='path to checkpoint file for the vgg19')
    parser.add_argument('--num_resblock', type=int, default=16,
                        help='How many residual blocks are there in the generator')
    parser.add_argument('--is_training', type=bool, default=True, help='Training => True, Testing => False')

    parser.add_argument('--input_dir_LR', type=str, default='./data/train/LR/')
    parser.add_argument('--input_dir_HR', type=str, default='./data/train/HR/')

    return parser.parse_args()


def main():
    args = get_parse()
    check_folder_all(args)

    # data_LR_list, data_HR_list = get_date_list(args)
    # get_data_next_batch(data_LR_list, data_HR_list, 0, 32, args)
    # print(args)
    # image_list_LR = os.listdir(args.dataset_train_LR_dir)
    # image_list_LR = sorted(image_list_LR)
    # print(image_list_LR)
    if args.mode == 'train':
        train(args)
    #data = data_loader(args)


if __name__ == '__main__':
    main()
