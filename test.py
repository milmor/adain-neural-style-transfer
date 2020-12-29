import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tensorflow debugging logs
import itertools
import tensorflow as tf
import numpy as np
import PIL.Image
from model import StyleTransferNetwork
from utils import convert, tensor_to_image
from hparams import hparams


def run_test(args):
    st_network = StyleTransferNetwork(hparams['test_size'])
    ckpt_dir = os.path.join(args.name, 'pretrained')
    ckpt = tf.train.Checkpoint(network=st_network, step=tf.Variable(0))
    ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
    print('\n################################')
    print('Adain Neural Style Transfer Test')
    print('################################\n')
    print('Restored {} step: {}\n'.format(args.name, str(ckpt.step.numpy())))
    
    dir_size = 'step_{}_{}x{}'.format(str(ckpt.step.numpy()),
                                      str(hparams['test_size'][0]),
                                      str(hparams['test_size'][1]))
    dir_model = 'output_img_{}'.format(args.name)
    out_dir = os.path.join(args.output_path, dir_model, dir_size)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    content_img_list = os.listdir(args.test_content_img)
    style_img_list = os.listdir(args.test_style_img)

    test_list = list(itertools.product(content_img_list, style_img_list))

    for c_file, s_file in test_list:
        content = convert(os.path.join(args.test_content_img, c_file), hparams['test_size'])[tf.newaxis, :]
        style = convert(os.path.join(args.test_style_img, s_file), hparams['test_size'])[tf.newaxis, :]
        output = st_network(content, style, training=False, alpha=hparams['test_alpha'])
        tensor = tensor_to_image(output)
        c_name = os.path.splitext(c_file)[0] 
        s_name = os.path.splitext(s_file)[0]
        save_path = os.path.join(out_dir, '{}_{}'.format(c_name, s_name))
        tensor.save(save_path + '.jpeg')
        print ('Image: {}.jpeg saved'.format(save_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='model')
    parser.add_argument('--test_content_img', default='./images/content_img/')
    parser.add_argument('--test_style_img', default='./images/style_img/')
    parser.add_argument('--output_path', default='./images/')

    args = parser.parse_args()

    run_test(args)

	
if __name__ == '__main__':
	main()
