'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Oct 2020
'''
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.applications import vgg19
import numpy as np
import time
from model import StyleTransferNetwork
from utils import convert, resize_convert, style_loss, content_loss, save_hparams, deprocess
from hparams import hparams

# Initialize DNN
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

AUTOTUNE = tf.data.experimental.AUTOTUNE
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def create_ds(args):
    content_list_ds = tf.data.Dataset.list_files(str(args.content_dir + '*.jpg'), shuffle=True)
    content_len = tf.data.experimental.cardinality(content_list_ds)
    content_list_ds = content_list_ds.cache().shuffle(content_len)
    content_images_ds = content_list_ds.map(resize_convert, num_parallel_calls=AUTOTUNE)

    style_list_ds = tf.data.Dataset.list_files(str(args.style_dir + '*.jpg'), shuffle=True)
    style_len = tf.data.experimental.cardinality(style_list_ds)
    style_list_ds = style_list_ds.cache().shuffle(style_len)
    style_images_ds = style_list_ds.map(resize_convert, num_parallel_calls=AUTOTUNE)

    print('Total content images: {}'.format(content_len.numpy()))
    print('Total style images: {}'.format(style_len.numpy()))

    ds = tf.data.Dataset.zip((content_images_ds, style_images_ds))
    return ds.repeat().batch(hparams['batch_size']).prefetch(AUTOTUNE)


def create_test_batch(args):# Paper original content-style images
    test_content_img = ['avril_cropped.jpg', 
                        'cornell_cropped.jpg',
                        'chicago_cropped.jpg']
    test_style_img = ['picasso.png', 
                      'woman_with_hat_matisse_cropped.jpg',
                      'ashville_cropped.jpg']

    test_content_batch = tf.concat(
        [convert(os.path.join(args.test_content_img, img))[tf.newaxis, :] for img in test_content_img], axis=0)
    test_style_batch = tf.concat(
        [convert(os.path.join(args.test_style_img, img))[tf.newaxis, :] for img in test_style_img], axis=0)

    return test_content_batch, test_style_batch


def run_training(args):   
    st_network = StyleTransferNetwork(hparams['input_size'])

    learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(hparams['initial_learning_rate'], 
                                                                   decay_steps = 1.0, 
                                                                   decay_rate = hparams['decay_rate'], 
                                                                   staircase=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

    ckpt_dir = os.path.join(args.name, 'pretrained')
    ckpt = tf.train.Checkpoint(network=st_network,
                               optimizer=optimizer,        
                               step=tf.Variable(0))
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, 
                                              max_to_keep=args.max_ckpt_to_keep)
                                              
    ckpt.restore(ckpt_manager.latest_checkpoint)
    log_dir = os.path.join(args.name, 'log_dir')

    print('\n#################################')
    print('Adain Neural Style Transfer Train')
    print('#################################\n')
    if ckpt_manager.latest_checkpoint:
        print('Restored {} from: {}'.format(args.name, ckpt_manager.latest_checkpoint))
    else:
        print('Initializing {} from scratch'.format(args.name))
        save_hparams(args.name)
    print('Start TensorBoard with: $ tensorboard --logdir ./\n')

    writer = tf.summary.create_file_writer(log_dir)
    total_loss_avg = tf.keras.metrics.Mean()
    content_loss_avg = tf.keras.metrics.Mean()
    style_loss_avg = tf.keras.metrics.Mean()
    
    dataset = create_ds(args)
    test_content_batch, test_style_batch = create_test_batch(args)

    @tf.function
    def test_step(content, style):
        prediction = st_network(content, style, training=False)   
        return deprocess(prediction)

    @tf.function
    def train_step(content, style):
        with tf.GradientTape() as tape:
            g_t = st_network(content, style)
            g_t_feats = st_network.encoder(g_t)
            c_loss = hparams['content_weight']*content_loss(g_t_feats[-1], st_network.t)
            s_loss = hparams['style_weight']*style_loss(g_t_feats, st_network.s_feats)        
            total_loss = c_loss + s_loss
            scaled_loss = optimizer.get_scaled_loss(total_loss)

        scaled_gradients = tape.gradient(scaled_loss, st_network.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        #gradients = tape.gradient(total_loss, st_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, st_network.trainable_variables))
        return total_loss, c_loss, s_loss

    total_start = time.time()
    for (content, style) in dataset:
        start = time.time()
        total_loss, c_loss, s_loss = train_step(content, style)
        total_loss_avg.update_state(total_loss)
        content_loss_avg.update_state(c_loss)
        style_loss_avg.update_state(s_loss)
        ckpt.step.assign_add(1)
        step_int = int(ckpt.step) # cast ckpt.step

        if (step_int) % args.ckpt_interval == 0:
            print('Time taken for step {} is {} sec'.format(step_int, time.time()-start))
            ckpt_manager.save(step_int)

            prediction_norm = test_step(test_content_batch, test_style_batch) 

            with writer.as_default():
                tf.summary.scalar('total loss', total_loss_avg.result(), step=step_int)
                tf.summary.scalar('content loss', content_loss_avg.result(), step=step_int)
                tf.summary.scalar('style loss', style_loss_avg.result(), step=step_int)     
                tf.summary.scalar('learning rate', optimizer.learning_rate(step_int), 
                                                                           step=step_int) 
                images = np.reshape(prediction_norm, (-1, hparams['input_size'][0], 
                                                          hparams['input_size'][1], 3))
                tf.summary.image('generated image', images, step=step_int, max_outputs=3)

            print('Step {} Loss: {:.4f}'.format(step_int, total_loss_avg.result())) 
            print('Loss content: {:.4f}'.format(content_loss_avg.result()))
            print('Loss style: {:.4f}'.format(style_loss_avg.result()))
            print('Learning rate: {}'.format(optimizer.learning_rate(step_int)))
            print('Total time: {} sec\n'.format(time.time()-total_start))
            total_loss_avg.reset_states() # reset mixed precision nan
            content_loss_avg.reset_states() # reset mixed precision nan
            style_loss_avg.reset_states() # reset mixed precision nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', default='./ms-coco/')
    parser.add_argument('--style_dir', default='./wikiart/')
    parser.add_argument('--name', default='model')
    parser.add_argument('--examples_dir', default='./examples/')
    parser.add_argument('--ckpt_interval', type=int, default=250)
    parser.add_argument('--max_ckpt_to_keep', type=int, default=20)
    parser.add_argument('--test_content_img', default='./images/content_img/')
    parser.add_argument('--test_style_img', default='./images/style_img/')
    
    args = parser.parse_args()

    run_training(args)

	
if __name__ == '__main__':
	main()
