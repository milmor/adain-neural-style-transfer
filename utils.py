import tensorflow as tf
import numpy as np
import PIL.Image
import os
import json
from hparams import hparams

def deprocess(img):
  img = tf.clip_by_value(img, 0, 255)
  return tf.cast(img, tf.uint8)

def resize_convert(file_path):
    new_val = hparams['resize_smallest_dim']
    img = tf.io.read_file(file_path)  
    img = tf.image.decode_jpeg(img, channels=3)
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    if height < width:
        new_height = new_val
        new_width  = int(width * new_val / height)
    else:
        new_width  = new_val
        new_height = int(height * new_val / width)
    img = tf.image.resize(img, (new_height, new_width))
    img = tf.image.random_crop(img, hparams['resize_dim'])    
    return img

def convert(file_path, shape=hparams['corp_size']):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, shape)
    return img

def tensor_to_image(tensor):
    tensor = tf.clip_by_value(tensor, 0, 255)
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def content_loss(output_features, adain):
    output_features = tf.cast(output_features, tf.float32) # mixed precision
    adain = tf.cast(adain, tf.float32) # mixed precision
    return tf.reduce_sum(
            tf.reduce_mean(tf.square(output_features - adain), axis=[1, 2]))

def style_loss(output_features, style_features, epsilon=1e-5):
    s_loss = 0  
    for o_feat, s_feat in zip(output_features, style_features):
        o_feat = tf.cast(o_feat, tf.float32) # mixed precision
        s_feat = tf.cast(s_feat, tf.float32) # mixed precision
        o_mean, o_var = tf.nn.moments(o_feat, [1, 2])
        s_mean, s_var = tf.nn.moments(s_feat, [1, 2])
        o_std = tf.sqrt(tf.add(o_var, epsilon))
        s_std = tf.sqrt(tf.add(s_var, epsilon))
        l2_mean  = tf.reduce_sum(tf.square(o_mean - s_mean))
        l2_std = tf.reduce_sum(tf.square(o_std - s_std))
        s_loss += l2_mean + l2_std
    return s_loss

def save_hparams(model_name):
    json_hparams = json.dumps(hparams)
    f = open(os.path.join(model_name, '{}_hparams.json'.format(model_name)), 'w')
    f.write(json_hparams)
    f.close()
