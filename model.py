'''Style Transfer Network model for Tensorflow.

# Reference paper
- Xun Huang and Serge Belongie. 
  [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization]
  (https://arxiv.org/abs/1703.06868) (ICCV 2017)

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Oct 2020
'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg19


class ConvReflect(tf.keras.layers.Layer):
    # 2D Convolution layer with reflection padding
    def __init__(self, filters, kernel_size, strides=(1, 1), 
                 activation=None,
                 kernel_initializer='glorot_normal'):
        super(ConvReflect, self).__init__()
        self.size_pad = kernel_size // 2
        self.padding = tf.constant([[0, 0], 
                                    [self.size_pad, self.size_pad], 
                                    [self.size_pad, self.size_pad], 
                                    [0, 0]])
        self.conv2d = layers.Conv2D(filters, kernel_size, strides,
                                    activation=activation,
                                    kernel_initializer=kernel_initializer)

    def call(self, x):
        x = tf.pad(x, self.padding, 'REFLECT') 
        return self.conv2d(x)


def Decoder(input_shape = (None, None, 512), initializer = 'glorot_normal'):
    inputs = tf.keras.Input(shape=input_shape)

    x = ConvReflect(256, 3, activation='relu',
                    kernel_initializer=initializer)(inputs)
    x = layers.UpSampling2D(2)(x)

    x = ConvReflect(256, 3, activation='relu',
                    kernel_initializer=initializer)(x)
    x = ConvReflect(256, 3, activation='relu',
                    kernel_initializer=initializer)(x)
    x = ConvReflect(256, 3, activation='relu',
                    kernel_initializer=initializer)(x)
    x = ConvReflect(128, 3, activation='relu',
                    kernel_initializer=initializer)(x)    
    x = layers.UpSampling2D(2)(x)  

    x = ConvReflect(128, 3, activation='relu',
                    kernel_initializer=initializer)(x)   
    x = ConvReflect(64, 3, activation='relu',
                    kernel_initializer=initializer)(x)   
    x = layers.UpSampling2D(2)(x)

    x = ConvReflect(64, 3, activation='relu',
                    kernel_initializer=initializer)(x)   
    x = ConvReflect(3, 3, kernel_initializer=initializer)(x)
    outputs = layers.Activation('linear', dtype='float32')(x)

    return tf.keras.models.Model(inputs, outputs)


def AdaIN(content, style, epsilon=1e-5):
    #content = tf.cast(content, tf.float32) # optional mixed precision
    #style = tf.cast(style, tf.float32) # optional mixed precision
    c_mean, c_var = tf.nn.moments(content, [1, 2], keepdims=True)
    s_mean, s_var = tf.nn.moments(style, [1, 2], keepdims=True)

    c_std = tf.sqrt(tf.add(c_var, epsilon))
    s_std = tf.sqrt(tf.add(s_var, epsilon))

    return s_std * ((content - c_mean) / c_std) + s_mean


class VGGEncoder(tf.keras.models.Model):
    def __init__(self, style_layers=['block1_conv1',
                                     'block2_conv1',
                                     'block3_conv1',
                                     'block4_conv1']):
        super(VGGEncoder, self).__init__()
        vgg = vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        #model_outputs = [vgg.get_layer(name).output for name in style_layers]
        # mixed precision cast
        model_outputs = [tf.cast(vgg.get_layer(name).output, tf.float32) for name in style_layers]
        self.model = tf.keras.models.Model([vgg.input], model_outputs)

    def call(self, x):
        x = vgg19.preprocess_input(x)
        return self.model(x)


class StyleTransferNetwork(tf.keras.models.Model):
    def __init__(self, input_size=(256, 256),
                 style_layers=['block1_conv1',
                               'block2_conv1',
                               'block3_conv1',
                               'block4_conv1'],
                 content_layer_index=-1):
        super(StyleTransferNetwork, self).__init__()
        self.encoder = VGGEncoder(style_layers)
        self.encoder.trainable = False
        self.decoder = Decoder()
        self.corp_layer = tf.keras.Sequential([   
            layers.experimental.preprocessing.RandomCrop(input_size[0], 
                                                         input_size[1])
        ])
        self.c_layer_idx = content_layer_index

    def call(self, content, style, alpha=1.0, training=True):     
        content_input = self.corp_layer(content, training=training)
        style_input = self.corp_layer(style, training=training)

        self.c_feat = self.encoder(content_input)[self.c_layer_idx]
        self.s_feats = self.encoder(style_input)
        self.t = (1 - alpha) * self.c_feat + alpha * AdaIN(self.c_feat, self.s_feats[self.c_layer_idx])    
        return self.decoder(self.t)  
