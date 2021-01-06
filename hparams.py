hparams = {'batch_size': 8,
           'content_weight': 1e-7,
           'style_weight': 5e-6,
           'initial_learning_rate': 1e-4,
           'decay_rate': 5e-5,
           'resize_smallest_dim': 512,
           'resize_dim': (512, 512, 3), # Size of content images before cropping
           'input_size': (256, 256), # preprocessing.RandomCrop size
           'style_layers': ['block1_conv1',
                            'block2_conv1',
                            'block3_conv1',
                            'block4_conv1'],
           'content_layer_index': -1, # Index of content layer
           'test_size': (512, 512),
           'test_alpha': 1.0
}
