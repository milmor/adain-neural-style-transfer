hparams = {'batch_size': 8,
           'content_weight': 1e-7, #
           'style_weight': 1e-6, #
           'initial_learning_rate': 1e-4,
           'decay_rate': 5e-5,
           'resize_smallest_dim': 512,
           'resize_dim': (512, 512, 3),
           'corp_size': (256, 256),
           'test_size': (512, 512),
           'test_alpha': 1.0,
}