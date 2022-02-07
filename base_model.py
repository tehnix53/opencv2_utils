import os

import tensorflow as tf
from tensorflow.keras import layers


class BaseModels:

    def __init__(self, model_name, img_h, img_w, weights):

        self.model_name = model_name
        self.img_h = img_h
        self.img_w = img_w
        self.weights = weights

        self.ALLOWED_MODELS = {'efficientnetb0': tf.keras.applications.EfficientNetB0,
                               'efficientnetb1': tf.keras.applications.EfficientNetB1,
                               'efficientnetb2': tf.keras.applications.EfficientNetB2,
                               'efficientnetb3': tf.keras.applications.EfficientNetB3,
                               'mobilenetV2': tf.keras.applications.MobileNetV2,
                               'xception': tf.keras.applications.Xception}

        self.PREPROCESS = {'efficientnetb0': tf.keras.applications.efficientnet.preprocess_input,
                           'efficientnetb1': tf.keras.applications.efficientnet.preprocess_input,
                           'efficientnetb2': tf.keras.applications.efficientnet.preprocess_input,
                           'efficientnetb3': tf.keras.applications.efficientnet.preprocess_input,
                           'mobilenetV2': tf.keras.applications.mobilenet_v2.preprocess_input,
                           'xception': tf.keras.applications.xception.preprocess_input}

        self.FT_AT = {'efficientnetb0': 216,
                      'efficientnetb1': 318,
                      'efficientnetb2': 318,
                      'efficientnetb3': 363,
                      'mobilenetV2': 100,
                      'xception': 105}

        if self.model_name in list(self.ALLOWED_MODELS)[0:4]:
            ALLOWED_WEIGHTS = ['imagenet', 'noisy-student']
        else:
            ALLOWED_WEIGHTS = ['imagenet']

        assert self.model_name in list(self.ALLOWED_MODELS), 'model is not allowed'
        assert self.weights in ALLOWED_WEIGHTS, 'weights is not allowed'

    def load_weights(self):

        # model = self.ALLOWED_MODELS[self.model_name],
        preprocess = self.PREPROCESS[self.model_name]
        if self.weights == 'imagenet':
            print(f'load weights: {self.weights}')
            model = self.ALLOWED_MODELS[self.model_name](input_shape=(self.img_h, self.img_w, 3),
                                                         weights='imagenet',
                                                         include_top=False)

            return model, preprocess

        elif self.weights == 'noisy-student':
            breed = self.model_name + ('_notop.h5')
            path_to_weights = os.path.join('./noisy_student', breed)  ### add dir with weights in root directory
            print(f'load weights: {self.weights}')
            model = self.ALLOWED_MODELS[self.model_name](input_shape=(self.img_h, self.img_w, 3),
                                                         weights=None,
                                                         include_top=False)
            model.load_weights(path_to_weights)

            return model, preprocess

    def build_model(self):
        base_model, preprocess_input = self.load_weights()

        inputs = tf.keras.Input((self.img_h, self.img_w, 3))

        x = preprocess_input(inputs)
        x = base_model(x, training=False)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs, outputs)

        return model

    @staticmethod
    def feature_extracting_state(model):
        model.trainable = False
        model.layers[-1].trainable = True  # unfreeze only out Dense
        return model

    def fine_tuning_state(self, model, last_weigths=None):
        # load last weights

        model.layers[1].trainable = False

        for i in model.layers[1].layers[self.FT_AT[self.model_name]:]:
            if isinstance(i, layers.BatchNormalization):
                i.trainable = False
            else:
                i.trainable = True

        return model

    @staticmethod
    def full_training_state_1(model, last_weights=None):
        # load last weights
        model.layers[1].trainable = True
        for i in model.layers[1].layers:
            if isinstance(i, layers.BatchNormalization):
                i.trainable = False
            else:
                i.trainable = True

        return model

    @staticmethod
    def full_training_state_2(model, last_weights=None):
        # load last weights
        model.trainable = True
        return model
