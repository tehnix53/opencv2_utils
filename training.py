import datetime
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from config import MODEL


class TrainModel:

    def __init__(self, logdir, modelname=MODEL):
        self.logdir = logdir
        self.modelname = modelname

    def create_subdirs(self):
        create = ['FT', 'FE', 'F1', 'F2']

        for i in create:
            new = os.path.join(self.logdir, i)
            if not os.path.exists(new):
                os.mkdir(new)
                print(f'create {new}')

    def compile_model(self, model, base_learning_rate, epochs):
        return model.compile(optimizer=tf.keras.optimizers.Adam(base_learning_rate,
                                                                decay=base_learning_rate // epochs),
                             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                                     label_smoothing=0.01),
                             metrics=['accuracy'])

    @staticmethod
    def create_plot(history, model_basepath):

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')

        print(model_basepath)
        plt.savefig(model_basepath + '.jpg')

    def train(self, model, train_dataset, base_learning_rate, epochs, train_label):

        model_basepath = os.path.join(self.logdir, train_label, \
                                      self.modelname + '_' + datetime.datetime.now().isoformat() + '_')

        self.compile_model(model, base_learning_rate, epochs)
        print(50 * '*')
        print('\n')
        print(f'start {train_label} training')
        print('\n')
        print(50 * '*')
        history = model.fit(train_dataset,
                            epochs=epochs,
                            callbacks=[
                                tf.keras.callbacks.CSVLogger(model_basepath + '.csv', append=True),
                                tf.keras.callbacks.ModelCheckpoint(
                                    filepath=model_basepath + '{epoch:02d}.h5',

                                    save_weights_only=True, verbose=1, save_freq="epoch")
                            ],
                            # validation_data=validation_dataset,
                            )

        # self.create_plot(history, model_basepath)
