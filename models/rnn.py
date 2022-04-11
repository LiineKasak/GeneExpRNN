import datetime
from abc import ABC
import os
from typing import Any

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

from models.base_model import BaseModel


class HistoneRNN(BaseModel, ABC):

    def __init__(self, n_histones: int = 7, n_bins: int = 50, loss: str = 'huber_loss', verbose: int = 1,
                 n_epochs: int = 100, batch_size: int = 128):
        self.n_histones = n_histones
        self.n_bins = n_bins
        self.loss = loss
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.model = None
        self.filename = None

        self.init_model()

    def init_model(self) -> None:
        self.filename = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape((1, self.n_histones, self.n_bins), input_shape=(self.n_histones, self.n_bins)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(300, kernel_size=(1, 7), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Reshape((1, 300)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Reshape((1, 128)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation='relu')
        ])
        self.model.compile(
            loss=self.loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
            metrics=[self.spearman_rankcor],
        )

    def callbacks(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/fit/' + self.filename, histogram_freq=1)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_spearman_rankcor', patience=30, mode='max')

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'checkpoint/{self.filename}.h5',
            monitor='spearman_rankcor',
            mode='max',
            save_weights_only=False,
            save_best_only=True)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)

        return [tensorboard_callback, model_checkpoint_callback, early_stop, reduce_lr]

    @staticmethod
    def spearman_rankcor(y_true, y_pred):
        return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                                           tf.cast(y_true, tf.float32)], Tout=tf.float32))

    def fit(self, x, y, x_val=None, y_val=None) -> None:
        if y_val is not None:
            self.model.fit(x, y, epochs=self.n_epochs, batch_size=self.batch_size, validation_data=(x_val, y_val),
                           callbacks=self.callbacks(), verbose=self.verbose)
        else:
            self.model.fit(x, y, epochs=self.n_epochs, batch_size=self.batch_size, callbacks=self.callbacks(),
                           verbose=self.verbose)

        self.model = tf.keras.models.load_model(f'checkpoint/{self.filename}.h5',
                                                custom_objects={"spearman_rankcor": self.spearman_rankcor})

    def predict(self, x) -> np.ndarray:
        return self.model.predict(x)


if __name__ == '__main__':
    model = HistoneRNN(verbose=1)
    model.fit_predict()
    # model.cross_validate()
