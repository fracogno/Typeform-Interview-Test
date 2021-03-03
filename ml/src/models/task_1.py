import tensorflow as tf


def get_MLP():
    return tf.keras.models.Sequential([tf.keras.layers.Dense(32),
                                       tf.keras.layers.BatchNormalization(),
                                       tf.keras.layers.LeakyReLU(),
                                       tf.keras.layers.Dropout(0.3),
                                       tf.keras.layers.Dense(16),
                                       tf.keras.layers.BatchNormalization(),
                                       tf.keras.layers.LeakyReLU(),
                                       tf.keras.layers.Dropout(0.3),
                                       tf.keras.layers.Dense(8),
                                       tf.keras.layers.BatchNormalization(),
                                       tf.keras.layers.LeakyReLU(),
                                       tf.keras.layers.Dropout(0.3),
                                       tf.keras.layers.Dense(1)
                                       ])
