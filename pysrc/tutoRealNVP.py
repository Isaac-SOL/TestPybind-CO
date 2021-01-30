# -*- coding: utf-8 -*-
# Test Real NVP

# Valeurs par défaut
#sample_dims = 2
#hidden_dims = 256
#coupling_layers = 8
#learning_rate = 1e-4
#allow_gpu = True

# Ces lignes font en sorte de n'utiliser que le CPU, même si le GPU est disponible.
if not allow_gpu:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.datasets import make_moons
import numpy as np
import tensorflow_probability as tfp

# Loading example data
# Note : ne marche que si sample_dims = 2

data = make_moons(3000, noise=0.05)[0].astype("float32")
norm = layers.experimental.preprocessing.Normalization()
norm.adapt(data)
normalized_data = norm(data)

# Defining coupling layers
# Creating a custom layer with keras API.

def Coupling(input_shape, output_dim, reg=0.01):
    input = keras.layers.Input(shape=input_shape)

    t_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    t_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_1)
    t_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_2)
    t_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_3)
    t_layer_5 = keras.layers.Dense(
        input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_4)

    s_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    s_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_1)
    s_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_2)
    s_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_3)
    s_layer_5 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_4)

    return keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])

# Real NVP class

class RealNVP(keras.Model):
    def __init__(self, num_coupling_layers):
        super(RealNVP, self).__init__()
        
        self.num_coupling_layers = num_coupling_layers

        # Distribution of the latent space.
        self.distribution = tfp.distributions.Uniform(
            low=[0.0,0.0], high=[1.0,1.0]
        )
        self.masks = np.array(
            [[0, 1], [1, 0]] * (num_coupling_layers // 2), dtype="float32"
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(sample_dims, hidden_dims) for i in range(num_coupling_layers)]

    @property
    def metrics(self):
        # List of the model's metrics.
        # We make sure the loss tracker is listed as part of `model.metrics`
        # so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        # at the start of each epoch and at the start of an `evaluate()` call.
        return [self.loss_tracker]

    def call(self, x, training=True):
        
        # logit
        x = -tf.math.log((1 / x) - 1)
        
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s,t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                    * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                    + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s,[1])
    
        # inverse logit
        x = 1 / (1 + tf.math.exp(-x))

        return x, log_det_inv

    def log_loss(self, x):
        y, logdet = self(x);
        log_prob = self.distribution.log_prob(y)
        log_prob = tf.math.reduce_sum(log_prob, axis=1);
        #log_prob = log_prob[:, 0] * log_prob[:, 1]
        #log_likelihood = self.distribution.log_prob(y) + logdet
        #return -tf.reduce_mean(log_likelihood)
        return -tf.reduce_sum(log_prob + logdet)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)
        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data);
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def forward(self, x):
        y, _ = self(x);
        return y

    def inverse(self, y):
        x, _ = self(y, training=False)
        return x

# Model Training

model = RealNVP(coupling_layers)
model.compile(optimizer=keras.optimizers.Adam(learning_rate))