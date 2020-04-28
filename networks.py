import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.distributions as tfd
import numpy as np


class VAE:

    def __init__(self,
                 original_dim,
                 intermediate_dim,
                 latent_dim,
                 sideinfo_dim):

        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.sideinfo_dim = sideinfo_dim
        self.latent_dim = latent_dim
        self.simplex_vrtxs = latent_dim + 1
        self.input_x = tfk.Input(shape=(self.original_dim,), name='encoder_input_x', dtype='float32')
        self.input_y = tfk.Input(shape=(self.sideinfo_dim,), name='encoder_input_y', dtype='float32')

        self.encoder, self.z_pred, self.sigma, self.mu = self.build_encoder()
        self.decoder = self.build_decoder()
        self.network, self.outputs = self.build_vae()

    def build_encoder(self):

        x = tfkl.Dense(self.intermediate_dim, activation='relu')(self.input_x)
        x = tfkl.Dense(self.intermediate_dim, activation='relu')(x)
        A = tfkl.Dense(self.simplex_vrtxs, activation='linear')(x)
        A = tfkl.Dense(self.simplex_vrtxs, activation=tf.nn.softmax)(A)
        B_t = tfkl.Dense(self.simplex_vrtxs, activation='linear')(x)
        B = tf.nn.softmax(tf.transpose(B_t), axis=1)

        z_fixed = self.get_zfixed()
        z_fixed = tf.constant(z_fixed, dtype='float32')
        mu = tf.matmul(A, z_fixed)
        z_pred = tf.matmul(B, mu)
        sigma = tfkl.Dense(self.latent_dim)(x)
        t = tfd.Normal(mu, sigma)

        y = tf.identity(self.input_y)

        encoder = tfk.Model([self.input_x, self.input_y], [t.sample(), mu, sigma, tf.transpose(B), y], name='encoder')
        encoder.summary()

        return encoder, z_pred, sigma, mu

    def build_decoder(self):

        latent_inputs = tfk.Input(shape=(self.latent_dim,), name='z_sampling')
        input_y_lat = tfk.Input(shape=(self.sideinfo_dim,), name='encoder_input_y_lat')

        x = tfkl.Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        x = tfkl.Dense(self.original_dim, activation='linear')(x)
        x_hat = tfkl.Dense(self.original_dim, activation='linear')(x)

        y = tfkl.Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        y = tfkl.Dense(self.intermediate_dim, activation='relu')(y)
        y_hat = tfkl.Dense(self.sideinfo_dim, activation='linear')(y)

        decoder = tfk.Model([latent_inputs, input_y_lat], [x_hat, y_hat], name='decoder')

        return decoder

    def build_vae(self):

        encoded = self.encoder([self.input_x, self.input_y])
        outputs = self.decoder([encoded[0], encoded[-1]])
        vae = tfk.Model([self.input_x, self.input_y], outputs, name='vae')

        return vae, outputs

    def add_loss(self):

        z_fixed = self.get_zfixed()

        reconstruction_loss = tfk.losses.mse(self.input_x, self.outputs[0])
        class_loss = tfk.losses.mse(self.input_y, self.outputs[1])
        archetype_loss = tf.reduce_sum(tfk.losses.mse(z_fixed, self.z_pred))
        kl_loss = 1 + self.sigma - tf.square(self.mu) - tf.exp(self.sigma)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        lambda_p = 1
        nu_p = 1

        vae_loss = tf.reduce_mean(nu_p * reconstruction_loss + lambda_p * class_loss + kl_loss + archetype_loss)
        self.network.add_loss(vae_loss)

    def get_zfixed(self):

        z_fixed_t = np.zeros([self.latent_dim, self.latent_dim + 1])

        for k in range(0, self.latent_dim):
            s = 0.0
            for i in range(0, k):
                s = s + z_fixed_t[i, k] ** 2

            z_fixed_t[k, k] = np.sqrt(1.0 - s)

            for j in range(k + 1, self.latent_dim + 1):
                s = 0.0
                for i in range(0, k):
                    s = s + z_fixed_t[i, k] * z_fixed_t[i, j]

                z_fixed_t[k, j] = (-1.0 / float(self.latent_dim) - s) / z_fixed_t[k, k]

        z_fixed = np.transpose(z_fixed_t)

        return z_fixed

