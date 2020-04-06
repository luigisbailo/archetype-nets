#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:31:47 2020

@author: oehlers
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.distributions as tfd
import pandas as pd
#Three archetypes are defined on a two dimensional space.



def build_network(intermediate_dim = 4, batch_size = 1024, latent_dim = 2, epochs = 100):
    
    def execute(data, at_loss_factor=8.0, target_loss_factor=8.0,recon_loss_factor=4.0,kl_loss_factor=4.0):
        def get_zfixed ( dim_latent_space ):
            
            z_fixed_t = np.zeros([dim_latent_space, dim_latent_space + 1])
        
            for k in range(0, dim_latent_space):
                s = 0.0
                for i in range(0, k):
                    s = s + z_fixed_t[i, k] ** 2
          
                z_fixed_t[k, k] = np.sqrt(1.0 - s)
        
                for j in range(k + 1, dim_latent_space + 1):
                    s = 0.0
                    for i in range(0, k):
                        s = s + z_fixed_t[i, k] * z_fixed_t[i, j]
        
                    z_fixed_t[k, j] = (-1.0 / float(dim_latent_space) - s) / z_fixed_t[k, k]
                    z_fixed = np.transpose(z_fixed_t)
                            
            return z_fixed
        
        
        #We use variational autoencoders to map the data set into a latent space. The neural network is constructed to force data in latent space to be defined within an arbitrary convex hull. We use a triangular convex hull as shown below. 
        
        zfixed = get_zfixed (2)
        
        #We construct a variational autoencoder that generates a mean $\mu$, and a standard deviation $\sigma$ for each data point. The point is then mapped into the latent space with a stochastic extraction from a Gaussian $\mathcal{N}(\mu,\,\sigma^{2})$, where $\mu$'s are by construction within a hull $z_{fixed}$. 
        x_train = data['train_feat']
        y_train = data['train_targets']
        x_test = data['test_feat']
        y_test = data['test_targets']
        
        
        original_dim = x_train.shape [1]
        try:
            sideinfo_dim = y_train.shape [1]
        except:
            sideinfo_dim = 1
        
        x_train = np.array(np.reshape(x_train, [-1, original_dim]), dtype='float32')
        x_test = np.array(np.reshape(x_test, [-1, original_dim]), dtype='float32')
        y_train = np.array(np.reshape(y_train, [-1, sideinfo_dim]), dtype='float32')
        y_test = np.array(np.reshape(y_test, [-1, sideinfo_dim]), dtype='float32')
        
        
        # network parameters
        simplex_vrtxs = latent_dim + 1
        
        
        # encoder
        input_x = tfk.Input(shape=(original_dim,), name='encoder_input_x', dtype='float32')
        
        x = tfkl.Dense(intermediate_dim, activation='relu')(input_x)
        x = tfkl.Dense(intermediate_dim, activation='relu')(x)
        A = tfkl.Dense (simplex_vrtxs, activation='linear')(x)
        A = tfkl.Dense (simplex_vrtxs, activation=tf.nn.softmax)(A)
        B_t = tfkl.Dense (simplex_vrtxs, activation='linear')(x)
        B = tf.nn.softmax(tf.transpose(B_t), axis=1)
        
        z_fixed = get_zfixed (latent_dim)
        z_fixed = tf.constant (z_fixed, dtype='float32')
        mu = tf.matmul(A, z_fixed)
        z_pred = tf.matmul(B,mu)
        sigma = tfkl.Dense(latent_dim)(x)
        t = tfd.Normal(mu,sigma)
        
        input_y = tfk.Input(shape=(sideinfo_dim,), name='encoder_input_y', dtype='float32')
        y = tf.identity(input_y)
        
        encoder = tfk.Model([input_x,input_y], [t.sample(),mu,sigma, tf.transpose(B) ,y], name='encoder')
        encoder.summary()
        
        
        
        # decoder
        latent_inputs = tfk.Input(shape=(latent_dim,), name='z_sampling')
        input_y_lat = tfk.Input(shape=(sideinfo_dim,), name='encoder_input_y_lat')
        
        x = tfkl.Dense(intermediate_dim, activation='relu')(latent_inputs)
        x = tfkl.Dense(original_dim, activation='linear')(x)
        x_hat = tfkl.Dense(original_dim, activation='linear')(x)
        
        y = tfkl.Dense(intermediate_dim, activation='relu')(latent_inputs)
        y = tfkl.Dense(intermediate_dim, activation='relu')(y)
        y_hat = tfkl.Dense(sideinfo_dim, activation='linear')(y) 
        
        decoder = tfk.Model([latent_inputs,input_y_lat], [x_hat,y_hat], name='decoder')
        decoder.summary()
        
        
        
        # VAE
        encoded = encoder([input_x,input_y])
        outputs = decoder( [encoded[0],encoded[-1]])
        vae = tfk.Model([input_x,input_y], outputs, name='vae')
        
        reconstruction_loss = tfk.losses.mse (input_x, outputs[0])
        class_loss = tfk.losses.mse ( input_y, outputs[1])
        archetype_loss = tf.reduce_sum( tfk.losses.mse(z_fixed, z_pred))
        
        kl_loss = 1 + sigma - tf.square(mu) - tf.exp(sigma)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        
        vae_loss = tf.reduce_mean(recon_loss_factor*reconstruction_loss 
                                  + target_loss_factor*class_loss 
                                  + kl_loss_factor*kl_loss 
                                  + at_loss_factor*archetype_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        vae.summary()
        
        
        vae.fit([x_train,y_train],  
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([x_test,y_test],None))
        
        
        
        # archetypes
        archetypes,_ = decoder ([z_pred, tf.zeros([3,3])])
        get_archtypes = tfk.Model (input_x, [archetypes,z_pred] , name='get_zpred')
        
        
        t,mu,sigma, B_t, y = encoder.predict([x_train,np.zeros(np.shape(y_train))])
        archetypes_pred, z_pred = get_archtypes(x_train)
        x_test_pred, y_test_pred = vae.predict([x_test,np.zeros(np.shape(y_test))])
        
        x_test1, x_test2 = x_test.T
        mu1, mu2 = mu.T
        x_test_pred1, x_test_pred2 = x_test_pred.T
        
        result_key = ('luigi', at_loss_factor,target_loss_factor,recon_loss_factor,kl_loss_factor)
        result_df = pd.DataFrame({'dim1':[x_test1, mu1, x_test_pred1],
                           'dim2':[x_test2, mu2, x_test_pred2],
                           'target_color':[y_test,[1]*len(mu1),y_test_pred]},
                        index=['real space', 'latent space', 'reconstructed real space'])
        result_dict = {result_key : result_df }
        
        return result_dict
    
    return execute
