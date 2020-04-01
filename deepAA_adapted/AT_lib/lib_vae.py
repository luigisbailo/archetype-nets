"""
Contains the Network Architectures.
"""
import tensorflow as tf
import numpy as np

tfd = tf.contrib.distributions


def share_variables(func):
    """
    Wrapper for tf.make_template as decorator.
    :param func:
    :return:
    """
    return tf.make_template(func.__name__, func, create_scope_now_=True)


def build_prior(dim_latentspace):
    """
    Creates N(0,1) Multivariate Normal prior.
    :param dim_latentspace:
    :return: mvn_diag
    """
    mu = tf.zeros(dim_latentspace)
    rho = tf.ones(dim_latentspace)
    mvn_diag = tfd.MultivariateNormalDiag(mu, rho)
    return mvn_diag

def dirichlet_prior(dim_latentspace, alpha=1.0):
    """

    :param dim_latentspace:
    :param alpha:
    :return:
    """
    nATs = dim_latentspace + 1
    alpha = [alpha] * nATs
    dist = tfd.Dirichlet(alpha)
    return dist

def build_encoder_basic(dim_latentspace, z_fixed, version='original'):
    """
    Basic DAA Encoder Architecture. Not actually used but more for showcasing the implementation.
    :param dim_latentspace:
    :param z_fixed:
    :return:z_predicted, mu_t, sigma_t, t
    """
    
    @share_variables
    def encoder(data):
        if version=='luigi':
            print("luigis version to be added")
            
        if version in ['original','milena']:
            nAT = dim_latentspace + 1
    
            #x = tf.layers.flatten(data)
            net = tf.layers.dense(data, 200, tf.nn.relu)
            net = tf.layers.dense(net, 100)
            mean_branch, var_branch = net[:, :50], net[:, 50:]
    
            # Weight Matrices
            weights_A = tf.layers.dense(mean_branch, nAT, tf.nn.softmax)
            weights_B_t = tf.layers.dense(mean_branch, nAT)
            weights_B = tf.nn.softmax(tf.transpose(weights_B_t), 1)
    
            # latent space parametrization
            mu_t = tf.matmul(weights_A, z_fixed)
            sigma_t = tf.layers.dense(var_branch, dim_latentspace, tf.nn.softplus)
            t = tfd.MultivariateNormalDiag(mu_t, sigma_t)
            
            # predicted archetypes
            z_predicted = tf.matmul(weights_B, mu_t)
    
        return {"z_predicted": z_predicted, "mu": mu_t, "sigma": sigma_t, "p": t}
    
    return encoder




def build_decoder(n_feats, n_targets, trainable_var=True, version='original'):
    """
     Builds Decoder for jaffe data
    :param n_feats:
    :param num_labels:
    :param trainable_var: Make the variance of the decoder trainable.
    :return:
    """
    
    @share_variables
    def decoder(latent_code):
        ## comment by MO: 
        # in original JAFFE version, var of feat. space (of pixel darkness) was 
        # set to be trainable but equal for all pixels. 
        # in our case, the features are fundamentally different to each other 
        # and if a variance is learned by the network, it should be one for each
        # feature -> thats why above, in var = ... [.]*(n_targets+n_feats) was added
        activation = tf.nn.relu
        
        if version=='luigi':
            print("luigis version to be added")
        if version=='milena':
            def decod_func(units):
                var = tf.Variable(initial_value=[1.0]*units, trainable=trainable_var)
    
                x = tf.layers.dense(latent_code, units=200, activation=activation)
                for i in np.arange(3):
                    x = tf.layers.dense(x, units=units, activation=activation)
                x_hat = tf.layers.dense(x, units=units, activation=tf.nn.sigmoid)
                shape = x_hat.shape
                print(shape,'SHAPE')
                x_m = tf.convert_to_tensor(x_hat) #tf.math.scalar_mul(1.0,x_hat)
                x_mean = x_m #tf.matmul(x_hat,x_hat)
                x_hat = tfd.Normal(x_hat, var)
                x_hat = tfd.Independent(x_hat, 2)
                #print('x_hat params:',x_hat.mean, x_hat.variance)
                return x_hat, x_mean, x_m
            
            x_hat, x_mean, x_m = decod_func(units=n_feats)
            y_hat, y_mean, y_m = decod_func(units=n_targets)
        
        if version=='original':
            var = tf.Variable(initial_value=1.0, trainable=trainable_var)

    
            activation = tf.nn.relu
            units = 49
            x = tf.layers.dense(latent_code, units=units, activation=activation)
            x = tf.layers.dense(x, units=units, activation=activation)
            
            #x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=np.prod(n_feats), activation=activation)
            x = tf.layers.dense(x, units=np.prod(n_feats), activation=tf.nn.sigmoid)

            x_mean = tf.convert_to_tensor(x)
            x_hat = tfd.Normal(x, var)
            x_hat = tfd.Independent(x_hat, 2)
    
            side_info = tf.layers.dense(latent_code, 200, tf.nn.relu)
            side_info = tf.layers.dense(side_info, n_targets, tf.nn.sigmoid) * 5
            y_mean = tf.convert_to_tensor(side_info)
            
            
        
        return {"x_hat": x_hat, "side_info": side_info, "x_mean": x_mean, "y_mean": y_mean}

    return decoder

