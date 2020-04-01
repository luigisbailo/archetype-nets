#!/usr/bin/env python
# util
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
from itertools import compress
from pathlib import Path

# scipy stack + tf
import numpy as np
import pandas as pd
import tensorflow as tf

# custom libs
from AT_lib import lib_vae, lib_at

tfd = tf.contrib.distributions

# If error message "Could not connect to any X display." is issued, uncomment the following line:
#os.environ['QT_QPA_PLATFORM']='offscreen'

def main():
    def build_loss(version='original'):
        """
        Build all the required losses for the Deep Archetype Model.
        :return: archetype_loss, target_loss, likelihood, divergence, elbo
        """
        likelihood = tf.reduce_sum(x_hat.log_prob(data)) 
        if args.vamp_prior or args.dir_prior:
            q_sample = t_posterior.sample(50)
            divergence = tf.reduce_mean(encoded_z_data["p"].log_prob(q_sample) - prior.log_prob(q_sample))
        else:
            divergence = tf.reduce_mean(tfd.kl_divergence(t_posterior, prior))

        if not args.vae:
            archetype_loss = tf.losses.mean_squared_error(z_predicted, z_fixed)
        else:
            archetype_loss = tf.constant(0, dtype=tf.float32)
        # Sideinformation Reconstruction loss
        if version=='milena':
            target_loss = tf.reduce_sum(y_hat.log_prob(side_information)) 
        if version in ['original','luigi']:
            target_loss = tf.losses.mean_squared_error(side_information, y_hat) # * args.target_loss_factor
        
            
        elbo = tf.reduce_mean(
            args.recon_loss_factor * likelihood
            + args.target_loss_factor * target_loss # + instead of - as in original paper due to loss definition as in paper now (continuous, single target variable)
            - args.at_loss_factor * archetype_loss
            - args.kl_loss_factor * divergence
        )

        return archetype_loss, target_loss, likelihood, divergence, elbo
    ####################################################################################################################
    ############################################# Create Data #################################################################
    archs = np.array([[-1,-1],
                  [2,-2],
                  [-2,2]])
    arch_target = np.array([[-1],[2],[-2]])
    X,Y = archs.T
    plt.scatter(X,Y, color='red')
    def generate_data (archs, arch_target, n_points, noise=0.1):
        k = len(archs)
        X,Y = archs.T 
        rand = np.random.uniform (0,1,[k,n_points])
        rand = (rand/np.sum(rand,axis=0)).T
        joined = np.concatenate([archs, arch_target], axis=1)
        data = np.matmul(rand,joined)
        data = data + np.random.normal(0,noise,size=data.shape)
        feat, target = data[:,:-1], data[:,-1]
        return feat, target

    x_train_feat, x_train_targets = generate_data (archs,arch_target,1000,noise=0.01)
    x_test_feat, x_test_targets = generate_data (archs,arch_target,100,noise=0.01)
    
    ####################################################################################################################
    # ########################################### Use Data #################################################################
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # # Japanese Face Expressions

    n_total_samples, n_feats = x_train_feat.shape[0], x_train_feat.shape[1]
    n_batches = int(n_total_samples / args.batch_size)

    def get_batch(batch_size, x_train_feat, x_train_targets, shuffle=True):
        x_train = pd.DataFrame(np.concatenate([x_train_feat,x_train_targets.reshape([n_total_samples,args.n_targets])],axis=1))
        x_train_sampled = x_train.sample(n=batch_size) if shuffle else x_train
        x_train_feat_sampled, x_train_targets_sampled = x_train_sampled.iloc[:,:-1],x_train_sampled.iloc[:,-1:] #.reshape((x_train_sampled[0],args.n_targets))
        return x_train_feat_sampled, x_train_targets_sampled
    
    def get_next_batch(batch_size):
        """
        Helper function for getting mini batches.
        :param batch_size:
        :return: mb_x, mb_y
        """
        mb_x, mb_y = get_batch(batch_size, x_train_feat, x_train_targets)
        return mb_x, mb_y

    all_feats, all_targets = get_batch(n_total_samples,x_train_feat, x_train_targets)
    
    some_feats = all_feats[:args.batch_size]
    some_targets = all_targets[:args.batch_size]
    all_imgs_ext = np.vstack((all_feats, all_feats[:args.batch_size]))
    #jaffe_meta_data = pd.read_csv(JAFFE_CSV_P, header=0, delimiter=" ")

    ####################################################################################################################
    # ########################################### Data Placeholders ####################################################
    data = tf.placeholder(tf.float32, [None, n_feats], 'data')
    side_information = tf.placeholder(tf.float32, [None, args.n_targets], 'targets')
    #latent_code = tf.placeholder(tf.float32, [None, args.dim_latentspace], 'latent_code')
    
    ####################################################################################################################
    # ########################################### Model Setup ##########################################################
    z_fixed_ = lib_at.create_z_fix(args.dim_latentspace)
    z_fixed = tf.cast(z_fixed_, tf.float32)
    
    encoder_net = lib_vae.build_encoder_basic(dim_latentspace=args.dim_latentspace, z_fixed=z_fixed, version=args.version)
    decoder = lib_vae.build_decoder(n_feats=n_feats, n_targets=args.n_targets, trainable_var=args.trainable_var, version=args.version)

    prior = lib_vae.build_prior(args.dim_latentspace)
    
    encoded_z_data = encoder_net(data)
    try:
        z_predicted, mu_t, sigma_t, t_posterior = [encoded_z_data[key] for key in ["z_predicted", "mu", "sigma", "p"]]
    except KeyError:
        assert args.vae
        mu_t, sigma_t, t_posterior = [encoded_z_data[key] for key in ["mu", "sigma", "p"]]
    decoded_post_sample = decoder(t_posterior.sample())
    x_hat, y_hat, x_mean, y_mean = decoded_post_sample["x_hat"], decoded_post_sample["side_info"],decoded_post_sample["x_mean"], decoded_post_sample["y_mean"]
    print(x_mean,y_mean,"HERE")
    
    # Build the loss
    archetype_loss, target_loss, likelihood, kl_divergence, elbo = build_loss(args.version)

    # Build the optimizer
    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(-elbo)
 
    # Specify what is to be logged.
    tf.summary.scalar(name='elbo', tensor=elbo)
    tf.summary.scalar(name='archetype_loss', tensor=archetype_loss)
    tf.summary.scalar(name='target_loss', tensor=target_loss)
    tf.summary.scalar(name='likelihood', tensor=likelihood)
    tf.summary.scalar(name='kl_divergence', tensor=kl_divergence)
    #tf.summary.tensor(name='x_sam', tensor=x_sam)
    #tf.summary.tensor(name='y_sam', tensor=y_sam)
    
    hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in vars(args).items()]
    tf.summary.text('hyperparameters', tf.stack(hyperparameters))

    summary_op = tf.summary.merge_all()

    def create_latent_df():
        """
        Create pandas DF with the latent mean coordinates + targets of the data.
        :return: Dataframe  pd.DataFrame(array_all, columns=['targets','ldim0',..., 'ldimN'])
        
        def get_var(var,feed_what_with_what_dict):
            var_stacked = None
            for i in range(all_imgs_ext.shape[0] // args.batch_size):
                min_idx = i * args.batch_size
                max_idx = (i + 1) * args.batch_size
                tmp_mu= sess.run(var, feed_dict={data: all_feats[min_idx:max_idx],
                                                   side_information: all_targets[min_idx:max_idx]})
                var_stacked = np.vstack((test_pos_mean, var)) if test_pos_mean is not None else tmp_mu
            
            var_stacked = var_stacked[:all_feats.shape[0]]
            array_all = np.hstack((test_pos_mean, all_targets))
            cols_dims = [f'ldim{i}' for i in range(args.dim_latentspace)]
            df = pd.DataFrame(array_all, columns=cols_dims + ['target'])
            return df
        """
        test_pos_mean = None
        for i in range(all_imgs_ext.shape[0] // args.batch_size):
            min_idx = i * args.batch_size
            max_idx = (i + 1) * args.batch_size
            tmp_mu= sess.run(mu_t, feed_dict={data: all_feats[min_idx:max_idx],
                                               side_information: all_targets[min_idx:max_idx]})
            test_pos_mean = np.vstack((test_pos_mean, tmp_mu)) if test_pos_mean is not None else tmp_mu
            
        test_pos_mean = test_pos_mean[:all_feats.shape[0]]
        array_all = np.hstack((test_pos_mean, all_targets))
        cols_dims = [f'ldim{i}' for i in range(args.dim_latentspace)]
        df = pd.DataFrame(array_all, columns=cols_dims + ['target'])
        return df
    
    def extract_xy_hat(df):
        #tmp_x_mean = sess.run(x_mean, feed_dict={latent_code: df})
        #decoded_post_sample = decoder(df)
        tmp_xmean = sess.run(x_mean, feed_dict={data: all_feats,
                                               side_information: all_targets})
        tmp_ymean = sess.run(y_mean, feed_dict={data: all_feats,
                                               side_information: all_targets})
        return tmp_xmean, tmp_ymean

    ####################################################################################################################
    ############################################# Training Loop ########################################################
    saver = tf.train.Saver()
    step = 0
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(logdir=TENSORBOARD_DIR, graph=sess.graph)
    for epoch in range(args.n_epochs):
        print('epoch no {} of {}'.format(epoch,args.n_epochs))
        for b in range(n_batches):
            mb_x, mb_y = get_next_batch(args.batch_size)

            feed_train = {data: mb_x, side_information: mb_y}
            sess.run(optimizer, feed_dict=feed_train)
            step += 1

        
        # evaluate metrics on some feats; NOTE that this is no real test set
        tensors_test = [summary_op, elbo,
                        likelihood,
                        kl_divergence, archetype_loss, target_loss]
        feed_test = {data: some_feats, side_information: some_targets}
        summary, test_total_loss, test_likelihood, test_kl, test_atl, test_targetl = sess.run(tensors_test,
                                                                                                    feed_test)
        if epoch % args.test_frequency_epochs == 0:
            writer.add_summary(summary, global_step=step)
            print(str(args.runNB) + '\nEpoch ' + str(epoch) + ':\n', 'Total Loss:', test_total_loss,
                  '\n Feature-Likelihood:', test_likelihood, #np.mean(test_likelihood),
                  '\n Divergence:', test_kl, # np.mean(test_kl),
                  '\n Archetype Loss:', test_atl,
                  '\n Target-Likelihood:', test_targetl, #np.mean(test_targetl), # / args.target_loss_factor,
                  )

        if epoch % args.save_each == 0 and epoch > 0:
            saver.save(sess, save_path=SAVED_MODELS_DIR / "save", global_step=epoch)

    saver.save(sess, save_path=SAVED_MODELS_DIR / "save", global_step=args.n_epochs)
    print("Model Trained!")
    print("Tensorboard Path: {}".format(TENSORBOARD_DIR))
    print("Saved Model Path: {}".format(SAVED_MODELS_DIR))

    # create folder for inference results in the folder of the most recently trained model
    if not FINAL_RESULTS_DIR.exists():
        os.mkdir(FINAL_RESULTS_DIR)

    df = create_latent_df()
    df.to_csv(FINAL_RESULTS_DIR / "latent_codes.csv", index=False)
    
    xhat,yhat = extract_xy_hat(df.iloc[:,:-args.n_targets])
    
    print('training datapoints in real space:')
    plt.scatter(x_train_feat[:,0],x_train_feat[:,1],c=x_train_targets)
    plt.show()
    print('training datapoints in latent space:')
    
    plt.scatter(df['ldim0'],df['ldim1'], c=df['target'])
    plt.show()
    
    print('reconstructed training datapoints in real space:')
    plt.scatter(xhat[:,0],xhat[:,1], c=yhat.reshape(n_total_samples))
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Logging Settings
    parser.add_argument('--runNB', type=str, default="1")
    parser.add_argument('--results-path', type=str, default='./Results/JAFFE')
    parser.add_argument('--test-frequency-epochs', type=int, default=100)
    parser.add_argument('--save_each', type=int, default=10000)

    # NN settings
    parser.add_argument('--version',type=str,default='original')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--n-epochs', type=int, default=100) # 5001 in original code
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--dim-latentspace', type=int, default=2,
                        help="Number of Archetypes = Latent Space Dimension + 1")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_targets', type=int, default=1)
    parser.add_argument('--discrete_target', type=int, default=0)
    parser.add_argument('--trainable-var', dest='trainable_var', action='store_true', default=False,
                        help="Learn variance of decoder. If false, set to constant '1.0'.")

    # DAA loss: weights
    parser.add_argument('--at-loss-factor', type=float, default=8.0) # 80.0 # 4
    parser.add_argument('--target-loss-factor', type=float, default=8.0) # 80.0 # 16
    parser.add_argument('--recon-loss-factor', type=float, default=4.0) # 1.0 # 16
    parser.add_argument('--kl-loss-factor', type=float, default=4.0) # 40.0 # 4

    # loading already existing model
    parser.add_argument('--test-model', dest='test_model', action='store_false', default=False)
    parser.add_argument('--model-substr', type=str, default=None)

    # Different settings for the prior
    parser.add_argument('--vamp-prior', dest='vamp_prior', action='store_true', default=False,
                        help="Use the vamp prior instead of a standard normal.")
    parser.add_argument('--dir-prior', dest='dir_prior', action='store_true', default=False,
                        help="Use the dirichlet + Gauss noise prior instead of a standard normal.")
    parser.add_argument('--vamp-num-inducing', dest='vamp_num_inducing', type=int, default=50,
                        help="Number of inducing points for the Vamp Prior.")
    parser.add_argument('--vae', dest='vae', action='store_true', default=False,
                        help="Train standard vae instead of AT.")

    args = parser.parse_args()
    print(args)

    assert not (args.dir_prior and args.vamp_prior), "The different priors are mutually exclusive."
    assert 0 < args.n_targets <= 5, "Choose up to 5 targets."

    nAT = args.dim_latentspace + 1

    # GPU targets
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # all the path/directory stuff
    CUR_DIR = Path(__file__).resolve().parent
    RESULTS_DIR = CUR_DIR / 'Results/JAFFE'
    if not args.test_model:
        # create new model directory
        MODEL_DIR = RESULTS_DIR / "{time}_{run_name}_{dim_lspace}_{mb_size}_{n_epochs}".format(
            time=datetime.now().replace(second=0, microsecond=0),
            run_name=args.runNB, dim_lspace=args.dim_latentspace, mb_size=args.batch_size, n_epochs=args.n_epochs)
    else:
        # get latest trained model matching to args.model_substr
        all_results = os.listdir(RESULTS_DIR)
        if args.model_substr is not None:
            idx = [args.model_substr in res for res in all_results]
            all_results = list(compress(all_results, idx))
        all_results.sort()
        MODEL_DIR = RESULTS_DIR / all_results[-1]

    FINAL_RESULTS_DIR = MODEL_DIR / 'final_results/'
    TENSORBOARD_DIR = MODEL_DIR / 'Tensorboard'
    IMGS_DIR = MODEL_DIR / 'imgs'
    SAVED_MODELS_DIR = MODEL_DIR / 'Saved_models/'
    VIDEO_IMGS_DIR = FINAL_RESULTS_DIR / "video_imgs"

    JAFFE_CSV_P = CUR_DIR / 'jaffe/targets.csv'
    JAFFE_IMGS_DIR = CUR_DIR / 'jaffe/feats'

    if not args.test_model:
        for path in [TENSORBOARD_DIR, SAVED_MODELS_DIR, IMGS_DIR]:
            os.makedirs(path, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    nAT = args.dim_latentspace + 1
    main()
