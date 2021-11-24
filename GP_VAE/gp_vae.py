import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import pandas as pd
sns.set()

tf.compat.v1.enable_eager_execution()

from GP_VAE.models import *
from ca_predictor.utils import create_normal_df, create_arrest_df, normalize_signals, create_windows


class MIMIC_GP_AVE():
    def __init__(self, data_dim=28, latent_dim=8, time_length=48, kernel_size=4, mc_samples=10,
                 importance_weights=1, beta=0.8, length_scale=0.5, kernel_scales=1, sigma=1.005, kernel='cauchy'):
        # # Load Data
        # data_path = '/scratch/gobi3/datasets/measurement_selection_data/final_cohort_7p_missingness.csv'
        # # data_path = '/scratch/ssd002/datasets/measurement_selection_data'
        # self.trainset, self.validset, self.testset, self.observation_list, self.normalization_specs = \
        #     mimic_observation_loader(data_path)
        # Load model
        decoder = GaussianDecoder
        encoder = JointEncoder
        self.model = GP_VAE(latent_dim=latent_dim, data_dim=data_dim, time_length=time_length, encoder_sizes=[128, 128, 64],
                       encoder=encoder, decoder_sizes=[64, 128, 128], decoder=decoder, kernel=kernel, sigma=sigma,
                       length_scale=length_scale, kernel_scales=kernel_scales, kernel_size=kernel_size,
                       beta=beta, M=mc_samples, K=importance_weights)

    def train(self, trainset, validset, lr=1e-4):
        checkpoint_directory = "./ckpt/GP_VAE"
        if not os.path.exists(checkpoint_directory):
            os.mkdir(checkpoint_directory)
        _ = tf.compat.v1.train.get_or_create_global_step()
        trainable_vars = self.model.get_trainable_vars()
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        summary_writer = tf.summary.create_file_writer("./logs/training_summary")
        with summary_writer.as_default():
            losses_train, losses_val = [], []
            for i, (x_seq, m_seq, x_len) in trainset.enumerate():
                x_seq = x_seq
                m_seq = m_seq
                with tf.GradientTape() as tape:
                    tape.watch(trainable_vars)
                    loss = self.model.compute_loss(x_seq, m_mask=m_seq, x_len=x_len)
                grads = tape.gradient(loss, trainable_vars)
                grads = [np.nan_to_num(grad) for grad in grads]

                optimizer.apply_gradients(zip(grads, trainable_vars),
                                          global_step=tf.compat.v1.train.get_or_create_global_step())

                # Print intermediate results
                if i % 20 == 0:
                    print("================================================")
                    print("Learning rate: {}".format(optimizer._lr))
                    loss, nll, kl = self.model.compute_loss(x_seq, m_mask=m_seq, x_len=x_len, return_parts=True)
                    losses_train.append(loss.numpy())
                    print("Train loss = {:.3f} | NLL = {:.3f} | KL = {:.3f}".format(loss, nll, kl))

                    # Validation loss
                    for i_v, (x_seq_v, m_seq_v, x_len_v) in validset.enumerate():
                        val_loss, val_nll, val_kl = self.model.compute_loss(x_seq_v, m_mask=m_seq_v, x_len=x_len_v,
                                                                       return_parts=True)
                        break
                    losses_val.append(val_loss.numpy())
                    print("Validation loss = {:.3f} | NLL = {:.3f} | KL = {:.3f}".format(val_loss, val_nll, val_kl))
                    self.model.save_weights('./ckpt/gp_vae_model')

            # Plot losses
            plt.plot(losses_train, label='Train loss')
            plt.plot(losses_val, label='Validation loss')
            plt.legend()
            plt.savefig('./outputs/plots/GP_VAE_loss.pdf')

    def impute(self, dataset):
        x_hat_mean, x_hat_std, x, lens = [], [], [], []
        self.model.load_weights('./ckpt/gp_vae_model')
        for i, (x_test, m_seq, x_len) in dataset.enumerate():
            x_test = x_test

            qz_x = self.model.encode(x_test)
            z = qz_x.sample()
            px_z = self.model.decode(z)

            x_hat_mean.extend(px_z.mean())
            x_hat_std.extend(px_z.stddev())
            x.extend(x_test)
            lens.extend(x_len)

        return x, x_hat_mean, x_hat_std, lens

    def plot_counterfactual(self, x, x_hat_mean, x_hat_std, x_len, observation_list, normalization_specs):
        x = x[:int(x_len)]
        n_feats = x.shape[1]
        x_hat_mean = x_hat_mean[:int(x_len)]
        x_hat_std = x_hat_std[:int(x_len)]
        if 'max' in normalization_specs.keys():
            x = x*normalization_specs['max'].numpy()
            x_hat_mean = x_hat_mean * normalization_specs['max'].numpy()
            x_hat_std = x_hat_std * normalization_specs['max'].numpy()
        elif 'mean' in normalization_specs.keys():
            x = (x*normalization_specs['std']+normalization_specs['mean']).numpy()
            x_hat_mean = (x_hat_mean * normalization_specs['std']+normalization_specs['mean']).numpy()
            x_hat_std = (x_hat_std * normalization_specs['std']).numpy()
        f, axs = plt.subplots(n_feats)
        t = np.arange(0, len(x))
        for feat in range(n_feats):
            axs[feat].plot(t, x[:, feat], '-x', label='observation (missingness)')
            axs[feat].plot(t, x_hat_mean[:, feat], '-x', label='counterfactual')
            axs[feat].fill_between(t, (x_hat_mean[:, feat] - x_hat_std[:, feat]),
                                   (x_hat_mean[:, feat] + x_hat_std[:, feat]), color='g', alpha=.1)
            axs[feat].set_title(observation_list[feat])
        plt.legend()
        f.set_figheight(20)
        f.set_figwidth(x_len//7)
        plt.tight_layout()
        plt.savefig('./plots/cf_plot.pdf')
        f.clear()

        x_imputed = np.where(x == 0, x_hat_mean, x)
        std_imputed = np.where(x == 0, x_hat_std, 0)

        f, axs = plt.subplots(n_feats)
        for feat in range(n_feats):
            axs[feat].plot(t, x_imputed[:, feat], '-x', label='counterfactual')
            axs[feat].fill_between(t, (x_imputed[:, feat] - std_imputed[:, feat]),
                                   (x_imputed[:, feat] + std_imputed[:, feat]), color='g', alpha=.1)
            axs[feat].set_title(observation_list[feat])
        plt.legend()
        f.set_figheight(20)
        f.set_figwidth(x_len//7)
        plt.tight_layout()
        plt.savefig('./plots/imputed_signals.pdf')
        f.clear()


def main(args):
    # Load Data
    signal_list = ['Pulse', 'HR', 'SpO2', 'etCO2', 'NBPm', 'NBPd', 'NBPs', 'RR']
    
    PRE_ARR_WINDOW = 120
    data_dir = '/datasets/sickkids/ca_data/processed_data'
    arrest_train_df, arrest_valid_df, ca_normal_train_df, ca_normal_valid_df = create_arrest_df(data_dir,
                                                                                                signal_list,
                                                                                                PRE_ARR_WINDOW)
    normal_train_df, normal_valid_df = create_normal_df(data_dir, signal_list, verbose=True)
    normal_train_df = pd.concat([ca_normal_train_df, normal_train_df])
    normal_valid_df = pd.concat([ca_normal_valid_df, normal_valid_df])

    window_size = 360
    signal_type = 'all'
    arrest_train_signals = create_windows(arrest_train_df, window_size, all=True)
    arrest_valid_signals = create_windows(arrest_valid_df, window_size, all=True)
    normal_train_signals = create_windows(normal_train_df, window_size, all=True)
    normal_valid_signals = create_windows(normal_valid_df, window_size, all=True)


    shuffle_inds = list(range(len(normal_train_signals)))
    random.shuffle(shuffle_inds)
    normal_train_signals = normal_train_signals[shuffle_inds]
    normal_train_signals = normal_train_signals[:min(len(normal_train_signals), 20*len(arrest_train_signals))]
    shuffle_inds = list(range(len(normal_valid_signals)))
    random.shuffle(shuffle_inds)
    normal_valid_signals = normal_valid_signals[shuffle_inds]
    normal_valid_signals = normal_valid_signals[:min(len(normal_valid_signals), 20*len(arrest_valid_signals))]

    train_data = np.concatenate((arrest_train_signals, normal_train_signals), axis=0)
    valid_data = np.concatenate((arrest_valid_signals, normal_valid_signals), axis=0)
    train_data, valid_data, normalization_specs = normalize_signals(train_data, valid_data, 'mean_normalized',
                                                                            data_dir, signal_type)

    trainset = tf.data.Dataset.from_tensor_slices(
        (train_data[:,0, :, :].transpose(0,2,1), train_data[:,1, :, :].transpose(0,2,1), np.ones(len(train_data))*window_size)) \
        .batch(100).repeat(10)
    validset = tf.data.Dataset.from_tensor_slices(
        (valid_data[:,0, :, :].transpose(0,2,1), valid_data[:,1, :, :].transpose(0,2,1), np.ones(len(valid_data))*window_size)) \
        .batch(100)

    beta = 0.1
    gp_vae = MIMIC_GP_AVE(latent_dim=16, data_dim=10, time_length=360, length_scale=0.01, window_size=12, beta=beta)

    if args.train:
        gp_vae.train(trainset=trainset, validset=validset)
    x, x_hat_mean, x_hat_std, x_lens = gp_vae.impute(validset)
    rnd_int = np.random.randint(len(x))
    gp_vae.plot_counterfactual(x[rnd_int], x_hat_mean[rnd_int], x_hat_std[rnd_int], x_lens[rnd_int], signal_list, normalization_specs)


if __name__ == "__main__":
    random.seed(2021)
    parser = argparse.ArgumentParser(description='Run change point for Evidation data')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    main(args)