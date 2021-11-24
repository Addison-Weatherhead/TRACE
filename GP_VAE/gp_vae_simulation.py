import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import pandas as pd
sns.set()
import pickle

tf.compat.v1.enable_eager_execution()

from GP_VAE.models import *
from GP_VAE.utils import create_normal_df, create_arrest_df, normalize_signals, create_windows

def create_mask(x):
    '''
    x should be of shape (num_samples, num_features, seq_length)
    Creates a mask of each sample. Masks are generated in such a way where the 0's cluster together, since it uses a geometric distribution to determine lengts of masks, rather
    than a simple bernoulli for determining if any individual element should be masked.
    '''
    x = torch.Tensor(x)
    r = 0.15 # prob of setting a value to 0

    l_m = 3 # mean len of a segment of 0's
    # Recall for a geometric dist., mean=(1-p)/p, where p defines the geom dist.
    p_0 = 1/(1+l_m) # p_0 is the value that defines the geom dist of lengths of seq's of 0's

    l_u = ((1-r)/r)*l_m  # mean len of a segment of 1's
    p_1 = 1/(1+l_u) # p_1 is the value that defines the geom dist of lengths of seq's of 1's

    # Recall x is of shape (num_samples, num_features, seq_len])
    x = torch.Tensor(x)
    num_samples, num_features, seq_len = x.shape

    mask = torch.ones(x.shape)
    
    start_indices = torch.zeros((num_samples, num_features))


    # done_values are booleans indicating if we've finished masking for the ij'th sequence, where i and j
    # are in the range of num_samples and num_features respectively. 0=Done Masking, 1=Not Done
    done_values = torch.ones((num_samples, num_features))
    while True:
        items, counts = torch.unique(done_values, return_counts=True)
        if counts[0] == num_samples*num_features and 0  in items: # done_values is filled with 0's
            break
        
        for i in range(num_samples):
            for j in range(num_features):
                if done_values[i][j] == 1:
                    one_seg_length = np.random.geometric(p_1)
                    mask[i, j, int(start_indices[i][j]): min(seq_len, int(start_indices[i][j]+one_seg_length * done_values[i][j]))] = 1
                    start_indices[i][j] += one_seg_length * done_values[i][j]
                    if start_indices[i][j] > seq_len:
                        done_values[i][j] = 0
                
                if done_values[i][j] == 1:
                    zero_seg_length = np.random.geometric(p_0)
                    mask[i, j, int(start_indices[i][j]): min(seq_len, int(start_indices[i][j]+zero_seg_length * done_values[i][j]))] = 0
                    start_indices[i][j] += zero_seg_length * done_values[i][j]
                    if start_indices[i][j] > seq_len:
                        done_values[i][j] = 0
    
    return mask.numpy()

class MIMIC_GP_AVE():
    def __init__(self, data_dim=28, latent_dim=8, time_length=48, window_size=4, mc_samples=10,
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
                       length_scale=length_scale, kernel_scales=kernel_scales, window_size=window_size,
                       beta=beta, M=mc_samples, K=importance_weights)

    def train(self, trainset, validset, lr=1e-4):
        checkpoint_directory = "./ckpt/GP_VAE"
        if not os.path.exists(checkpoint_directory):
            os.mkdir(checkpoint_directory)
        _ = tf.compat.v1.train.get_or_create_global_step()
        trainable_vars = self.model.get_trainable_vars()
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        summary_writer = tf.summary.create_file_writer("./outputs/logs/training_summary")
        with summary_writer.as_default():
            losses_train, losses_val = [], []
            for i, (x_seq, m_seq, x_len) in trainset.enumerate():
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
                    self.model.save_weights('./ckpt/GP_VAE/')

            # Plot losses
            plt.plot(losses_train, label='Train loss')
            plt.plot(losses_val, label='Validation loss')
            plt.legend()
            plt.savefig('./outputs/plots/GP_VAE/loss.pdf')

    def impute(self, dataset):
        x, x_hat_mean, x_hat_std, lens = [], [], [], []
        self.model.load_weights('./ckpt/GP_VAE/')
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
        plt.savefig('./outputs/plots/GP_VAE/cf_plot.pdf')
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
        plt.savefig('./outputs/plots/GP_VAE/imputed_signals.pdf')
        f.clear()


def main(args):
    # Load Data
    
    signal_list = ['Pulse', 'HR', 'SpO2', 'etCO2', 'NBPm', 'NBPd', 'NBPs', 'RR', 'CVPm', 'awRR']
    col_list = ['Pulse', 'HR', 'SpO2', 'etCO2', 'NBPm', 'NBPd', 'NBPs', 'RR', 'CVPm', 'awRR',
                'Pulse_map', 'HR_map', 'SpO2_map', 'etCO2_map', 'NBPm_map', 'NBPd_map',
                'NBPs_map', 'RR_map', 'CVPm_map', 'awRR_map']
   
    path = ''
    
    normal_train_df, normal_valid_df = create_normal_df(path, signal_list)
    
    window_size = 50
    # Reshape so that we have samples of length window_size instead of seq_length 
    trainset, validset = create_torch_dataset(normal_train_df, normal_valid_df, signal_list, window_size)
    
    

    beta = 0.1
    gp_vae = MIMIC_GP_AVE(latent_dim=25, data_dim=3, time_length=50, length_scale=2, window_size=50, beta=beta)

    if args.train:
        print("Training..")
        gp_vae.train(trainset=trainset, validset=validset)
    print('done')
    x, x_hat_mean, x_hat_std, x_lens = gp_vae.impute(trainset) # x is unmodified test data, x_hat_mean and std are mean and std of the dist the model outputs over the input space
    rnd_int = np.random.randint(len(x))
    gp_vae.plot_counterfactual(x[rnd_int], x_hat_mean[rnd_int], x_hat_std[rnd_int], x_lens[rnd_int], signal_list, {})


if __name__ == "__main__":
    random.seed(2021)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    if not os.path.exists('./outputs/logs'):
        os.mkdir('./outputs/logs')
    if not os.path.exists('./outputs/plots'):
        os.mkdir('./outputs/plots')
    main(args)