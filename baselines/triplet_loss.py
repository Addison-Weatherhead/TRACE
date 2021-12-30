"""
Implementation of the Triplet Loss baseline based on the original code available on
https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
"""

import torch
import numpy as np
import argparse
import os
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from tnc.tnc import train_linear_classifier

from tnc.models import Chomp1d, SqueezeChannels, CausalConvolutionBlock, CausalCNN
from tnc.utils import plot_distribution, model_distribution
#from tnc.evaluations import ClassificationPerformanceExperiment, WFClassificationExperiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).
    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param encoding_size Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param window_size windows of data to encode at a time. It should be ensured that this evenly
            divides into the length of the time series passed in. E.g. window_size is 120, and 
            x.shape[-1] (where x is passed into forward) is 120*c for some natural number c
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 encoding_size, kernel_size, device, window_size):
        super(CausalCNNEncoder, self).__init__()
        self.encoding_size = encoding_size
        self.device = device
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, encoding_size)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        ).to(device)

        self.window_size = window_size

        

    def forward(self, x):
        x = x.to(self.device)
        
        
        if len(tuple(x.shape)) == 2:
            x = torch.unsqueeze(x, 0) # Make it a batch of size 1, shape is (1, num_features, seq_len)
            

        return self.network(x)

    def forward_seq(self, x, sliding_gap=None):
        '''Takes a tensor of shape (num_samples, num_features, seq_len) of timeseries data.
        
        Returns a tensor of shape (num_samples, seq_len/winow_size, encoding_size)'''
        
        assert x.shape[-1] % self.window_size == 0
        if len(tuple(x.shape)) == 2:
            x = torch.unsqueeze(x, 0)

        if sliding_gap:
            # This is a tensor of indices. If the data is of shape (num_samples, 2, num_features, 10), and window_size = 4 and sliding_gap=2, then inds is an array
            # of [0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9]
            inds = torch.cat([torch.arange(ind, ind+self.window_size) for ind in range(0, x.shape[-1]-self.window_size+1, sliding_gap)])
            
            # Now for each sample we have the window_size windows concatenated for each sliding gap on the last axis.
            # So if window_size is 120 and sliding_gap is 20, then for each sample, the time dimension will go 
            # [0, 1, 2, ..., 119, 20, 21, ..., 139, 140, ...]
            x = torch.index_select(input=x, dim=3, index=inds)

        
        
        num_samples, num_features, seq_len = x.shape
        #print('entering forward_seq!')
        #print('num_samples, two, num_features, seq_len', num_samples, two, num_features, seq_len)
        #x = torch.reshape(x, (num_samples, num_features*2, seq_len)) # now x is of shape (num_samples, 2*num_features, seq_len)
        #print(x.shape, '==', num_samples, 2*num_features, seq_len)
        x = x.permute(1, 0, 2) # now of shape (num_features, num_samples, seq_len)
        x = x.reshape(num_features, -1) # Now of shape (num_features, num_samples*seq_len)
        #print(x.shape, '==', 2*num_features, num_samples*seq_len)
        x = torch.stack(torch.split(x, self.window_size, dim=1)) # Now of shape (num_samples*(seq_len/window_size), 2*num_features, window_size)
        #print(x.shape, '==', num_samples*(seq_len/self.window_size), 2*num_features, self.window_size)

        encodings = self.forward(x) # encodings is of shape (num_samples*(seq_len/window_size), encoding_size)
        #print(encodings.shape, '==', num_samples*(seq_len/self.window_size), self.encoding_size)
        encodings = encodings.reshape(num_samples, int(seq_len/self.window_size), self.encoding_size)
        
        
        return encodings




class TripletLoss(torch.nn.modules.loss._Loss):
    """
    Triplet loss for representations of time series. Optimized for training
    sets where all time series have the same length.
    Takes as input a tensor as the chosen batch to compute the loss,
    a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing
    the training set, where `B` is the batch size, `C` is the number of
    channels and `L` is the length of the time series, as well as a boolean
    which, if True, enables to save GPU memory by propagating gradients after
    each loss term, instead of doing it after computing the whole loss.
    The triplets are chosen in the following manner. First the size of the
    positive and negative samples are randomly chosen in the range of lengths
    of time series in the dataset. The size of the anchor time series is
    randomly chosen with the same length upper bound but the the length of the
    positive samples as lower bound. An anchor of this length is then chosen
    randomly in the given time series of the train set, and positive samples
    are randomly chosen among subseries of the anchor. Finally, negative
    samples of the chosen length are randomly chosen in random time series of
    the train set.
    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty):
        super(TripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = np.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train, save_memory=False):
        batch=batch.to(device)
        train=train.to(device)
        encoder = encoder.to(device)
        batch_size = batch.size(0)
        train_size = train.size(0)
        length = min(self.compared_length, train.size(2))

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = np.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # Choice of length of positive and negative samples
        length_pos_neg = self.compared_length
        # length_pos_neg = np.random.randint(1, high=length + 1)


        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = self.compared_length

        beginning_batches = np.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = np.random.randint(
            0, high=random_length + 1, size=batch_size
        )  # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg + np.random.randint(0,self.compared_length)

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = np.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )


        representation = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]).to(device))  # Anchors representations

        positive_representation = encoder(torch.cat(
            [batch[
                j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
            ] for j in range(batch_size)]
        ))  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = encoder(
                torch.cat([train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)])
            )
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss


def epoch_run(data, encoder, device, window_size, optimizer=None, train=True):
    if train:
        encoder.train()
    else:
        encoder.eval()
    encoder = encoder.to(device)
    loss_criterion = TripletLoss(compared_length=window_size, nb_random_samples=10, negative_penalty=1)

    epoch_loss = 0
    acc = 0
    dataset = torch.utils.data.TensorDataset(torch.Tensor(data).to(device), torch.zeros((len(data),1)).to(device))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)
    i = 0
    for x_batch,y in data_loader:
        loss = loss_criterion(x_batch.to(device), encoder, torch.Tensor(data).to(device))
        epoch_loss += loss.item()
        i += 1
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return epoch_loss/i, acc/i


def learn_encoder(x, window_size, data, lr=0.001, decay=0, n_epochs=100, device='cpu', n_cross_val=1):
    if not os.path.exists("./DONTCOMMITplots/%s_trip/"%data):
        os.mkdir("./DONTCOMMITplots/%s_trip/"%data)
    if not os.path.exists("./ckpt/%s_trip/"%data):
        os.mkdir("./ckpt/%s_trip/"%data)
    for cv in range(n_cross_val):
        if 'ICU' in data:
            encoding_size = 16
            encoder = CausalCNNEncoder(in_channels=10, channels=8, depth=2, reduced_size=60, encoding_size=encoding_size, kernel_size=3, window_size=window_size, device=device)
        elif 'HiRID' in data:
            encoding_size = 10
            encoder = CausalCNNEncoder(in_channels=18, channels=4, depth=1, reduced_size=2, encoding_size=encoding_size, kernel_size=6, window_size=window_size, device=device)
        
        params = encoder.parameters()
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
        inds = list(range(len(x)))
        random.shuffle(inds)
        x = x[inds]
        n_train = int(0.8*len(x))
        train_loss, test_loss = [], []
        best_loss = np.inf
        for epoch in range(n_epochs):
            epoch_loss, acc = epoch_run(x[:n_train], encoder, device, window_size, optimizer=optimizer, train=True)
            epoch_loss_test, acc_test = epoch_run(x[n_train:], encoder, device, window_size, optimizer=optimizer, train=False)
            if epoch%10==0:
                print('\nEpoch ', epoch)
                print('Train ===> Loss: ', epoch_loss, '\t Test ===> Loss: ', epoch_loss_test )
            train_loss.append(epoch_loss)
            test_loss.append(epoch_loss_test)
            if epoch_loss_test<best_loss:
                print('Save new ckpt')
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict()
                }
                best_loss = epoch_loss_test
                torch.save(state, './ckpt/%s_trip/checkpoint_%d.pth.tar' %(data, cv))
        plt.figure()
        plt.plot(np.arange(n_epochs), train_loss, label="Train")
        plt.plot(np.arange(n_epochs), test_loss, label="Test")
        plt.title("Loss")
        plt.legend()
        plt.savefig(os.path.join("./DONTCOMMITplots/%s_trip/loss_%d.pdf"%(data,cv)))


def main(is_train, data_type, cv):
    if not os.path.exists("./DONTCOMMITplots"):
        os.mkdir("./DONTCOMMITplots")
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")

    if data_type == 'ICU':
        length_of_hour = int(60*60/5)
        pos_sample_name = 'arrest'
        path = '/datasets/sickkids/TNC_ICU_data/'
        signal_list = ["Pulse", "HR", "SpO2", "etCO2", "NBPm", "NBPd", "NBPs", "RR", "CVPm", "awRR"]
        window_size = 120
        encoding_size = 16
        n_epochs = 150
        lr = 1e-3

        num_pre_positive_encodings = 12 # 6 per hr, last 2 hrs is the pre arrest window
       
        TEST_mixed_data_maps = torch.from_numpy(np.load(os.path.join(path, 'test_mixed_data_maps.npy')))
        TEST_mixed_labels = torch.from_numpy(np.load(os.path.join(path, 'test_mixed_labels.npy')))

        train_mixed_data_maps = torch.from_numpy(np.load(os.path.join(path, 'train_mixed_data_maps.npy')))
        train_mixed_labels = torch.from_numpy(np.load(os.path.join(path, 'train_mixed_labels.npy')))

    elif data_type == 'HiRID':
        window_size = 3
        encoding_size = 10
        n_epochs = 150
        lr = 1e-3
        length_of_hour = int((60*60)/300) # 60 seconds * 60 / 300 (which is num seconds in 5 min)
        pos_sample_name = 'mortality'
        path = '../DONTCOMMITdata/hirid_numpy'
        signal_list = ['vm1', 'vm3', 'vm4', 'vm5', 'vm13', 'vm20', 'vm28', 'vm62', 'vm136', 'vm146', 'vm172', 'vm174', 'vm176', 'pm41', 'pm42', 'pm43', 'pm44', 'pm87']
        sliding_gap = 1
        pre_positive_window = int((24*60*60)/300) # 24 hrs
        num_pre_positive_encodings = int(pre_positive_window/window_size)

        TEST_mixed_data_maps = torch.from_numpy(np.load(os.path.join(path, 'TEST_data_maps.npy'))).float()
        TEST_mixed_labels = torch.from_numpy(np.load(os.path.join(path, 'TEST_labels.npy'))).float()

        train_mixed_data_maps = torch.from_numpy(np.load(os.path.join(path, 'train_data_maps.npy'))).float()
        train_mixed_labels = torch.from_numpy(np.load(os.path.join(path, 'train_labels.npy'))).float()


    if is_train:
        learn_encoder(train_mixed_data_maps[:,0,:,:], window_size, lr=lr, decay=1e-5, data=data_type, n_epochs=n_epochs, device=device, n_cross_val=cv)
        
        
    classifier_validation_aurocs = []
    classifier_validation_auprcs = []
    classifier_TEST_aurocs = []
    classifier_TEST_auprcs = []
    for encoder_cv in range(cv):
        print('Encoder CV: ', encoder_cv)
        seed_val = 111*encoder_cv+2
        random.seed(seed_val)
        print("Seed set to: ", seed_val)

        checkpoint = torch.load('ckpt/%s_trip/checkpoint_%d.pth.tar'%(data_type, encoder_cv))
        if data_type == 'ICU':
            encoder = CausalCNNEncoder(in_channels=10, channels=8, depth=2, reduced_size=60, encoding_size=encoding_size, kernel_size=3, window_size=window_size, device=device)
        elif data_type == 'HiRID':
            encoder = CausalCNNEncoder(in_channels=18, channels=4, depth=1, reduced_size=2, encoding_size=encoding_size, kernel_size=6, window_size=window_size, device=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.eval()


        # Shuffle data
        print('Original shape of train data: ')
        print(train_mixed_data_maps.shape)
        # shuffle for this cv:
        inds = np.arange(len(train_mixed_data_maps))
        random.shuffle(inds)
        print("First 15 inds: ", inds[:15])

        n_valid = int(0.2*len(train_mixed_data_maps))
        validation_mixed_data_maps_cv = train_mixed_data_maps[inds][0:n_valid]
        validation_mixed_labels_cv = train_mixed_labels[inds][0:n_valid]
        print("Size of valid data: ", validation_mixed_data_maps_cv.shape)
        print("Size of valid labels: ", validation_mixed_labels_cv.shape)
        print("num positive valid samples: ", sum([1 in validation_mixed_labels_cv[ind] for ind in range(len(validation_mixed_labels_cv))]))

        train_mixed_data_maps_cv = train_mixed_data_maps[inds][n_valid:]
        train_mixed_labels_cv = train_mixed_labels[inds][n_valid:]
        print("Size of train data: ", train_mixed_data_maps_cv.shape)
        print("Size of train labels: ", train_mixed_labels_cv.shape)
        print("num positive train samples: ", sum([1 in train_mixed_labels_cv[ind] for ind in range(len(train_mixed_labels_cv))]))

        print('Size of TEST data: ', TEST_mixed_data_maps.shape)
        print('Size of TEST labels: ', TEST_mixed_labels.shape)
        print("Num positive TEST samples: ", sum([1 in TEST_mixed_labels[ind] for ind in range(len(TEST_mixed_labels))]))
            
        print('Levels of missingness, for Train, Validation, and TEST')
            
        print(len(torch.where(train_mixed_data_maps_cv[:, 1, :, :] == 0)[0])/ \
        (train_mixed_data_maps_cv.shape[0]*train_mixed_data_maps_cv.shape[2]*train_mixed_data_maps_cv.shape[3]))

        print(len(torch.where(validation_mixed_data_maps_cv[:, 1, :, :] == 0)[0])/ \
        (validation_mixed_data_maps_cv.shape[0]*validation_mixed_data_maps_cv.shape[2]*validation_mixed_data_maps_cv.shape[3]))

        print(len(torch.where(TEST_mixed_data_maps[:, 1, :, :] == 0)[0])/ \
        (TEST_mixed_data_maps.shape[0]*TEST_mixed_data_maps.shape[2]*TEST_mixed_data_maps.shape[3]))
            
        print('Mean of signal 1 for train:')
        print(torch.mean(train_mixed_data_maps_cv[:, 0, 0, :]))
        print('Mean of signal 2 for train:')
        print(torch.mean(train_mixed_data_maps_cv[:, 0, 1, :]))
        print('Mean of signal 3 for train:')
        print(torch.mean(train_mixed_data_maps_cv[:, 0, 2, :]))


        print('Mean of signal 1 for validation:')
        print(torch.mean(validation_mixed_data_maps_cv[:, 0, 0, :]))
        print('Mean of signal 2 for validation:')
        print(torch.mean(validation_mixed_data_maps_cv[:, 0, 1, :]))
        print('Mean of signal 3 for validation:')
        print(torch.mean(validation_mixed_data_maps_cv[:, 0, 2, :]))


        print('Mean of signal 1 for TEST:')
        print(torch.mean(TEST_mixed_data_maps[:, 0, 0, :]))
        print('Mean of signal 2 for TEST:')
        print(torch.mean(TEST_mixed_data_maps[:, 0, 1, :]))
        print('Mean of signal 3 for TEST:')
        print(torch.mean(TEST_mixed_data_maps[:, 0, 2, :]))


        print("TRAINING LINEAR CLASSIFIER")
        classifier_train_labels = torch.Tensor([1 in label for label in train_mixed_labels_cv]) # Sets labels for positive samples to 1
        classifier_validation_labels = torch.Tensor([1 in label for label in validation_mixed_labels_cv]) # Sets labels for positive samples to 1
        classifier_TEST_labels = torch.Tensor([1 in label for label in TEST_mixed_labels]) # Sets labels for positive samples to 1
            

        train_mixed_data_maps_cv = train_mixed_data_maps_cv[:, 0, :, :] # Only keep data, not mask
            
        validation_mixed_data_maps_cv = validation_mixed_data_maps_cv[:, 0, :, :] # Only keep data, not mask
            
        TEST_mixed_data_maps = TEST_mixed_data_maps[:, 0, :, :] # Only keep data, not mask

        rnn, classifier, valid_auroc, valid_auprc, TEST_auroc, TEST_auprc = train_linear_classifier(X_train=train_mixed_data_maps_cv, y_train=classifier_train_labels, 
                X_validation=validation_mixed_data_maps_cv, y_validation=classifier_validation_labels, 
                X_TEST=TEST_mixed_data_maps, y_TEST=classifier_TEST_labels,
                encoding_size=encoder.encoding_size, batch_size=20, num_pre_positive_encodings=num_pre_positive_encodings, 
                encoder=encoder, return_models=True, return_scores=True, pos_sample_name=pos_sample_name, 
                data_type=data_type, classification_cv=0, encoder_cv=encoder_cv)

        classifier_validation_aurocs.append(valid_auroc)
        classifier_validation_auprcs.append(valid_auprc)
        classifier_TEST_aurocs.append(TEST_auroc)
        classifier_TEST_auprcs.append(TEST_auprc)



    print("CLASSIFICATION VALIDATION RESULT OVER CV")
    print("AUC: %.2f +- %.2f, AUPRC: %.2f +- %.2f"% \
        (np.mean(classifier_validation_aurocs), 
        np.std(classifier_validation_aurocs), 
        np.mean(classifier_validation_auprcs), 
        np.std(classifier_validation_auprcs)))

    print("CLASSIFICATION TEST RESULT OVER CV")
    print("AUC: %.2f +- %.2f, AUPRC: %.2f +- %.2f"% \
        (np.mean(classifier_TEST_aurocs), 
        np.std(classifier_TEST_aurocs), 
        np.mean(classifier_TEST_auprcs), 
        np.std(classifier_TEST_auprcs)))


if __name__=="__main__":
    random.seed(1234)
    parser = argparse.ArgumentParser(description='Run Triplet Loss')
    parser.add_argument('--data', type=str, default='ICU')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    main(args.train, args.data, args.cv)
