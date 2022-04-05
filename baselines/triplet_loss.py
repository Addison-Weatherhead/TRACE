"""
Implementation of the Triplet Loss baseline based on the original code available on
https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
"""

import torch
import numpy as np
import argparse
import os
import random
import math
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#from tnc.tnc import train_linear_classifier

from tnc.models import Chomp1d, SqueezeChannels, CausalConvolutionBlock, CausalCNN, RnnPredictor, LinearClassifier
from tnc.utils import plot_distribution, model_distribution
#from tnc.tnc import linear_classifier_epoch_run 
#from tnc.evaluations import ClassificationPerformanceExperiment, WFClassificationExperiment
from statsmodels.tsa import stattools
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, roc_curve
from sklearn.cluster import KMeans

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

def linear_classifier_epoch_run(dataset, train, classifier, optimizer, data_type, window_size, encoder, encoding_size):
    if train:
        classifier.train()
    else:
        classifier.eval()

    epoch_predictions, epoch_losses, epoch_labels = [], [], []
    for data_batch, label_batch in dataset:

        # data is of shape (num_samples, num_encodings_per_sample, encoding_size)
        if encoder is None:
            encoding_batch = data_batch
        else:
            encoding_batch = encoder.forward_seq(data_batch)
        #encoding_batch, encoding_mask = encoder.forward_seq(data_batch, return_encoding_mask=True)
        encoding_batch = encoding_batch.to(device)
        if data_type == 'ICU':
            # Takes every window_size'th label for each sample (so it matches the frequency of encodings)
            # Then reshapes to be of shape (num_samples*num_encodings_per_sample)
            label_batch = label_batch[:, -1]#[:,window_size-1::window_size].to(device)

            label_batch = label_batch.reshape(-1,)

            # Now the encodings are of shape (num_samples*num_encodings_per_sample, encoding_size)
            # and the masks are of shape (num_samples*num_encodings_per_sample)
            if encoder is None:
                encoding_batch = encoding_batch[:,-12*60:,:]
            else:
                encoding_batch = encoding_batch[:,-12:,:]
            #encoding_batch = encoding_batch.reshape(-1, encoding_size)
            #encoding_batch, encoding_mask = encoding_batch.reshape(-1, encoding_size), encoding_mask.reshape(-1,)

            # Now we'll remove encodings that were derived from fully imputed data
            #encoding_batch = encoding_batch[torch.where(encoding_mask != -1)]

            # Remove the corresponding labels
            #label_batch = label_batch[torch.where(encoding_mask != -1)]

            # Now encoding_batch is of shape (num_encodings_kept, encoding_size)
            # and label_batch is of shape (num_encodings_kept,)

        elif data_type == 'HiRID':
            label_batch = torch.Tensor([1 in label for label in label_batch]).to(device)


        # Shape of predictions and window_labels is (batch_size)
        #window_labels = torch.squeeze(window_labels).to(device)
        # print('Manually setting pos_weight to 10')
        predictions = torch.squeeze(classifier(encoding_batch))
        pos_weight = torch.Tensor([10]).to(device)
        if train:
            # Ratio of num negative examples divided by num positive examples is pos_weight
            # pos_weight = torch.Tensor([negative_encodings.shape[0] / max(positive_encodings.shape[0], 1)]).to(device)
            #print('pos_weight: ', pos_weight)

            #print('No positive weight set')
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Applies sigmoid to outputs passed in so we shouldn't have sigmoid in the model. 
            loss = loss_fn(predictions, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(predictions, label_batch)

        epoch_loss = loss.item()

        # Apply sigmoid to predictions since we didn't apply it for the loss function since the loss function does sigmoid on its own.
        predictions = torch.nn.Sigmoid()(predictions)

        # Move Tensors to CPU and remove gradients so they can be converted to NumPy arrays in the sklearn functions
        #window_labels = window_labels.cpu().detach()
        predictions = predictions.cpu().detach()
        encoding_batch = encoding_batch.cpu() # Move off GPU memory
        #neg_window_labels = neg_window_labels.cpu()
        #pos_window_labels = pos_window_labels.cpu()
        label_batch = label_batch.cpu()

        epoch_losses.append(epoch_loss)
        epoch_predictions.append(predictions)
        epoch_labels.append(label_batch)

    return epoch_predictions, epoch_losses, epoch_labels


def train_linear_classifier(X_train, y_train, X_validation, y_validation, X_TEST, y_TEST, encoding_size, num_pre_positive_encodings, encoder, window_size, exp_type, batch_size=32, return_models=False, return_scores=False, pos_sample_name='arrest', data_type='ICU', classification_cv=0, encoder_cv=0, ckpt_path="../ckpt",  plt_path="../DONTCOMMITplots"):
    '''
    Trains an RNN and linear classifier jointly. X_train is of shape (num_samples, num_windows_per_hour, encoding_size)
    and y_train is of shape (num_samples)

    '''
    if not os.path.exists(os.path.join(ckpt_path, '%s_%s'%(data_type, exp_type))):
        os.mkdir(os.path.join(ckpt_path, '%s_%s'%(data_type, exp_type)))
    if not os.path.exists(os.path.join(plt_path, '%s_%s'%(data_type, exp_type))):
        os.mkdir(os.path.join(plt_path, '%s_%s'%(data_type, exp_type)))
    
    print("Training Linear Classifier", flush=True)
    if data_type=='ICU':
        classifier = RnnPredictor(encoding_size=encoding_size, hidden_size=8).to(device)
        #classifier = LinearClassifier(input_size=encoding_size).to(device)
    elif data_type=='HiRID':
        classifier = RnnPredictor(encoding_size=encoding_size, hidden_size=8).to(device)


    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = torch.utils.data.TensorDataset(X_validation, y_validation)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    TEST_dataset = torch.utils.data.TensorDataset(X_TEST, y_TEST)
    TEST_data_loader = torch.utils.data.DataLoader(TEST_dataset, batch_size=batch_size, shuffle=True)

    params = classifier.parameters() 
    lr = .001
    weight_decay = .005
    print('Learning Rate for classifier training: ', lr)
    print('Weight Decay for classifier training: ', weight_decay)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    train_losses = []
    valid_losses = []

    for epoch in range(1, 101):
        epoch_train_predictions, epoch_train_losses, epoch_train_labels = linear_classifier_epoch_run(dataset=train_data_loader, train=True,
                                                    classifier=classifier,
                                                    optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=encoding_size)

        epoch_validation_predictions, epoch_validation_losses, epoch_validation_labels = linear_classifier_epoch_run(dataset=validation_data_loader, train=False,
                                                    classifier=classifier,
                                                    optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=encoding_size)

        epoch_TEST_predictions, epoch_TEST_losses, epoch_TEST_labels = linear_classifier_epoch_run(dataset=TEST_data_loader, train=False,
                                                    classifier=classifier,
                                                    optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=encoding_size)


        #epoch_train_predictions, epoch_train_losses, epoch_train_labels = linear_classifier_epoch_run(data=X_train, labels=y_train, train=True,
        #                                            num_pre_positive_encodings=num_pre_positive_encodings,
        #                                            batch_size=batch_size, classifier=classifier,
        #                                            optimizer=optimizer, encoder=encoder)

        #epoch_validation_predictions, epoch_validation_losses, epoch_validation_labels = linear_classifier_epoch_run(data=X_validation, labels=y_validation, train=False,
        #                                            num_pre_positive_encodings=num_pre_positive_encodings,
        #                                            batch_size=batch_size, classifier=classifier,
        #                                            optimizer=optimizer, encoder=encoder)




        #epoch_TEST_predictions, epoch_TEST_losses, epoch_TEST_labels = linear_classifier_epoch_run(data=X_TEST, labels=y_TEST, train=False,
        #                                            num_pre_positive_encodings=num_pre_positive_encodings,
        #                                            batch_size=batch_size, classifier=classifier,
        #                                            optimizer=optimizer, encoder=encoder)


        # TRAIN 
        # Compute average over all batches in the epoch
        epoch_train_loss = np.mean(epoch_train_losses)
        epoch_train_predictions = torch.cat(epoch_train_predictions)
        epoch_train_labels = torch.cat(epoch_train_labels)

        epoch_train_auroc = roc_auc_score(epoch_train_labels, epoch_train_predictions)
        # Compute precision recall curve
        train_precision, train_recall, _ = precision_recall_curve(epoch_train_labels, epoch_train_predictions)
        # Compute AUPRC
        epoch_train_auprc = auc(train_recall, train_precision) # precision is the y axis, recall is the x axis, computes AUC of this curve

        train_losses.append(epoch_train_loss)

        # VALIDATION
        epoch_validation_loss = np.mean(epoch_validation_losses)
        epoch_validation_predictions = torch.cat(epoch_validation_predictions)
        epoch_validation_labels = torch.cat(epoch_validation_labels)

        epoch_validation_auroc = roc_auc_score(epoch_validation_labels, epoch_validation_predictions)
        # Compute precision recall curve
        valid_precision, valid_recall, _ = precision_recall_curve(epoch_validation_labels, epoch_validation_predictions)
        # Compute AUPRC
        epoch_validation_auprc = auc(valid_recall, valid_precision) # precision is the y axis, recall is the x axis, computes AUC of this curve
        valid_losses.append(epoch_validation_loss)

        # TEST
        epoch_TEST_loss = np.mean(epoch_TEST_losses)
        epoch_TEST_predictions = torch.cat(epoch_TEST_predictions)
        epoch_TEST_labels = torch.cat(epoch_TEST_labels)

        epoch_TEST_auroc = roc_auc_score(epoch_TEST_labels, epoch_TEST_predictions)
        # Compute precision recall curve
        TEST_precision, TEST_recall, _ = precision_recall_curve(epoch_TEST_labels, epoch_TEST_predictions)
        # Compute AUPRC
        epoch_TEST_auprc = auc(TEST_recall, TEST_precision) # precision is the y axis, recall is the x axis, computes AUC of this curve


        if epoch%10==0:
            print('Epoch %d Classifier Loss =====> Training Loss: %.5f \t Training AUROC: %.5f \t Training AUPRC: %.5f \n\t Validation Loss: %.5f \t Validation AUROC: %.5f \t Validation AUPRC %.5f \n\t TEST Loss: %.5f \t TEST AUROC: %.5f \t TEST AUPRC %.5f'
                                % (epoch, epoch_train_loss, epoch_train_auroc, epoch_train_auprc, epoch_validation_loss, epoch_validation_auroc, 
                                    epoch_validation_auprc, epoch_TEST_loss, epoch_TEST_auroc, epoch_TEST_auprc))
            
            # Checkpointing the classifier model
            state = {
                    'epoch': epoch,
                    'classifier_state_dict': classifier.state_dict(),
                }

            torch.save(state, os.path.join(ckpt_path, '%s_%s/encoder_checkpoint_%d_Classifier_checkpoint_%d.tar'%(data_type, exp_type, encoder_cv, classification_cv)))                    
                                
    


    # Plot ROC and PR Curves
    plt.figure()
    plt.plot(train_recall, train_precision)
    plt.title('Train Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(plt_path, '%s_%s/train_PR_curve_%d_%d.pdf'%(data_type, exp_type, encoder_cv, classification_cv)))

    plt.figure()
    fpr, tpr, _ = roc_curve(epoch_train_labels, epoch_train_predictions)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Train ROC Curve')
    plt.savefig(os.path.join(plt_path, '%s_%s/train_ROC_curve_%d_%d.pdf'%(data_type, exp_type, encoder_cv, classification_cv)))



    plt.figure()
    plt.plot(valid_recall, valid_precision)
    plt.title('Validation Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(plt_path, '%s_%s/valid_PR_curve_%d_%d.pdf'%(data_type, exp_type, encoder_cv, classification_cv)))

    plt.figure()
    fpr, tpr, _ = roc_curve(epoch_validation_labels, epoch_validation_predictions)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Validation ROC Curve')
    plt.savefig(os.path.join(plt_path, '%s_%s/valid_ROC_curve_%d_%d.pdf'%(data_type, exp_type, encoder_cv, classification_cv)))

    plt.figure()
    plt.plot(TEST_recall, TEST_precision)
    plt.title('TEST Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(plt_path, '%s_%s/TEST_PR_curve_%d_%d.pdf'%(data_type, exp_type, encoder_cv, classification_cv)))

    plt.figure()
    fpr, tpr, _ = roc_curve(epoch_TEST_labels, epoch_TEST_predictions)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('TEST ROC Curve')

    # Plot Loss curves
    plt.figure()
    plt.plot(np.arange(1, 101), train_losses, label="Train")
    plt.plot(np.arange(1, 101), valid_losses, label="Validation")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join(plt_path, "%s_%s/Classifier_loss_%d_%d.pdf"%(data_type, exp_type, encoder_cv, classification_cv)))


    epoch_train_predictions[epoch_train_predictions >= 0.5] = 1
    epoch_train_predictions[epoch_train_predictions < 0.5] = 0

    epoch_validation_predictions[epoch_validation_predictions >= 0.5] = 1
    epoch_validation_predictions[epoch_validation_predictions < 0.5] = 0

    epoch_TEST_predictions[epoch_TEST_predictions >= 0.5] = 1
    epoch_TEST_predictions[epoch_TEST_predictions < 0.5] = 0

    print("Train classification report: ")
    print('epoch_train_labels shape: ', epoch_train_labels.shape, 'epoch_train_predictions shape: ', epoch_train_predictions.shape)
    print(classification_report(epoch_train_labels.to('cpu'), epoch_train_predictions, target_names=['normal', pos_sample_name]))
    print("Validation classification report: ")
    print(classification_report(epoch_validation_labels.to('cpu'), epoch_validation_predictions, target_names=['normal', pos_sample_name]))
    print("TEST classification report: ")
    print(classification_report(epoch_TEST_labels.to('cpu'), epoch_TEST_predictions, target_names=['normal', pos_sample_name]))

    if return_models and return_scores:
        return (classifier, epoch_validation_auroc, epoch_validation_auprc, epoch_TEST_auroc, epoch_TEST_auprc)
    if return_models:
        return (classifier)
    if return_scores:
        return (epoch_validation_auroc, epoch_validation_auroc)






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


def learn_encoder(x, window_size, data, encoding_size, lr=0.001, decay=0, n_epochs=100, device='cpu', n_cross_val=1):
    if not os.path.exists("./DONTCOMMITplots/%s_trip/"%data):
        os.mkdir("./DONTCOMMITplots/%s_trip/"%data)
    if not os.path.exists("./ckpt/%s_trip/"%data):
        os.mkdir("./ckpt/%s_trip/"%data)
    for cv in range(n_cross_val):
        if 'ICU' in data:
            #encoding_size = 16
            encoder = CausalCNNEncoder(in_channels=10, channels=8, depth=2, reduced_size=30, encoding_size=encoding_size, kernel_size=3, window_size=window_size, device=device)
        elif 'HiRID' in data:
            #encoding_size = 10
            encoder = CausalCNNEncoder(in_channels=18, channels=4, depth=1, reduced_size=2, encoding_size=encoding_size, kernel_size=2, window_size=window_size, device=device)
        
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


def main(is_train, data_type, lr, cv):
    if not os.path.exists("./DONTCOMMITplots"):
        os.mkdir("./DONTCOMMITplots")
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")

    if data_type == 'ICU':
        length_of_hour = int(60*60/5)
        pos_sample_name = 'arrest'
        path = '/datasets/sickkids/TNC_ICU_data/'
        signal_list = ["Pulse", "HR", "SpO2", "etCO2", "NBPm", "NBPd", "NBPs", "RR", "CVPm", "awRR"]
        window_size = 60
        encoding_size = 6#16
        n_epochs = 150
        lr = 1e-3

        num_pre_positive_encodings = 12 # 6 per hr, last 2 hrs is the pre arrest window
       
        TEST_mixed_data_maps = torch.from_numpy(np.load(os.path.join(path, 'test_mixed_data_maps.npy')))
        TEST_mixed_labels = torch.from_numpy(np.load(os.path.join(path, 'test_mixed_labels.npy')))

        train_mixed_data_maps = torch.from_numpy(np.load(os.path.join(path, 'train_mixed_data_maps.npy')))
        train_mixed_labels = torch.from_numpy(np.load(os.path.join(path, 'train_mixed_labels.npy')))
        # ******************************** TRAIN ENCODER *************************************
        if is_train:
            learn_encoder(train_mixed_data_maps[:,0,:,:], window_size, encoding_size=encoding_size, lr=lr, decay=1e-5, data=data_type, n_epochs=n_epochs, device=device, n_cross_val=1)

    elif data_type == 'HiRID':
        window_size = 8
        encoding_size = 6#10
        n_epochs = 150
        lr = 1e-3
        length_of_hour = int((60*60)/300) # 60 seconds * 60 / 300 (which is num seconds in 5 min)
        pos_sample_name = 'mortality'
        path = '../DONTCOMMITdata/hirid_numpy'
        signal_list = ['vm1', 'vm3', 'vm4', 'vm5', 'vm13', 'vm20', 'vm28', 'vm62', 'vm136', 'vm146', 'vm172', 'vm174', 'vm176', 'pm41', 'pm42', 'pm43', 'pm44', 'pm87']
        sliding_gap = 1
        pre_positive_window = int((24*60*60)/300) # 24 hrs
        num_pre_positive_encodings = int(pre_positive_window/window_size)
        
        # NOTE THE MAP CHANNEL HAS 1'S FOR OBSERVED VALUES, 0'S FOR MISSING VALUES
        # data_maps arrays are of shape (num_samples, 2, 18, 1152). 4 days of data per sample
        TEST_mixed_data_maps = torch.from_numpy(np.load(os.path.join(path, 'TEST_mortality_data_maps.npy'))).float()
        TEST_mixed_mortality_labels = torch.from_numpy(np.load(os.path.join(path, 'TEST_mortality_labels.npy'))).float()
        TEST_mixed_labels = TEST_mixed_mortality_labels
        #TEST_PIDs = torch.from_numpy(np.load(os.path.join(path, 'TEST_PIDs.npy'))).float()
        TEST_Apache_Groups = torch.from_numpy(np.load(os.path.join(path, 'TEST_Apache_Groups.npy'))).float()


        train_mixed_data_maps = torch.from_numpy(np.load(os.path.join(path, 'train_mortality_data_maps.npy'))).float()
        train_mixed_mortality_labels = torch.from_numpy(np.load(os.path.join(path, 'train_mortality_labels.npy'))).float()
        train_mixed_labels = train_mixed_mortality_labels
        #train_PIDs = torch.from_numpy(np.load(os.path.join(path, 'train_PIDs.npy'))).float()
        train_Apache_Groups = torch.from_numpy(np.load(os.path.join(path, 'train_Apache_Groups.npy'))).float()


        # Used for training encoder
        train_encoder_data_maps = torch.from_numpy(np.load(os.path.join(path, 'train_encoder_data_maps.npy'))).float()
        TEST_encoder_data_maps = torch.from_numpy(np.load(os.path.join(path, 'TEST_encoder_data_maps.npy'))).float()

        # Apache groups are either apache 2 or apache 4 codes. We consolodate into a single set of codes
        #  according to mapping here: https://docs.google.com/spreadsheets/d/16IYawLlASYbCekQe2_kUKxQZrwjIe4ibkPt7UU1u5pE/edit?usp=sharing

        apache_codes = [98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 190, 191, 192, 193, 197, 194, 195, 196, 198, 199, 201, 200, 202, 203, 204, 205, 206]
        mappings = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19]
        apache_names = ['Cardiovascular', 'Pulmonary', 'Gastrointestinal', 'Neurological', 'Sepsis', 'Urogenital', 'Trauma', 'Metabolic/Endocrinology', 'Hematology', 'Other', 'Surgical Cardiovascular', 'Surgical Respiratory', 'Surgical Gastrointestinal', 'Surgical Neurological', 'Surgical Trauma', 'Surgical Urogenital', 'Surgical Gynecology', 'Surgical Orthopedics', 'Surgical others', 'Intoxication']
        for i in range(len(apache_codes)):
            train_Apache_Groups[torch.where(train_Apache_Groups == apache_codes[i])] = mappings[i]
            TEST_Apache_Groups[torch.where(TEST_Apache_Groups == apache_codes[i])] = mappings[i]

        # Now for any patients that did not have any code, we'll categorize that as 'other'
        train_Apache_Groups[torch.where(train_Apache_Groups == -1)] = 9
        TEST_Apache_Groups[torch.where(TEST_Apache_Groups == -1)] = 9

        assert -1 not in train_Apache_Groups
        assert -1 not in TEST_Apache_Groups

        (unique, counts) = np.unique(train_Apache_Groups, return_counts=True)
        print('Distribution of train Apache states:')
        for i in range(len(unique)):
            print(unique[i], ': ', counts[i])
        print()
        (unique, counts) = np.unique(TEST_Apache_Groups, return_counts=True)
        print('Distribution of TEST Apache states:')
        for i in range(len(unique)):
            print(unique[i], ': ', counts[i])

        # ******************************** TRAIN ENCODER *************************************
        if is_train:
            learn_encoder(train_encoder_data_maps[:,0,:,:], window_size, lr=lr, encoding_size=encoding_size, decay=1e-5, data=data_type, n_epochs=n_epochs, device=device, n_cross_val=1)
        
        
    classifier_validation_aurocs = []
    classifier_validation_auprcs = []
    classifier_TEST_aurocs = []
    classifier_TEST_auprcs = []
    for encoder_cv in range(cv):
        print('Encoder CV: ', encoder_cv)
        seed_val = 111*encoder_cv+2
        random.seed(seed_val)
        print("Seed set to: ", seed_val)

        checkpoint = torch.load('ckpt/%s_trip/checkpoint_0.pth.tar'%(data_type))
        if data_type == 'ICU':
            encoder = CausalCNNEncoder(in_channels=10, channels=8, depth=2, reduced_size=30, encoding_size=encoding_size, kernel_size=3, window_size=window_size, device=device)
        elif data_type == 'HiRID':
            encoder = CausalCNNEncoder(in_channels=18, channels=4, depth=1, reduced_size=2, encoding_size=encoding_size, kernel_size=2, window_size=window_size, device=device)
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
            

        print("TRAINING LINEAR CLASSIFIER")
        #classifier_train_labels = torch.Tensor([1 in label for label in train_mixed_labels_cv]) # Sets labels for positive samples to 1
        #classifier_validation_labels = torch.Tensor([1 in label for label in validation_mixed_labels_cv]) # Sets labels for positive samples to 1
        #classifier_TEST_labels = torch.Tensor([1 in label for label in TEST_mixed_labels]) # Sets labels for positive samples to 1
            

        #train_mixed_data_maps_cv = train_mixed_data_maps_cv[:, 0, :, :] # Only keep data, not mask
            
        #validation_mixed_data_maps_cv = validation_mixed_data_maps_cv[:, 0, :, :] # Only keep data, not mask
            
        #TEST_mixed_data_maps = TEST_mixed_data_maps[:, 0, :, :] # Only keep data, not mask

        classifier, valid_auroc, valid_auprc, TEST_auroc, TEST_auprc = train_linear_classifier(X_train=train_mixed_data_maps_cv[:, 0, :, :], y_train=train_mixed_labels_cv,
                X_validation=validation_mixed_data_maps_cv[:, 0, :, :], y_validation=validation_mixed_labels_cv, 
                X_TEST=TEST_mixed_data_maps[:, 0, :, :], y_TEST=TEST_mixed_labels, exp_type="trip", window_size=window_size,
                encoding_size=encoder.encoding_size, batch_size=20, num_pre_positive_encodings=num_pre_positive_encodings,
                encoder=encoder, return_models=True, return_scores=True, pos_sample_name=pos_sample_name,
                data_type=data_type, classification_cv=0, encoder_cv=encoder_cv, plt_path="./DONTCOMMITplots", ckpt_path="./ckpt")

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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()


    main(args.train, args.data, args.lr, args.cv)
