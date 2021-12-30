import torch
import numpy as np
import os
import pickle as pkl
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


from tnc.models import CausalCNNEncoder, LinearClassifier, RnnPredictor
from sklearn.metrics import roc_auc_score, average_precision_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(data_type, n_cv):
    overal_loss, overal_auc, overal_auprc = [], [], []
    for cv in range(n_cv):
        np.random.seed(cv * 9)
        if data_type == 'HiRID':
            pass
        '''
            window_size = 4
            train_path = './data/mimic/train'
            test_path = './data/mimic/test'
            lab_IDs = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                       'HEMATOCRIT', 'HEMOGLOBIN',
                       'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM',
                       'BUN',
                       'WBC']
            vital_IDs = ['HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2', 'Glucose', 'Temp']

            # Load processed MIMIC
            with open(os.path.join(train_path, 'x.pkl'), 'rb') as f:
                x_train = pkl.load(f)  # shape: [n_batch, mask, 28(n_features), time steps]
                x_train = x_train
            with open(os.path.join(train_path, 'y.pkl'), 'rb') as f:
                y_train = pkl.load(f)

            with open(os.path.join(test_path, 'x.pkl'), 'rb') as f:
                x_test = pkl.load(f)  # shape: [n_batch, mask, 28(n_features), time steps]
                x_test = x_test
            with open(os.path.join(test_path, 'y.pkl'), 'rb') as f:
                y_test = pkl.load(f)
            # Create model
            classifier = LinearClassifier(input_size=32)
            rnn = RnnPredictor(10, 32)
            encoder = GRUDEncoder(num_features=28, hidden_size=64, num_layers=1, encoding_size=10, extra_layer_types=None, device=device)
        '''

        if data_type == 'ICU':
            window_size = 120
            path = '/datasets/sickkids/TNC_ICU_data'
            signal_list = ["Pulse", "HR", "SpO2", "etCO2", "NBPm", "NBPd", "NBPs", "RR", "CVPm", "awRR"]
            pre_positive_window = int(2*(60*60/5))
            num_pre_positive_encodings = int(pre_positive_window/window_size)

            TEST_mixed_data_maps = torch.from_numpy(np.load(os.path.join(path, 'test_mixed_data_maps.npy')))
            TEST_mixed_labels = torch.from_numpy(np.load(os.path.join(path, 'test_mixed_labels.npy')))

            train_mixed_data_maps = torch.from_numpy(np.load(os.path.join(path, 'train_mixed_data_maps.npy')))
            train_mixed_labels = torch.from_numpy(np.load(os.path.join(path, 'train_mixed_labels.npy'))) 

            x_test = TEST_mixed_data_maps
            y_test = TEST_mixed_labels[:, -1] # Just grab last label, that's all we need for classifier
            x_train = train_mixed_data_maps
            y_train = train_mixed_labels[:, -1]
            # Create model
            classifier = LinearClassifier(input_size=32)
            rnn = RnnPredictor(16, 32)
            encoder = CausalCNNEncoder(in_channels=20, channels=8, depth=2, reduced_size=60, encoding_size=16, kernel_size=3, device=device, window_size=window_size)

        inds = list(range(len(x_train)))
        random.shuffle(inds)
        x_train = x_train[inds]
        y_train = y_train[inds]
        n_train = int(0.8 * len(inds))

        trainset = torch.utils.data.TensorDataset(torch.Tensor(x_train[:n_train]), torch.Tensor(y_train[:n_train]))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        validset = torch.utils.data.TensorDataset(torch.Tensor(x_train[n_train:]), torch.Tensor(y_train[n_train:]))
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=True)
        testset = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)


        params = list(classifier.parameters()) + list(encoder.parameters()) + list(rnn.parameters())
        optimizer = torch.optim.Adam(params, lr=0.0005)
        train_loss_trend, valid_loss_trend = [], []
        for epoch in range(1, 101):
            train_epoch_losses, train_pred_all, train_y_all = epoch_run(train_loader, classifier, encoder, rnn, window_size=window_size, optimizer=optimizer, train=True, num_pre_positive_encodings=num_pre_positive_encodings)
            valid_epoch_losses, valid_pred_all, valid_y_all = epoch_run(valid_loader, classifier, encoder, rnn, window_size=window_size, optimizer=optimizer, train=False, num_pre_positive_encodings=num_pre_positive_encodings)
            
            train_loss = np.mean(train_epoch_losses)
            valid_loss = np.mean(valid_epoch_losses)

            train_auc = roc_auc_score(np.concatenate(train_y_all, axis=0), np.concatenate(train_pred_all, axis=0))
            train_auprc = average_precision_score(np.concatenate(train_y_all, axis=0), np.concatenate(train_pred_all, axis=0))

            valid_auc = roc_auc_score(np.concatenate(valid_y_all, axis=0), np.concatenate(valid_pred_all, axis=0))
            valid_auprc = average_precision_score(np.concatenate(valid_y_all, axis=0), np.concatenate(valid_pred_all, axis=0))
            
            
            train_loss_trend.append(train_loss)
            valid_loss_trend.append(valid_loss)

            if epoch%10==0:
                print('***** Epoch %d *****' % epoch)
                print('Training Loss: %.3f \t Training AUROC: %.3f  \t Training AUPRC: %.3f'
                    '\t Valid Loss: %.3f \t Valid AUROC: %.3f \t Valid AUPRC: %.3f' % (train_loss, train_auc, train_auprc, valid_loss, valid_auc, valid_auprc))
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'rnn_state_dict':rnn.state_dict(), 
                    'loss': valid_loss,
                    'auc': valid_auc
                }
                if not os.path.exists('../ckpt/baselines/'):
                    os.mkdir('../ckpt/baselines/')
                torch.save(state, '../ckpt/baselines/e2e%s_%d.pth.tar'%(data_type, cv))

        # Plot loss
        plt.figure()
        plt.plot(train_loss_trend, label='Train loss')
        plt.plot(valid_loss_trend, label='Validation loss')
        plt.legend()
        plt.savefig('../DONTCOMMITplots/%s_e2e/e2e%s_%d.pdf'%(data_type, data_type, cv))

        test_epoch_losses, test_pred_all, test_y_all = epoch_run(test_loader, classifier, encoder, rnn, window_size=window_size, optimizer=optimizer, train=False, num_pre_positive_encodings=num_pre_positive_encodings)
        
        test_auc = roc_auc_score(np.concatenate(test_y_all, axis=0), np.concatenate(test_pred_all, axis=0))
        test_auprc = average_precision_score(np.concatenate(test_y_all, axis=0), np.concatenate(test_pred_all, axis=0))

        test_loss = np.mean(test_epoch_losses)
        overal_loss.append(test_loss)
        overal_auc.append(test_auc)
        overal_auprc.append(test_auprc)

        print()
    print('Final Test performance:\t Loss: %.3f +- %.3f \t Test AUROC: %.3f +- %.3f \t Test AUPRC: %.3f +- %.3f' %
        (np.mean(overal_loss), np.std(overal_loss), np.mean(overal_auc), np.std(overal_auc), np.mean(overal_auprc),
         np.std(overal_auprc)))


def epoch_run(data_loader, classifier, encoder, rnn, window_size, optimizer=None, train=False, num_pre_positive_encodings=None):
    if train:
        encoder.train()
        classifier.train()
        rnn.train()
    else:
        encoder.eval()
        classifier.eval()
        rnn.eval()

    encoder = encoder.to(device)
    classifier = classifier.to(device)
    rnn = rnn.to(device)

    epoch_losses = []
    pred_all, y_all = [], []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        # encoding_batch is of shape (batch_size, seq_len/window_size, encoding_size)
        encoding_batch, encoding_mask = encoder.forward_seq(x, return_encoding_mask=True) # encoding_mask is of shape (batch_size, seq_len/window_size)
        encoding_batch = encoding_batch.to(device)
        encoding_mask = encoding_mask.to(device)
        
        
        # encoding_batch and y are of size (batch_size, seq_len/window_size, encoding_size) and (batch_size)
        rnn_window_size = int(torch.randint(low=1, high=4, size=(1,))) # generates a number between 1 and 3 inclusive. This is the number of encodings the rnn will be fed at a time
        positive_inds = torch.where(y==1)
        positive_encodings = encoding_batch[positive_inds] # of shape (num_pos_in_batch, seq_len/window_size, encoding_size)
        positive_encodings_mask = encoding_mask[positive_inds] # of shape (num_pos_in_batch, seq_len/window_size)
        
        negative_inds = torch.where(y==0)
        negative_encodings = encoding_batch[negative_inds] # of shape (num_neg_in_batch, seq_len/window_size, encoding_size)
        negative_encodings_mask = encoding_mask[negative_inds] # of shape (num_neg_in_batch, seq_len/window_size)

        positive_encodings = positive_encodings[:, -num_pre_positive_encodings:, :] # now of shape (num_pos_in_batch, num_pre_positive_encodings, encoding_size)
        positive_encodings_mask = positive_encodings_mask[:, -num_pre_positive_encodings:] # now of shape (num_pos_in_batch, num_pre_positive_encodings)
        

        positive_encodings = positive_encodings[:, -(positive_encodings.shape[1]//rnn_window_size)*rnn_window_size:, :] # Clips each sample on the left side so the number of encodings is divisible by rnn_window_size
        positive_encodings_mask = positive_encodings_mask[:, -(positive_encodings_mask.shape[1]//rnn_window_size)*rnn_window_size:]
        negative_encodings = negative_encodings[:, -(negative_encodings.shape[1]//rnn_window_size)*rnn_window_size:, :]
        negative_encodings_mask = negative_encodings_mask[:, -(negative_encodings_mask.shape[1]//rnn_window_size)*rnn_window_size:]


        negative_encodings = negative_encodings.reshape(-1, rnn_window_size, negative_encodings.shape[-1]) # Now of shape (num_neg_rnn_window_sizes_over_all_encodings, rnn_window_size, encoding_size)
        positive_encodings = positive_encodings.reshape(-1, rnn_window_size, positive_encodings.shape[-1]) # Now of shape (num_pos_rnn_window_sizes_over_all_encodings, rnn_window_size, encoding_size)
        negative_encodings_mask = negative_encodings_mask.reshape(-1, rnn_window_size) # Now of shape (num_neg_rnn_window_sizes_over_all_encodings, rnn_window_size)
        positive_encodings_mask = positive_encodings_mask.reshape(-1, rnn_window_size) # Now of shape (num_pos_rnn_window_sizes_over_all_encodings, rnn_window_size)

        # Now we'll do the mode of each rnn window of encodings. So for each sequence of encodings, we'll take the mode value of the mask
        negative_encodings_mask = torch.mode(negative_encodings_mask, dim=1)[0] # of shape (num_neg_rnn_window_sizes_over_all_encodings,)
        positive_encodings_mask = torch.mode(positive_encodings_mask, dim=1)[0] # of shape (num_pos_rnn_window_sizes_over_all_encodings,)

        neg_window_labels = torch.zeros(negative_encodings.shape[0], 1)
        pos_window_labels = torch.ones(positive_encodings.shape[0], 1)
        

        window_samples = torch.vstack([positive_encodings, negative_encodings]) # of shape (num_rnn_window_sizes_over_all_encodings, rnn_window_size, encoding_size)
        window_labels = torch.cat([pos_window_labels, neg_window_labels]) # of shape (num_rnn_window_sizes_over_all_encodings, 1)
        window_masks = torch.cat([positive_encodings_mask, negative_encodings_mask]) # of shape (num_rnn_window_sizes_over_all_encodings,)

        # Remove all sequences of encodings that were at least partly derived from fully imputed data.
        window_samples = window_samples[torch.where(window_masks!=-1)]
        window_labels = window_labels[torch.where(window_masks!=-1)]


        '''
        print("SHUFFLING DATA BEFORE FEEDING INTO RNN!")
        # Shuffling before feeding to RNN
        inds = np.arange(len(window_samples))
        random.shuffle(inds)
        window_samples = window_samples[inds]
        window_labels = window_labels[inds]
        window_masks = window_masks[inds]
        '''

        _, hidden_and_cell = rnn(window_samples)
        hidden_state = hidden_and_cell[0] # hidden state is of shape (1, batch_size, hidden_size). Contains the last hidden state for each sample in the batch
        hidden_state = torch.squeeze(hidden_state)
        
        predictions = torch.squeeze(classifier(hidden_state))
        # Shape of predictions and window_labels is (batch_size)
        window_labels = torch.squeeze(window_labels).to(device)
        if train:
            # Ratio of num negative examples divided by num positive examples is pos_weight
            pos_weight = torch.Tensor([negative_encodings.shape[0] / max(positive_encodings.shape[0], 1)]).to(device)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Applies sigmoid to outputs passed in so we shouldn't have sigmoid in the model. 
            loss = loss_fn(predictions, window_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(predictions, window_labels)
        
        epoch_loss = loss.item()

        # Apply sigmoid to predictions since we didn't apply it for the loss function since the loss function does sigmoid on its own.
        predictions = torch.nn.Sigmoid()(predictions)

        # Move Tensors to CPU and remove gradients so they can be converted to NumPy arrays in the sklearn functions
        window_labels = window_labels.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        encoding_batch = encoding_batch.cpu() # Move off GPU memory
        neg_window_labels = neg_window_labels.cpu()
        pos_window_labels = pos_window_labels.cpu()
        y = y.cpu()

        epoch_losses.append(epoch_loss)
        pred_all.append(predictions)
        y_all.append(window_labels)

    return epoch_losses, pred_all, y_all








    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run End2End supervised model')
    parser.add_argument('--data', type=str, default='ICU')
    parser.add_argument('--cv', type=int, default=3)
    args = parser.parse_args()

    main(args.data, args.cv)
