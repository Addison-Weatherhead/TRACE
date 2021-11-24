import torch
import numpy as np
import os
import pickle as pkl
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#from tnc.models import LinearClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LinearClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super(LinearClassifier, self).__init__()
        self.input_size = input_size

        self.classifier =torch.nn.Sequential(torch.nn.Linear(self.input_size, 8),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.3),
                                        torch.nn.Linear(8, 1),
                                        torch.nn.Dropout(p=0.3))
        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        torch.nn.init.xavier_uniform_(self.classifier[3].weight)

    def forward(self, x):
        logits = self.classifier(x)
        return logits


class RnnPredictor(torch.nn.Module):
    def __init__(self, encoding_size, hidden_size):
        super(RnnPredictor, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=encoding_size,  hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, x, past=None):
        x = x.permute(0,2,1)        # Feature should be the last dimension
        if past is None:
            output, (h_n, c_n) = self.rnn(x)
        else:
            output, (h_n, c_n) = self.rnn(x, past)
        return output, h_n


def main(data_type, n_cv):
    overal_loss, overal_auc, overal_auprc = [], [], []
    for cv in range(n_cv):
        np.random.seed(cv*9)
        if data_type=='mimic':
            train_path = './data/mimic/train'
            test_path = './data/mimic/test'
            lab_IDs = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                       'HEMATOCRIT', 'HEMOGLOBIN',
                       'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN',
                       'WBC']
            vital_IDs = ['HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2', 'Glucose', 'Temp']

            # Load processed MIMIC
            with open(os.path.join(train_path, 'x.pkl'), 'rb') as f:
                x_train = pkl.load(f)     # shape: [n_batch, mask, 28(n_features), time steps]
                x_train = x_train[:,0,:,:]      # Mask channel is not needed for this baseline
            with open(os.path.join(train_path, 'y.pkl'), 'rb') as f:
                y_train = pkl.load(f)

            with open(os.path.join(test_path, 'x.pkl'), 'rb') as f:
                x_test = pkl.load(f)     # shape: [n_batch, mask, 28(n_features), time steps]
                x_test = x_test[:,0,:,:]        # Mask channel is not needed for this baseline
            with open(os.path.join(test_path, 'y.pkl'), 'rb') as f:
                y_test = pkl.load(f)
            # Create model
            classifier = LinearClassifier(input_size=32)
            encoder = RnnPredictor(28, 32)

        if data_type == 'icu':
            path = 'DONTCOMMITdata/ICU'#'/dataset/sickkids/ca_data/processed_data'
            signal_list = ['Pulse', 'HR', 'SpO2', 'etCO2', 'NBPm', 'NBPd', 'NBPs', 'RR']

            test_normal_data_maps = torch.Tensor(np.load(os.path.join(path, 'test_normal_data_maps.npy')))
            test_normal_labels = torch.Tensor(np.load(os.path.join(path, 'test_normal_labels.npy')))
            train_normal_data_maps = torch.Tensor(np.load(os.path.join(path, 'train_normal_data_maps.npy')))
            train_normal_labels = torch.Tensor(np.load(os.path.join(path, 'train_normal_labels.npy')))

            test_ca_data_maps = torch.Tensor(np.load(os.path.join(path, 'test_ca_data_maps.npy')))
            test_ca_labels = torch.Tensor(np.load(os.path.join(path, 'test_ca_labels.npy')))
            train_ca_data_maps = torch.Tensor(np.load(os.path.join(path, 'train_ca_data_maps.npy')))
            train_ca_labels = torch.Tensor(np.load(os.path.join(path, 'train_ca_labels.npy')))

            x_test = torch.vstack([test_normal_data_maps, test_ca_data_maps])[:,0,:,-5*60*12:].squeeze() # Take only the last 5 hours
            y_test = torch.vstack([test_normal_labels, test_ca_labels])[:,-1]
            x_train = torch.vstack([train_normal_data_maps, train_ca_data_maps])[:,0,:,-5*60*12:].squeeze() # Take only the last 5 hours
            y_train = torch.vstack([train_normal_labels, train_ca_labels])[:,-1]
            # Create model
            classifier = LinearClassifier(input_size=32)
            encoder = RnnPredictor(8, 32)

        inds = list(range(len(x_train)))
        random.shuffle(inds)
        x_train = x_train[inds]
        y_train = y_train[inds]
        n_train = int(0.8 * len(inds))

        trainset = torch.utils.data.TensorDataset(torch.Tensor(x_train[:n_train]), torch.Tensor(y_train[:n_train]))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True)
        validset = torch.utils.data.TensorDataset(torch.Tensor(x_train[n_train:]), torch.Tensor(y_train[n_train:]))
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=20, shuffle=True)
        testset = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        test_loader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=True)


        params = list(classifier.parameters()) + list(encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=0.0001)
        train_loss_trend, valid_loss_trend = [], []
        for epoch in range(101):
            train_loss, train_auc, train_auprc = epoch_run(train_loader, classifier, encoder, optimizer=optimizer, train=True)
            valid_loss, valid_auc, valid_auprc = epoch_run(valid_loader, classifier, encoder, optimizer=optimizer, train=False)
            train_loss_trend.append(train_loss)
            valid_loss_trend.append(valid_loss)

            if epoch%10==0:
                print('***** Epoch %d *****' % epoch)
                print('Training Loss: %.3f \t Training AUROC: %.3f  \t Training AUROC: %.3f'
                    '\t Valid Loss: %.3f \t Valid AUROC: %.3f \t Valid AUROC: %.3f' % (train_loss, train_auc, train_auprc, valid_loss, valid_auc, valid_auprc))
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'loss': valid_loss,
                    'auc': valid_auc
                }
                if not os.path.exists('./ckpt/baselines/'):
                    os.mkdir('./ckpt/baselines/')
                torch.save(state, './ckpt/baselines/raw_%s_%d.pth.tar'%(data_type, cv))

        # Plot loss
        plt.plot(train_loss_trend, label='Train loss')
        plt.plot(valid_loss_trend, label='Validation loss')
        plt.legend()
        plt.savefig('./plots/raw_%s_%d.pdf'%(data_type, cv))
        test_loss, test_auc, test_auprc = epoch_run(test_loader, classifier, encoder, optimizer=optimizer, train=False)
        overal_loss.append(test_loss)
        overal_auc.append(test_auc)
        overal_auprc.append(test_auprc)

        test_loss, test_auc, test_auprc = epoch_run(test_loader, classifier, encoder, optimizer=optimizer, train=False)
        overal_loss.append(test_loss)
        overal_auc.append(test_auc)
        overal_auprc.append(test_auprc)
    print('Final Test performance: \t Loss: %.3f +- %.3f \t Training AUROC: %.3f +- %.3f \t Training AUPRC: %.3f +- %.3f'%
          (np.mean(overal_loss), np.std(overal_loss), np.mean(overal_auc), np.std(overal_auc), np.mean(overal_auprc), np.std(overal_auprc)))


def epoch_run(data_loader, classifier, encoder, optimizer=None, train=False):
    if train:
        encoder.train()
        classifier.train()
    else:
        encoder.eval()
        classifier.eval()

    encoder.to(device)
    classifier.to(device)

    epoch_loss = 0
    epoch_auc = 0
    pred_all, y_all = [], []
    for x,y in data_loader:
        x.to(device)
        y.to(device)
        if torch.sum(torch.isnan(x))>1:
            continue
        encodings, _ = encoder(x)
        logits = classifier(encodings[:,-1,:]).squeeze()
        predictions = torch.nn.Sigmoid()(logits)
        loss = torch.nn.BCEWithLogitsLoss()(logits, y)
        epoch_loss += loss.item()
        y_all.append(y.detach().numpy())
        pred_all.append(predictions.detach().numpy())
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch_auc = roc_auc_score(np.concatenate(y_all, axis=0), np.concatenate(pred_all, axis=0))
    epoch_auprc = average_precision_score(np.concatenate(y_all, axis=0), np.concatenate(pred_all, axis=0))
    return epoch_loss/len(data_loader), epoch_auc, epoch_auprc


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run TNC')
    parser.add_argument('--data', type=str, default='mimic')
    parser.add_argument('--cv', type=int, default=3)
    args = parser.parse_args()

    main(args.data, args.cv)
