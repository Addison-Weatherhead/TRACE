import torch
import numpy as np
import os
import pickle as pkl
import random
import argparse

#from tnc.models import LinearClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

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

    def forward(self, x, time_last=True, past=None):
        if time_last:
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
            with open(os.path.join(train_path, 'y.pkl'), 'rb') as f:
                y_train = pkl.load(f)

            with open(os.path.join(test_path, 'x.pkl'), 'rb') as f:
                x_test = pkl.load(f)     # shape: [n_batch, mask, 28(n_features), time steps]
            with open(os.path.join(test_path, 'y.pkl'), 'rb') as f:
                y_test = pkl.load(f)

            inds = list(range(len(x_train)))
            random.shuffle(inds)
            x_train = x_train[inds]
            y_train = y_train[inds]
            n_train = int(0.8*len(inds))

            trainset = torch.utils.data.TensorDataset(torch.Tensor(x_train[:n_train]), torch.Tensor(y_train[:n_train]))
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True)
            validset = torch.utils.data.TensorDataset(torch.Tensor(x_train[n_train:]), torch.Tensor(y_train[n_train:]))
            valid_loader = torch.utils.data.DataLoader(validset, batch_size=20, shuffle=True)
            testset = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
            test_loader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=True)

            # Create model
            classifier = LinearClassifier(input_size=32)
            encoder_rnn = RnnPredictor(28, 32)
            encoder = GRUDEncoder()## Fill this in the same size as the tnc-icu one

        params = list(classifier.parameters()) + list(encoder.parameters()+list(encoder_rnn.parameters()))
        optimizer = torch.optim.Adam(params, lr=0.001)
        for epoch in range(100):
            train_loss, train_auc, train_auprc = epoch_run(train_loader, classifier, encoder, optimizer=optimizer, train=True)
            valid_loss, valid_auc, valid_auprc = epoch_run(valid_loader, classifier, encoder, optimizer=optimizer, train=False)
            
            if epoch%5==0:
                print('***** Epoch %d *****' % epoch)
                print('Training Loss: %.3f \t Training AUROC: %.3f  \t Training AUROC: %.3f'
                    '\t Valid Loss: %.3f \t Valid AUROC: %.3f \t Valid AUROC: %.3f' % (train_loss, train_auc, train_auprc, valid_loss, valid_auc, valid_auprc))
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'encoder_rnn_state_dict': encoder_rnn.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'loss': valid_loss,
                    'auc': valid_auc
                }
                if not os.path.exists('./ckpt/baselines/'):
                    os.mkdir('./ckpt/baselines/')
                torch.save(state, './ckpt/baselines/raw_%s_%d.pth.tar'%(data_type, cv))

        # Plot loss
        plt.plot(train_loss_all, label='Train loss')
        plt.plot(valid_train_loss, label='Validation loss')
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
          (np.mean(overal_loss), np.std(overal_loss), np.mean(overal_auc), np.std(overal_auc, np.mean(overal_auprc), np.std(overal_auprc))))


def epoch_run(data_loader, classifier, encoder, encoder_rnn, optimizer=None, train=False):
    if train:
        encoder.train()
        classifier.train()
        encoder_rnn.train()
    else:
        encoder.eval()
        classifier.eval()
        encoder_rnn.eval()

    # encoder.to(device)
    # classifier.to(device)

    epoch_loss = 0
    epoch_auc = 0
    pred_all, y_all = [], []
    for x,y in data_loader:
        if torch.sum(torch.isnan(x))>1:
            continue
        encodings - encoder_rnn(x)
        encodings, _ = encoder(encodings)
        logits = classifier(encodings).squeeze()
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
