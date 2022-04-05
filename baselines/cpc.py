import torch
import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import argparse

from tnc.models import Chomp1d, SqueezeChannels, CausalConvolutionBlock, CausalCNN
from tnc.utils import plot_distribution, model_distribution
from baselines.triplet_loss import CausalCNNEncoder, train_linear_classifier, linear_classifier_epoch_run 
#from tnc.tnc import linear_classifier_epoch_run
#from tnc.evaluations import ClassificationPerformanceExperiment, WFClassificationExperiment
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def epoch_run(data, ds_estimator, auto_regressor, encoder, device, window_size, n_size=5, optimizer=None, train=True):
    if train:
        encoder.train()
        ds_estimator.train()
        auto_regressor.train()
    else:
        encoder.eval()
        ds_estimator.eval()
        auto_regressor.eval()
    encoder.to(device)
    ds_estimator.to(device)
    auto_regressor.to(device)

    epoch_loss = 0
    acc = 0
    for n_i, sample in enumerate(data):
        # Recall each sample is of shape (2, num_features, signal_length)
        rnd_t = np.random.randint(5*window_size, sample.shape[-1]-5*window_size) # Choose random time in the timeseries
        sample = torch.Tensor(sample[..., max(0,(rnd_t-20*window_size)):min(sample.shape[-1], rnd_t+20*window_size)]) # sample is now redefined as being of length 40*window_size centered at rnd_t

        T = sample.shape[-1]
        windowed_sample = np.split(sample[..., :(T // window_size) * window_size], (T // window_size), -1) # splits the sample into window_size pieces
    
        windowed_sample = torch.tensor(np.stack(windowed_sample, 0), device=device) # of shape (num_samples, num_features, window_size) where num_samples = ~40
        encodings = encoder(windowed_sample) # of shape (num_samples, encoding_size)
        
        window_ind = torch.randint(2,len(encodings)-2, size=(1,)) # window_ind is the last window we'll have access to in the AR model. After that, we want to predict the future
        _, c_t = auto_regressor(encodings[max(0, window_ind[0]-10):window_ind[0]+1].unsqueeze(0)) # Feeds the last 10 encodings preceding and including the window_ind windowed sample into the AR model.
        density_ratios = torch.bmm(encodings.unsqueeze(1),
                                       ds_estimator(c_t.squeeze(1).squeeze(0)).expand_as(encodings).unsqueeze(-1)).view(-1,) # Just take the dot product of the encodings and the output of the estimator?
        r = set(range(0, window_ind[0] - 2))
        r.update(set(range(window_ind[0] + 3, len(encodings))))
        rnd_n = np.random.choice(list(r), n_size)
        X_N = torch.cat([density_ratios[rnd_n], density_ratios[window_ind[0] + 1].unsqueeze(0)], 0)
        if torch.argmax(X_N)==len(X_N)-1:
            acc += 1
        labels = torch.Tensor([len(X_N)-1]).to(device)
        loss = torch.nn.CrossEntropyLoss()(X_N.view(1, -1), labels.long())
        epoch_loss += loss.item()
        if n_i%20==0:
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return epoch_loss / len(data), acc/(len(data))


def learn_encoder(x, window_size, encoding_size, lr=0.001, decay=0, n_size=5, n_epochs=50, data='simulation', device='cpu', n_cross_val=1):
    if not os.path.exists("./DONTCOMMITplots/%s_cpc/"%data):
        os.mkdir("./DONTCOMMITplots/%s_cpc/"%data)
    if not os.path.exists("./ckpt/%s_cpc/"%data):
        os.mkdir("./ckpt/%s_cpc/"%data)
    accuracies = []
    for cv in range(n_cross_val):
        if data == 'ICU':
            #encoding_size = 16
            encoder = CausalCNNEncoder(in_channels=10, channels=8, depth=2, reduced_size=30, encoding_size=encoding_size, kernel_size=3, window_size=window_size, device=device)
        elif 'HiRID' in data:
            #encoding_size = 10
            encoder = CausalCNNEncoder(in_channels=18, channels=4, depth=1, reduced_size=2, encoding_size=encoding_size, kernel_size=6, window_size=window_size, device=device)
        
        ds_estimator = torch.nn.Linear(encoder.encoding_size, encoder.encoding_size) # Predicts future latent data given context vector
        auto_regressor = torch.nn.GRU(input_size=encoding_size, hidden_size=encoding_size, batch_first=True)
        params = list(ds_estimator.parameters()) + list(encoder.parameters()) + list(auto_regressor.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
        inds = list(range(len(x)))
        random.shuffle(inds)
        x = x[inds]
        n_train = int(0.8*len(x))
        best_acc = 0
        best_loss = np.inf
        train_loss, test_loss = [], []
        # Recall x is of shape (num_samples, 2, num_features, signal_length)
        for epoch in range(n_epochs):
            epoch_loss, acc = epoch_run(x[:n_train], ds_estimator, auto_regressor, encoder, device, window_size, optimizer=optimizer,
                                        n_size=n_size, train=True)
            epoch_loss_test, acc_test = epoch_run(x[n_train:], ds_estimator, auto_regressor, encoder, device, window_size, n_size=n_size, train=False)
            if epoch%20==0:
                print('\nEpoch ', epoch)
                print('Train ===> Loss: ', epoch_loss, '\t Accuracy: ', acc)
                print('Test ===> Loss: ', epoch_loss_test, '\t Accuracy: ', acc_test)
            train_loss.append(epoch_loss)
            test_loss.append(epoch_loss_test)
            if epoch_loss_test<best_loss:
                print('Save new ckpt')
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict()
                }
                best_loss = epoch_loss_test
                best_acc = acc_test
                torch.save(state, './ckpt/%s_cpc/checkpoint_%d.pth.tar' %(data, cv))
        accuracies.append(best_acc)
        plt.figure()
        plt.plot(np.arange(n_epochs), train_loss, label="Train")
        plt.plot(np.arange(n_epochs), test_loss, label="Test")
        plt.title("CPC Loss")
        plt.legend()
        plt.savefig(os.path.join("./DONTCOMMITplots/%s_cpc/loss_%d.pdf"%(data, cv)))
    print('=======> Performance Summary:')
    print('Accuracy: %.2f +- %.2f' % (100 * np.mean(accuracies), 100 * np.std(accuracies)))


def main(is_train, data_type, lr,  cv):
    if not os.path.exists("./DONTCOMMITplots"):
        os.mkdir("./DONTCOMMITplots")
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")

    if data_type == 'ICU':
        length_of_hour = int(60*60/5)
        window_size = 60
        encoding_size = 6#16
        n_epochs = 400
        lr = 1e-3
        pos_sample_name = 'arrest'
        path = '/datasets/sickkids/TNC_ICU_data/'
        signal_list = ["Pulse", "HR", "SpO2", "etCO2", "NBPm", "NBPd", "NBPs", "RR", "CVPm", "awRR"]
        num_pre_positive_encodings = 12 # 6 per hr, last 2 hrs is the pre arrest window

        # NOTE THE MAP CHANNEL HAS 1'S FOR OBSERVED VALUES, 0'S FOR MISSING VALUES
        # data_maps arrays are of shape (num_samples, 2, 10, 5040).
        # 5040 is 7 hrs, 10 features, and there are 2 channels. Channel 0 is data, channel 1 is map.
            
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
        n_epochs = 300
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
        (unique, counts) = np.unique(TEST_Apache_Groups, return_counts=True)

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

        checkpoint = torch.load('ckpt/%s_cpc/checkpoint_0.pth.tar'%(data_type))
        if data_type == 'ICU':
            encoder = CausalCNNEncoder(in_channels=10, channels=8, depth=2, reduced_size=30, encoding_size=encoding_size, kernel_size=3, window_size=window_size, device=device)
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
        #classifier_train_labels = torch.Tensor([1 in label for label in train_mixed_labels_cv]) # Sets labels for positive samples to 1
        #classifier_validation_labels = torch.Tensor([1 in label for label in validation_mixed_labels_cv]) # Sets labels for positive samples to 1
        #classifier_TEST_labels = torch.Tensor([1 in label for label in TEST_mixed_labels]) # Sets labels for positive samples to 1


        #train_mixed_data_maps_cv = train_mixed_data_maps_cv[:, 0, :, :] # Only keep data, not mask

        #validation_mixed_data_maps_cv = validation_mixed_data_maps_cv[:, 0, :, :] # Only keep data, not mask

        #TEST_mixed_data_maps = TEST_mixed_data_maps[:, 0, :, :] # Only keep data, not mask
        classifier, valid_auroc, valid_auprc, TEST_auroc, TEST_auprc = train_linear_classifier(X_train=train_mixed_data_maps_cv[:, 0, :, :], y_train=train_mixed_labels_cv,
                X_validation=validation_mixed_data_maps_cv[:, 0, :, :], y_validation=validation_mixed_labels_cv,
                X_TEST=TEST_mixed_data_maps[:, 0, :, :], y_TEST=TEST_mixed_labels, exp_type="cpc", window_size=window_size,
                encoding_size=encoding_size, batch_size=20, num_pre_positive_encodings=num_pre_positive_encodings,
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
    parser = argparse.ArgumentParser(description='Run CPC')
    parser.add_argument('--data', type=str, default='ICU')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    
    
    main(args.train, args.data, args.lr, args.cv)

