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
from baselines.triplet_loss import train_linear_classifier, linear_classifier_epoch_run
#from tnc.tnc import linear_classifier_epoch_run
#from tnc.evaluations import ClassificationPerformanceExperiment, WFClassificationExperiment
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(data_type, lr,  cv):
    if not os.path.exists("./DONTCOMMITplots"):
        os.mkdir("./DONTCOMMITplots")
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")

    if data_type == 'ICU':
        length_of_hour = int(60*60/5)
        n_epochs = 300
        window_size=60
        lr = 1e-2
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

    elif data_type == 'HiRID':
        n_epochs = 300
        lr = 1e-3
        window_size = 8
        length_of_hour = int((60*60)/300) # 60 seconds * 60 / 300 (which is num seconds in 5 min)
        pos_sample_name = 'mortality'
        path = '../DONTCOMMITdata/hirid_numpy'
        signal_list = ['vm1', 'vm3', 'vm4', 'vm5', 'vm13', 'vm20', 'vm28', 'vm62', 'vm136', 'vm146', 'vm172', 'vm174', 'vm176', 'pm41', 'pm42', 'pm43', 'pm44', 'pm87']
        sliding_gap = 1
        pre_positive_window = int((24*60*60)/300) # 24 hrs
        num_pre_positive_encodings = int(pre_positive_window/window_size)

        # NOTE THE MAP CHANNEL HAS 1'S FOR OBSERVED VALUES, 0'S FOR MISSING VALUES
        # data_maps arrays are of shape (num_samples, 2, 18, 1152). 4 days of data per sample
        TEST_mixed_data_maps = torch.from_numpy(np.load(os.path.join(path, 'TEST_data_maps.npy'))).float()
        TEST_mixed_mortality_labels = torch.from_numpy(np.load(os.path.join(path, 'TEST_mortality_labels.npy'))).float()
        TEST_mixed_labels = TEST_mixed_mortality_labels
        TEST_PIDs = torch.from_numpy(np.load(os.path.join(path, 'TEST_PIDs.npy'))).float()
        TEST_Apache_Groups = torch.from_numpy(np.load(os.path.join(path, 'TEST_Apache_Groups.npy'))).float()


        train_mixed_data_maps = torch.from_numpy(np.load(os.path.join(path, 'train_data_maps.npy'))).float()
        train_mixed_mortality_labels = torch.from_numpy(np.load(os.path.join(path, 'train_mortality_labels.npy'))).float()
        train_mixed_labels = train_mixed_mortality_labels
        train_PIDs = torch.from_numpy(np.load(os.path.join(path, 'train_PIDs.npy'))).float()
        train_Apache_Groups = torch.from_numpy(np.load(os.path.join(path, 'train_Apache_Groups.npy'))).float()

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


    # ************************* Train Classifier **************************
    
    classifier_validation_aurocs = []
    classifier_validation_auprcs = []
    classifier_TEST_aurocs = []
    classifier_TEST_auprcs = []
    for encoder_cv in range(cv):
        seed_val = 111*encoder_cv+2
        random.seed(seed_val)
        encoder = None

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
        classifier_train_labels = torch.Tensor([1 in label for label in train_mixed_labels_cv]) # Sets labels for positive samples to 1
        classifier_validation_labels = torch.Tensor([1 in label for label in validation_mixed_labels_cv]) # Sets labels for positive samples to 1
        classifier_TEST_labels = torch.Tensor([1 in label for label in TEST_mixed_labels]) # Sets labels for positive samples to 1
        classifier, valid_auroc, valid_auprc, TEST_auroc, TEST_auprc = train_linear_classifier(X_train=train_mixed_data_maps_cv[:, 0, :, :], y_train=train_mixed_labels_cv,
                X_validation=validation_mixed_data_maps_cv[:, 0, :, :], y_validation=validation_mixed_labels_cv,
                X_TEST=TEST_mixed_data_maps[:, 0, :, :], y_TEST=TEST_mixed_labels, exp_type="raw", window_size=window_size,
                encoding_size=train_mixed_data_maps_cv.shape[-1], batch_size=20, num_pre_positive_encodings=num_pre_positive_encodings,
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
    parser = argparse.ArgumentParser(description='Run baseline on raw data')
    parser.add_argument('--data', type=str, default='ICU')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()


    main(args.data, args.lr, args.cv)
