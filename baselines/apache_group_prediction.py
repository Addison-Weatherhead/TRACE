import torch
import random
from tnc.models import RnnPredictor
from baselines.triplet_loss import CausalCNNEncoder
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
from sklearn.metrics import roc_curve
from tnc.models import CausalCNNEncoder
from tnc.utils import detect_incr_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def linear_classifier_epoch_run(dataset, train, classifier, class_weights, optimizer, data_type, window_size, encoder, encoding_size):
    if train:
        classifier.train()
    else:
        classifier.eval()
    
    epoch_losses = []
    epoch_predictions = []
    epoch_labels = []
    for data_batch, label_batch in dataset:
        if encoder is None:
            encoding_batch = data_batch.float().to(device).permute(0,2,1)
        else:
            encoding_batch = encoder.forward_seq(data_batch.float().to(device))
        predictions = torch.squeeze(classifier(encoding_batch)) # of shape (bs, n_classes)
        label_batch = label_batch.to(device)
        if train:
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights) # Applies softmax to outputs passed in so we shouldn't have softmax in the model. 
            loss = loss_fn(predictions, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(predictions, label_batch)
        
        epoch_loss = loss.item()

        # Apply sigmoid to predictions since we didn't apply it for the loss function since the loss function does sigmoid on its own.
        predictions = torch.nn.Softmax(dim=1)(predictions)

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

def train_linear_classifier(X_train, y_train, X_validation, y_validation, X_TEST, y_TEST, encoding_size, baseline_type, encoder, window_size, class_weights, target_names, batch_size=32, return_models=False, return_scores=False, data_type='ICU', classification_cv=0, encoder_cv=0, ckpt_path="../ckpt",  plt_path="../DONTCOMMITplots", classifier_name=""):
    '''
    Trains a classifier to predict positive events in samples. 
    X_train is of shape (num_train_samples, 2, num_features, seq_len)
    y_train is of shape (num_train_samples, seq_len)

    '''
    print("Training Linear Classifier", flush=True)
    if encoder is None:
        (unique, counts) = np.unique(y_train, return_counts=True)
        classifier = RnnPredictor(encoding_size=18, hidden_size=8, n_classes=len(unique)).to(device)
        weight_decay = 0.0005
        lr = 0.001
        n_epochs = 200
    elif baseline_type=='tnc-icu':
        (unique, counts) = np.unique(y_train, return_counts=True)
        classifier = RnnPredictor(encoding_size=6, hidden_size=8, n_classes=len(unique)).to(device)
        encoder.eval()
        weight_decay = 0.0001
        lr = 0.001
        n_epochs = 300
    elif baseline_type=='tnc':
        (unique, counts) = np.unique(y_train, return_counts=True)
        classifier = RnnPredictor(encoding_size=6, hidden_size=8, n_classes=len(unique)).to(device)
        encoder.eval()
        weight_decay = 0.0005
        lr = 0.001
        n_epochs = 250
    else:
        (unique, counts) = np.unique(y_train, return_counts=True)
        classifier = RnnPredictor(encoding_size=6, hidden_size=8, n_classes=len(unique)).to(device)
        encoder.eval()
        weight_decay = 0.0005
        lr = 0.001
        n_epochs = 200
    
    print('X_train shape: ', X_train.shape)
    print('batch_size: ', batch_size)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = torch.utils.data.TensorDataset(X_validation, y_validation)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    TEST_dataset = torch.utils.data.TensorDataset(X_TEST, y_TEST)
    TEST_data_loader = torch.utils.data.DataLoader(TEST_dataset, batch_size=batch_size, shuffle=True)
    
    params = list(classifier.parameters())
    lr = .001
    weight_decay = 0.0001
    print('Learning Rate for classifier training: ', lr)
    print('Weight Decay for classifier training: ', weight_decay)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    train_losses = []
    valid_losses = []
    
    for epoch in range(1, n_epochs+1):
        # linear_classifier_epoch_run(dataset, train, classifier, optimizer, data_type, window_size, encoder, encoding_size):
        epoch_train_predictions, epoch_train_losses, epoch_train_labels = linear_classifier_epoch_run(dataset=train_data_loader, train=True,
                                                    classifier=classifier, class_weights=class_weights,
                                                    optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=encoding_size)

        
        classifier.eval()
        epoch_validation_predictions, epoch_validation_losses, epoch_validation_labels = linear_classifier_epoch_run(dataset=validation_data_loader, train=False,
                                                    classifier=classifier, class_weights=class_weights,
                                                    optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=encoding_size)

        
        

        epoch_TEST_predictions, epoch_TEST_losses, epoch_TEST_labels = linear_classifier_epoch_run(dataset=TEST_data_loader, train=False,
                                                    classifier=classifier, class_weights=class_weights,
                                                    optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=encoding_size)

        
        # TRAIN 
        # Compute average over all batches in the epoch
        epoch_train_loss = np.mean(epoch_train_losses)
        epoch_train_predictions = torch.vstack(epoch_train_predictions) # of shape (num_samples, num_classes)
        epoch_train_labels = torch.cat(epoch_train_labels)


        epoch_train_auroc = roc_auc_score(epoch_train_labels, epoch_train_predictions, multi_class='ovr')
        # Compute precision recall curve
        #train_precision, train_recall, _ = precision_recall_curve(epoch_train_labels, epoch_train_predictions)
        # Compute AUPRC
        #epoch_train_auprc = auc(train_recall, train_precision) # precision is the y axis, recall is the x axis, computes AUC of this curve
        train_losses.append(epoch_train_loss)
        
        epoch_train_predictions_argmax = torch.argmax(epoch_train_predictions, dim=1) # now of shape (num_samples,)
        epoch_train_accuracy = len(torch.where(epoch_train_labels==epoch_train_predictions_argmax)[0])/epoch_train_labels.shape[0]
        # VALIDATION
        epoch_validation_loss = np.mean(epoch_validation_losses)
        epoch_validation_predictions = torch.vstack(epoch_validation_predictions) # of shape (num_samples, num_classes)
        epoch_validation_labels = torch.cat(epoch_validation_labels)

        epoch_validation_auroc = roc_auc_score(epoch_validation_labels, epoch_validation_predictions, multi_class='ovr')

        # Compute precision recall curve
        #valid_precision, valid_recall, _ = precision_recall_curve(epoch_validation_labels, epoch_validation_predictions)
        # Compute AUPRC
        #epoch_validation_auprc = auc(valid_recall, valid_precision) # precision is the y axis, recall is the x axis, computes AUC of this curve
        valid_losses.append(epoch_validation_loss)

        epoch_validation_predictions_argmax = torch.argmax(epoch_validation_predictions, dim=1)
        epoch_validation_accuracy = len(torch.where(epoch_validation_labels==epoch_validation_predictions_argmax)[0])/epoch_train_labels.shape[0]

        # TEST
        epoch_TEST_loss = np.mean(epoch_TEST_losses)
        epoch_TEST_predictions = torch.cat(epoch_TEST_predictions, 0)#torch.vstack(epoch_TEST_predictions) # of shape (num_samples, num_classes)
        epoch_TEST_labels = torch.cat(epoch_TEST_labels, 0)

        epoch_TEST_auroc = roc_auc_score(epoch_TEST_labels, epoch_TEST_predictions, multi_class='ovr')
        # Compute precision recall curve
        #TEST_precision, TEST_recall, _ = precision_recall_curve(epoch_TEST_labels, epoch_TEST_predictions)
        # Compute AUPRC
        #epoch_TEST_auprc = auc(TEST_recall, TEST_precision) # precision is the y axis, recall is the x axis, computes AUC of this curve
        
        epoch_TEST_predictions_argmax = torch.argmax(epoch_TEST_predictions, dim=1)
        epoch_TEST_accuracy = torch.sum(epoch_TEST_labels==epoch_TEST_predictions_argmax)/len(epoch_TEST_labels)#len(torch.where(epoch_TEST_labels==epoch_TEST_predictions_argmax)[0])/epoch_TEST_labels.shape[0]


        if epoch%10==0 or detect_incr_loss(valid_losses, 5):
            print('Epoch %d Classifier Loss =====> Training Loss: %.5f \t Training AUROC: %.5f \t Training Accuracy: %.5f \n\t Validation Loss: %.5f \t Validation AUROC: %.5f \t Validation Accuracy %.5f \n\t TEST Loss: %.5f \t TEST AUROC: %.5f \t TEST Accuracy %.5f'
                                % (epoch, epoch_train_loss, epoch_train_auroc, epoch_train_accuracy, epoch_validation_loss, epoch_validation_auroc, epoch_validation_accuracy, epoch_TEST_loss, epoch_TEST_auroc, epoch_TEST_accuracy))
            
            
            # Checkpointing the classifier model
            state = {
                    'epoch': epoch,
                    'classifier_state_dict': classifier.state_dict(),
                }

            torch.save(state, os.path.join(ckpt_path, classifier_name))
            if detect_incr_loss(valid_losses, 5):
                print('Validation loss increasing! Early stopping now.')
                break
   
    print("Train classification report: ")
    print('epoch_train_labels shape: ', epoch_train_labels.shape, 'epoch_train_predictions_argmax shape: ', epoch_train_predictions_argmax.shape)
    print(classification_report(epoch_train_labels.to('cpu'), epoch_train_predictions_argmax, target_names=target_names))
    print("Validation classification report: ")
    print(classification_report(epoch_validation_labels.to('cpu'), epoch_validation_predictions_argmax, target_names=target_names))
    print()
    print("TEST classification report: ")
    print(classification_report(epoch_TEST_labels.to('cpu'), epoch_TEST_predictions_argmax, target_names=target_names))
    print()
    
    

    # Plot Loss curves
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Validation")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join(plt_path, "%s_Classifier_loss_%d_%s.pdf"%(baseline_type, encoder_cv, classification_cv)))

    if return_models and return_scores:
        return (classifier, epoch_validation_auroc, epoch_validation_accuracy, epoch_TEST_auroc, epoch_TEST_accuracy)
    if return_models:
        return (classifier)
    if return_scores:
        return epoch_TEST_accuracy, epoch_TEST_auroc



if __name__ == '__main__':
    path = '../DONTCOMMITdata/hirid_numpy'
    train_first_24_hrs_data_maps = torch.from_numpy(np.load(os.path.join(path, 'train_first_24_hrs_data_maps.npy')))
    TEST_first_24_hrs_data_maps = torch.from_numpy(np.load(os.path.join(path, 'TEST_first_24_hrs_data_maps.npy')))

    train_first_24_hrs_PIDs = torch.from_numpy(np.load(os.path.join(path, 'train_first_24_hrs_PIDs.npy')))
    TEST_first_24_hrs_PIDs = torch.from_numpy(np.load(os.path.join(path, 'TEST_first_24_hrs_PIDs.npy')))

    train_Apache_Groups = torch.from_numpy(np.load(os.path.join(path, 'train_Apache_Groups.npy')))
    TEST_Apache_Groups = torch.from_numpy(np.load(os.path.join(path, 'TEST_Apache_Groups.npy')))

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

    print('train_first_24_hrs_data_maps shape: ', train_first_24_hrs_data_maps.shape)
    print('TEST_first_24_hrs_data_maps shape: ', TEST_first_24_hrs_data_maps.shape)
    print('train_first_24_hrs_PIDs shape: ', train_first_24_hrs_PIDs.shape)
    print('TEST_first_24_hrs_PIDs shape: ', TEST_first_24_hrs_PIDs.shape)
    print('train_Apache_Groups shape: ', train_Apache_Groups.shape)
    print('TEST_Apache_Groups shape: ', TEST_Apache_Groups.shape)


    # Note class 5 is missing from the TEST data, so well remove from training
    train_Apache_Groups[torch.where(train_Apache_Groups==5)] = -1
    train_first_24_hrs_data_maps = train_first_24_hrs_data_maps[torch.where(train_Apache_Groups != -1)]
    train_first_24_hrs_PIDs = train_first_24_hrs_PIDs[torch.where(train_Apache_Groups != -1)]
    train_Apache_Groups = train_Apache_Groups[torch.where(train_Apache_Groups != -1)]
    apache_names.pop(5)

    # Now to fix indexing so class numbers are still sequential
    for i in range(6, 20):
        train_Apache_Groups[torch.where(train_Apache_Groups==i)] = i-1
        TEST_Apache_Groups[torch.where(TEST_Apache_Groups==i)] = i-1


    (unique, counts) = np.unique(train_Apache_Groups, return_counts=True)
    too_few_samples = []
    for i in range(len(unique)):
        if counts[i] < 200:
            too_few_samples.append(i)
    classes_to_keep = [i for i in range(len(apache_names)) if i not in too_few_samples]

    data_maps_to_keep = []
    apache_groups_to_keep = []
    for sample, label in zip(train_first_24_hrs_data_maps, train_Apache_Groups):
        if int(label) in classes_to_keep:
            data_maps_to_keep.append(sample)
            apache_groups_to_keep.append(classes_to_keep.index(int(label)))

    train_first_24_hrs_data_maps = torch.stack(data_maps_to_keep)
    train_Apache_Groups = torch.Tensor(apache_groups_to_keep)
    train_Apache_Groups = train_Apache_Groups.to(torch.int64)

    data_maps_to_keep = []
    apache_groups_to_keep = []
    for sample, label in zip(TEST_first_24_hrs_data_maps, TEST_Apache_Groups):
        if int(label) in classes_to_keep:
            data_maps_to_keep.append(sample)
            apache_groups_to_keep.append(classes_to_keep.index(int(label)))

    TEST_first_24_hrs_data_maps = torch.stack(data_maps_to_keep)
    TEST_Apache_Groups = torch.Tensor(apache_groups_to_keep)
    TEST_Apache_Groups = TEST_Apache_Groups.to(torch.int64)
    apache_names = [apache_names[i] for i in range(len(apache_names)) if i not in too_few_samples]


    print('Updated Distributions:')
    class_weights = []
    (unique, counts) = np.unique(train_Apache_Groups, return_counts=True)
    print('Distribution of train Apache states:')
    for i in range(len(unique)):
        print(unique[i], ': ', apache_names[i], ': ', counts[i])
        class_weights.append(counts[i])
    print()
    class_weights = torch.Tensor(class_weights).to(device)
    max_weight = torch.max(class_weights)
    class_weights = max_weight/class_weights # So classes with high counts will have lower weights.
    (unique, counts) = np.unique(TEST_Apache_Groups, return_counts=True)
    print('Distribution of TEST Apache states:')
    for i in range(len(unique)):
        print(unique[i], ': ', apache_names[i], ': ', counts[i])


    for baseline_type in ['trip', 'cpc', 'raw']:
        np.random.seed(1234)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if baseline_type=='raw':
            encoder = None
        elif baseline_type=='tnc':
            checkpoint = torch.load('../ckpt/HiRID/0761_CausalCNNEncoder_HiRID_checkpoint_0.tar')
            encoder = CausalCNNEncoder(in_channels=18, channels=4, depth=1, reduced_size=2, encoding_size=6, kernel_size=(6 if baseline_type=='cpc' else 2), window_size=12, device=device) 
            encoder_cv = 0
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        elif baseline_type=='tnc-icu':
            checkpoint = torch.load('../ckpt/HiRID/0764_CausalCNNEncoder_HiRID_checkpoint_0.tar')
            encoder = CausalCNNEncoder(in_channels=36, channels=4, depth=1, reduced_size=2, encoding_size=10, kernel_size=2, window_size=12, device=device) 
            encoder_cv = 0
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            encoder.pruning_mask = checkpoint['pruning_mask']
            encoder.pruned_encoding_size = int(torch.sum(encoder.pruning_mask))
        else:
            checkpoint = torch.load('./ckpt/HiRID_%s/checkpoint_0.pth.tar'%baseline_type)
            encoder_cv = 0
            encoder = CausalCNNEncoder(in_channels=18, channels=4, depth=1, reduced_size=2, encoding_size=6, kernel_size=(6 if baseline_type=='cpc' else 2), window_size=12, device=device)
            #encoder = CausalCNNEncoder(in_channels=36, channels=4, depth=1, reduced_size=2, encoding_size=10, kernel_size=2, window_size=12, device=device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        all_acc, all_auroc = [], []
        for cv in range(3):
            inds = list(range(len(train_first_24_hrs_data_maps)))
            random.shuffle(inds)
            if baseline_type=='tnc-icu':
                train_first_24_hrs_data_maps_cv = train_first_24_hrs_data_maps[inds][..., :12*12]
                train_Apache_Groups_cv = train_Apache_Groups[inds]
                TEST_first_24_hrs_data_maps_cv = TEST_first_24_hrs_data_maps[..., :12*12]
                TEST_Apache_Groups_cv = TEST_Apache_Groups
            else:
                train_first_24_hrs_data_maps_cv = train_first_24_hrs_data_maps[inds][:,0,:, :12*12]
                train_Apache_Groups_cv = train_Apache_Groups[inds]
                TEST_first_24_hrs_data_maps_cv = TEST_first_24_hrs_data_maps[:,0,:,:12*12]
                TEST_Apache_Groups_cv = TEST_Apache_Groups
            n_train = int(0.8*len(inds))
            acc, auroc = train_linear_classifier(X_train=train_first_24_hrs_data_maps_cv[:n_train], y_train=train_Apache_Groups_cv[:n_train], 
                    X_validation=train_first_24_hrs_data_maps_cv[n_train:], y_validation=train_Apache_Groups_cv[n_train:], classification_cv=cv, 
                    X_TEST=TEST_first_24_hrs_data_maps_cv, y_TEST=TEST_Apache_Groups_cv, encoding_size=6, class_weights=class_weights,
                    encoder=encoder, window_size=12, target_names=apache_names, encoder_cv=encoder_cv, ckpt_path='./ckpt', baseline_type=baseline_type, 
                    return_scores=True, plt_path='./DONTCOMMITplots/HiRID_apache_classification', classifier_name='apache_classifier_%s'%baseline_type)
            all_acc.append(acc.item())
            all_auroc.append(auroc.item())
        print("*"*50)
        print("Final performance for %s baseline"%baseline_type)
        print("Accuracy: %.3f $\pm$ %.3f    AUROC: %.3f $\pm$ %.3f"%(np.mean(all_acc), np.std(all_acc), np.mean(all_auroc), np.std(all_auroc)))
        print()







