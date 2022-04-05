import torch
import numpy as np
from tnc.utils import plot_pca_trajectory_binned, plot_tsne_trajectory_binned, dim_reduction, plot_heatmap_subset_signals_with_risk, plot_pca_trajectory, detect_incr_loss
from tnc.models import CausalCNNEncoder, RnnPredictor
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
import matplotlib.pyplot as plt
import random
import argparse 


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
            loss = loss_fn(predictions, label_batch.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(predictions, label_batch.long())
        
        epoch_loss = loss.item()

        # Apply softmax to predictions since we didn't apply it for the loss function since the loss function does softmax on its own.
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

def train_linear_classifier(X_train, y_train, X_validation, y_validation, X_TEST, y_TEST, classifier_input_size, baseline_type, encoder, window_size, class_weights, target_names, batch_size=32, return_models=False, return_scores=False, data_type='ICU', classification_cv=0, encoder_cv=0, ckpt_path="../ckpt",  plt_path="../DONTCOMMITplots", classifier_name=""):
    '''
    Trains a classifier to predict positive events in samples. 
    X_train is of shape (num_train_samples, 2, num_features, seq_len)
    y_train is of shape (num_train_samples, seq_len)

    '''
    cv_aurocs = []
    cv_auprcs = []
    for cv in range(3):
        print("Training Linear Classifier", flush=True)
        if encoder is None:
            (unique, counts) = np.unique(y_train.detach().cpu(), return_counts=True)
            classifier = RnnPredictor(encoding_size=18, hidden_size=8, n_classes=len(unique)).to(device)
        else:
            (unique, counts) = np.unique(y_train.detach().cpu(), return_counts=True)
            classifier = RnnPredictor(encoding_size=classifier_input_size, hidden_size=8, n_classes=len(unique)).to(device)
            encoder.eval()
        
        print('X_train shape: ', X_train.shape)
        print('batch_size: ', batch_size)
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataset = torch.utils.data.TensorDataset(X_validation, y_validation)
        validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        TEST_dataset = torch.utils.data.TensorDataset(X_TEST, y_TEST)
        TEST_data_loader = torch.utils.data.DataLoader(TEST_dataset, batch_size=batch_size, shuffle=True)
        
        if baseline_type == 'e2e':
            params = list(classifier.parameters()) + list(encoder.parameters())
        else:
            params = list(classifier.parameters())
        lr = .001
        weight_decay = 0.0001
        print('Learning Rate for classifier training: ', lr)
        print('Weight Decay for classifier training: ', weight_decay)
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        train_losses = []
        valid_losses = []
        
        for epoch in range(1, 501):
            # linear_classifier_epoch_run(dataset, train, classifier, optimizer, data_type, window_size, encoder, encoding_size):
            epoch_train_predictions, epoch_train_losses, epoch_train_labels = linear_classifier_epoch_run(dataset=train_data_loader, train=True,
                                                        classifier=classifier, class_weights=class_weights,
                                                        optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=classifier_input_size)

            
            classifier.eval()
            epoch_validation_predictions, epoch_validation_losses, epoch_validation_labels = linear_classifier_epoch_run(dataset=validation_data_loader, train=False,
                                                        classifier=classifier, class_weights=class_weights,
                                                        optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=classifier_input_size)

            
            

            epoch_TEST_predictions, epoch_TEST_losses, epoch_TEST_labels = linear_classifier_epoch_run(dataset=TEST_data_loader, train=False,
                                                        classifier=classifier, class_weights=class_weights,
                                                        optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=classifier_input_size)

            
            # TRAIN 
            # Compute average over all batches in the epoch
            epoch_train_loss = np.mean(epoch_train_losses)
            epoch_train_predictions = torch.vstack(epoch_train_predictions) # of shape (num_samples, num_classes)
            epoch_train_labels = torch.cat(epoch_train_labels)


            epoch_train_auroc = roc_auc_score(epoch_train_labels, epoch_train_predictions[:, 1])
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

            epoch_validation_auroc = roc_auc_score(epoch_validation_labels, epoch_validation_predictions[:, 1])

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

            epoch_TEST_auroc = roc_auc_score(epoch_TEST_labels, epoch_TEST_predictions[:, 1])
            cv_aurocs.append(epoch_TEST_auroc)
            # Compute precision recall curve
            TEST_precision, TEST_recall, _ = precision_recall_curve(epoch_TEST_labels, epoch_TEST_predictions[:, 1])
            # Compute AUPRC
            epoch_TEST_auprc = auc(TEST_recall, TEST_precision) # precision is the y axis, recall is the x axis, computes AUC of this curve
            cv_auprcs.append(epoch_TEST_auprc)
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
        
        # Plot ROC and PR Curves
        '''
        plt.figure()
        plt.plot(train_recall, train_precision)
        plt.title('Train Precision Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(os.path.join(plt_path, '%s/%s/train_PR_curve_%d_%d.pdf'%(data_type, UNIQUE_ID, encoder_cv, classification_cv)))
        '''

        #plt.figure()
        #fpr, tpr, _ = roc_curve(epoch_train_labels, epoch_train_predictions)
        #plt.plot(fpr, tpr)
        #plt.xlabel('FPR')
        #plt.ylabel('TPR')
        #plt.title('Train ROC Curve')
        #plt.savefig(os.path.join(plt_path, 'train_ROC_curve_%d_%s.pdf'%(classification_cv, baseline_type)))
        
        
        '''
        plt.figure()
        plt.plot(valid_recall, valid_precision)
        plt.title('Validation Precision Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(os.path.join(plt_path, 'valid_PR_curve_%d_%s.pdf'%(classification_cv, baseline_type)))
        '''

        #plt.figure()
        #fpr, tpr, _ = roc_curve(epoch_validation_labels, epoch_validation_predictions)
        #plt.plot(fpr, tpr)
        #plt.xlabel('FPR')
        #plt.ylabel('TPR')
        #plt.title('Validation ROC Curve')
        #plt.savefig(os.path.join(plt_path, 'valid_ROC_curve_%d_%s.pdf'%(classification_cv, baseline_type)))

        '''
        plt.figure()
        plt.plot(TEST_recall, TEST_precision)
        plt.title('TEST Precision Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(os.path.join(plt_path, '%s/%s/TEST_PR_curve_%d_%d.pdf'%(data_type, UNIQUE_ID, encoder_cv, classification_cv)))
        '''

        #plt.figure()
        #fpr, tpr, _ = roc_curve(epoch_TEST_labels, epoch_TEST_predictions)
        #plt.plot(fpr, tpr)
        #plt.xlabel('FPR')
        #plt.ylabel('TPR')
        #plt.title('TEST ROC Curve')
        #plt.savefig(os.path.join(plt_path, 'TEST_ROC_curve_%d_%s.pdf'%(classification_cv, baseline_type)))

        # Plot Loss curves
        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(valid_losses, label="Validation")
        plt.title("Loss")
        plt.legend()
        plt.savefig(os.path.join(plt_path, "Classifier_loss_%d_%s.pdf"%(classification_cv, baseline_type)))

    print('Final TEST results:')
    print('AUC: ', np.mean(cv_aurocs), ' +- ', np.std(cv_aurocs))
    print('AUPRC: ', np.mean(cv_auprcs), ' +- ', np.std(cv_auprcs))

    if return_models and return_scores:
        return (classifier, epoch_validation_auroc, epoch_validation_accuracy, epoch_TEST_auroc, epoch_TEST_accuracy)
    if return_models:
        return (classifier)
    if return_scores:
        return epoch_TEST_accuracy, epoch_TEST_auroc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run circulatory failure prediction')
    parser.add_argument('--encoder_type', type=str, default=None)
    parser.add_argument('--checkpoint_file', type=str, default=None)
    args = parser.parse_args()

    encoder_type = args.encoder_type
    checkpoint_str = args.checkpoint_file
    print('Cutting off last 3 hrs of data')
    truncate_amt = 36
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '../DONTCOMMITdata/hirid_numpy'
    train_circulatory_data_maps = torch.from_numpy(np.load(os.path.join(data_path, 'train_circulatory_data_maps.npy')))[:, :, :, :-truncate_amt].float()
    TEST_circulatory_data_maps = torch.from_numpy(np.load(os.path.join(data_path, 'TEST_circulatory_data_maps.npy')))[:, :, :, :-truncate_amt].float()

    train_circulatory_labels = torch.from_numpy(np.load(os.path.join(data_path, 'train_circulatory_labels.npy')))[:, truncate_amt:].float() # Clip the left side to keep the 1's for positive samples
    TEST_circulatory_labels = torch.from_numpy(np.load(os.path.join(data_path, 'TEST_circulatory_labels.npy')))[:, truncate_amt:].float()

    train_circulatory_PIDs = torch.from_numpy(np.load(os.path.join(data_path, 'train_circulatory_PIDs.npy'))).float()
    TEST_circulatory_PIDs = torch.from_numpy(np.load(os.path.join(data_path, 'TEST_circulatory_PIDs.npy'))).float()
    
    
    random.seed(100)
    if encoder_type == 'TNC_ICU':
        print('checkpoint_str: ', checkpoint_str)
        checkpoint = torch.load(checkpoint_str)
        plot_heatmap=True
        encoder = CausalCNNEncoder(in_channels=36, channels=4, depth=1, reduced_size=2, encoding_size=10, kernel_size=2, window_size=12, device=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
        encoder.pruning_mask = checkpoint['pruning_mask']
        encoder = encoder.to(device)
        encoder.pruned_encoding_size = int(torch.sum(encoder.pruning_mask))
        print('encoder pruned_encoding_size: ', encoder.pruned_encoding_size)
        classifier_input_size = encoder.pruned_encoding_size
        print('encoder pruning_mask: ', encoder.pruning_mask)
    elif encoder_type == 'TNC':
        print('checkpoint_str: ', checkpoint_str)
        checkpoint = torch.load(checkpoint_str)

        plot_heatmap=False
        encoder = CausalCNNEncoder(in_channels=18, channels=4, depth=1, reduced_size=2, encoding_size=6, kernel_size=2, window_size=12, device=device)
        classifier_input_size = 6
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        train_circulatory_data_maps = train_circulatory_data_maps[:, 0, :, :]
        TEST_circulatory_data_maps = TEST_circulatory_data_maps[:, 0, :, :]
    elif encoder_type == 'CPC':
        checkpoint_str = './ckpt/HiRID_cpc/checkpoint_0.pth.tar'
        print('checkpoint_str: ', checkpoint_str)
        checkpoint = torch.load(checkpoint_str)

        plot_heatmap=False
        encoder = CausalCNNEncoder(in_channels=18, channels=4, depth=1, reduced_size=2, encoding_size=6, kernel_size=6, window_size=12, device=device)
        classifier_input_size = 6
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        train_circulatory_data_maps = train_circulatory_data_maps[:, 0, :, :]
        TEST_circulatory_data_maps = TEST_circulatory_data_maps[:, 0, :, :]
    
    elif encoder_type == 'triplet_loss':
        checkpoint_str = './ckpt/HiRID_trip/checkpoint_0.pth.tar'
        print('checkpoint_str: ', checkpoint_str)
        checkpoint = torch.load(checkpoint_str)

        plot_heatmap=False
        encoder = CausalCNNEncoder(in_channels=18, channels=4, depth=1, reduced_size=2, encoding_size=6, kernel_size=2, window_size=12, device=device)
        classifier_input_size = 6
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        train_circulatory_data_maps = train_circulatory_data_maps[:, 0, :, :]
        TEST_circulatory_data_maps = TEST_circulatory_data_maps[:, 0, :, :]
    
    elif encoder_type == 'raw':
        encoder = None
        train_circulatory_data_maps = train_circulatory_data_maps[:, 0, :, :]
        TEST_circulatory_data_maps = TEST_circulatory_data_maps[:, 0, :, :]
        classifier_input_size = 18 # 18 features in HiRID

    elif encoder_type == 'e2e':
        encoder = CausalCNNEncoder(in_channels=18, channels=4, depth=1, reduced_size=2, encoding_size=6, kernel_size=2, window_size=12, device=device)
        train_circulatory_data_maps = train_circulatory_data_maps[:, 0, :, :]
        TEST_circulatory_data_maps = TEST_circulatory_data_maps[:, 0, :, :]
        classifier_input_size = 6
    
    encoder_cv=0
    
    patients_with_circ_failure = train_circulatory_data_maps[torch.Tensor([1 in label for label in train_circulatory_labels]).bool()]
    print('Number of training samples with circ failure: ', len(patients_with_circ_failure))

    print('train_circulatory_data_maps shape: ', train_circulatory_data_maps.shape)
    print('train_circulatory_labels shape: ', train_circulatory_labels.shape)
    '''
    encodings = encoder.forward_seq(patients_with_circ_failure[0:1, :, :, -144:].to(device)) # Encode the last 12 hrs of each sample
    print('encodings shape: ', encodings.shape)
    plot_pca_trajectory_binned(encodings=encodings, path='./circulatory_failure_plots', 
    pca_file_name='trajectories_pca', event_name='Circulatory Failure')

    plot_tsne_trajectory_binned(encodings=encodings, path='./circulatory_failure_plots', 
    pca_file_name='trajectories_tsne', event_name='Circulatory Failure')

    patients_without_circ_failure = train_circulatory_data_maps[torch.Tensor([1 not in label for label in train_circulatory_labels]).bool()]
    print('Number of training samples without circ failure: ', len(patients_without_circ_failure))

    positive_encodings = encoder.forward_seq(patients_with_circ_failure[0:50, :, :, -36:].to(device)) # Encode the last 3 hrs of each sample
    negative_encodings = encoder.forward_seq(patients_without_circ_failure[0:50, :, :, -36:].to(device))
    
    positive_encodings = positive_encodings.reshape(-1, positive_encodings.shape[-1])
    negative_encodings = negative_encodings.reshape(-1, negative_encodings.shape[-1])
    positive_labels = torch.ones(positive_encodings.shape[0])
    negative_labels = torch.zeros(negative_encodings.shape[0])
    
    encodings = torch.vstack([positive_encodings, negative_encodings])
    labels = torch.cat([positive_labels, negative_labels])

    dim_reduction(encodings=encodings.detach().cpu(), labels=labels.detach().cpu(), save_path='./circulatory_failure_plots', plot_name='negative_and_positive_TSNE', label_names=['Normal', 'Circulatory Failure'], reduction_type='TSNE')
    dim_reduction(encodings=encodings.detach().cpu(), labels=labels.detach().cpu(), save_path='./circulatory_failure_plots', plot_name='negative_and_positive_PCA', label_names=['Normal', 'Circulatory Failure'], reduction_type='PCA')
    '''

    y_train = torch.Tensor([1 in label for label in train_circulatory_labels]).to(device)
    y_TEST = torch.Tensor([1 in label for label in TEST_circulatory_labels]).to(device)

    class_weights = torch.Tensor([len(torch.where(y_train==0)[0]), len(torch.where(y_train==1)[0])]).to(device)
    print('class counts: ', class_weights)
    class_weights = torch.max(class_weights)/class_weights
    print('class_weights: ', class_weights)
    print('y_train shape: ', y_train.shape)
    print('y_TEST shape: ', y_TEST.shape)
    
    # (X_train, y_train, X_validation, y_validation, X_TEST, y_TEST, 
    # classifier_input_size, baseline_type, encoder, window_size, class_weights, 
    # target_names, batch_size=32, return_models=False, return_scores=False, data_type='ICU', 
    # classification_cv=0, encoder_cv=0, ckpt_path="../ckpt",  plt_path="../DONTCOMMITplots", 
    # classifier_name=""):
    classifier = train_linear_classifier(X_train=train_circulatory_data_maps, y_train=y_train,
                            X_validation=train_circulatory_data_maps, y_validation=y_train,
                            X_TEST=TEST_circulatory_data_maps, y_TEST=y_TEST, 
                            classifier_input_size=classifier_input_size, baseline_type=encoder_type, encoder=encoder, window_size=12, 
                            target_names=['Normal', 'Circulatory Failure'], encoder_cv=encoder_cv, ckpt_path='./ckpt', 
                            plt_path='./DONTCOMMITplots/HiRID_circulatory_classification', 
                            classifier_name='circulatory_classifier', class_weights=class_weights, return_models=True, data_type='HiRID')
    
    if plot_heatmap:
        signal_list = ['Heart rate', 'Systolic BP', 'Diastolic BP', 'MAP', 'Cardiac output', 'SpO2', 'RASS',
                    'peak inspiratory pressure (ventilator)', 'Arterial Lactate', 'Lactate venous', 'INR',
                    'Serum glucose', '-reactive protein', 'Dobutamine', 'Milrinone', 'Levosimendan', 'Theophyllin',
                    'Non-opiod analgesics']

        inds = []
        while len(inds) < 40:
            ind = random.randrange(0, len(TEST_circulatory_data_maps))
            if y_TEST[ind] == 1:
                inds.append(ind)
        
        print('inds: ', inds)

        for ind in inds:
            sample = TEST_circulatory_data_maps[ind]
            sample_mask=[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            encodings = encoder.forward_seq(sample.unsqueeze(0).to(device)).squeeze()
            
            
            risk_scores = torch.nn.Softmax(dim=1)(classifier.forward(encodings.unsqueeze(0), return_full_seq=True).squeeze()).detach().cpu()[:, 1] # Score for positive class, i.e. risk of circ failure
            
            plot_heatmap_subset_signals_with_risk(sample=sample.cpu(), sample_mask=sample_mask, encodings=encodings.detach().cpu(),
            risk_scores=risk_scores, path='./DONTCOMMITplots/HiRID_circulatory_classification', hm_file_name='risk_heatmap%d.pdf'%ind,
            risk_plot_title='Risk of Circulatory Failure', signal_list=signal_list, length_of_hour=12, 
            risk_x_axis_label='Hours until circulatory failure', truncate_amt=truncate_amt)
            
            plot_pca_trajectory(encodings=encodings.detach().cpu(), path='./DONTCOMMITplots/HiRID_circulatory_classification', pca_file_name='trajectory%d.pdf'%ind, pos_sample_name='circulatory failure')




























