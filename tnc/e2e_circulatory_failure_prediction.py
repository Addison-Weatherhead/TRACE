import torch
import random
from tnc.models import CausalCNNEncoder, RnnPredictor
from tnc.utils import dim_reduction, detect_incr_loss
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
from sklearn.metrics import roc_curve

def linear_classifier_epoch_run(dataset, train, classifier, optimizer, data_type, window_size, encoder, encoding_size, class_weights, device):
    if train:
        classifier.train()
    else:
        classifier.eval()
    
    epoch_losses = []
    epoch_predictions = []
    epoch_labels = []
    for data_batch, label_batch in dataset:
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
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(predictions, label_batch.long())
        
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

def train_linear_classifier(X_train, y_train, X_validation, y_validation, X_TEST, y_TEST, encoding_size, encoder, window_size, target_names, class_weights, device, lr_list, weight_decay_list, n_epochs_list, batch_size=32, return_models=False, return_scores=False, data_type='ICU', classification_cv=0, encoder_cv=0, ckpt_path="../ckpt",  plt_path="../DONTCOMMITplots", classifier_name=""):
    '''
    Trains a classifier to predict positive events in samples. 
    X_train is of shape (num_train_samples, 2, num_features, seq_len)
    y_train is of shape (num_train_samples,)

    '''
    print("Training Linear Classifier", flush=True)

    print('X_train shape: ', X_train.shape)
    print('batch_size: ', batch_size)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = torch.utils.data.TensorDataset(X_validation, y_validation)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    TEST_dataset = torch.utils.data.TensorDataset(X_TEST, y_TEST)
    TEST_data_loader = torch.utils.data.DataLoader(TEST_dataset, batch_size=batch_size, shuffle=True)
    

    
    for lr in lr_list:
        for weight_decay in weight_decay_list:
            for n_epochs in n_epochs_list:
                cv_auroc = []
                cv_auprc = []
                for cv in range(3):
                    (unique, counts) = np.unique(y_train.cpu(), return_counts=True)
                    n_classes = len(unique)
                    classifier = RnnPredictor(encoding_size=encoder.pruned_encoding_size, hidden_size=8, n_classes=n_classes).to(device)
                    params = list(classifier.parameters()) + list(encoder.parameters())
                    #lr_list = [.007] #[0.01, 0.001, 0.0005, 0.007]
                    #weight_decay_list = [.001] #[0.001, 0.0001, 0]
                    #n_epochs_list = [120] #[100, 125, 150]

                    print('Learning Rate for classifier training: ', lr)
                    print('Weight Decay for classifier training: ', weight_decay)
                    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
                    train_losses = []
                    valid_losses = []
                    
                    for epoch in range(1, n_epochs + 1):
                        encoder.eval()
                        # linear_classifier_epoch_run(dataset, train, classifier, optimizer, data_type, window_size, encoder, encoding_size):
                        epoch_train_predictions, epoch_train_losses, epoch_train_labels = linear_classifier_epoch_run(dataset=train_data_loader, train=True,
                                                                    classifier=classifier,
                                                                    optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=encoding_size, class_weights=class_weights, device=device)

                        
                        classifier.eval()
                        epoch_validation_predictions, epoch_validation_losses, epoch_validation_labels = linear_classifier_epoch_run(dataset=validation_data_loader, train=False,
                                                                    classifier=classifier,
                                                                    optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=encoding_size, class_weights=class_weights, device=device)

                        
                        

                        epoch_TEST_predictions, epoch_TEST_losses, epoch_TEST_labels = linear_classifier_epoch_run(dataset=TEST_data_loader, train=False,
                                                                    classifier=classifier,
                                                                    optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=encoding_size, class_weights=class_weights, device=device)

                        
                        # TRAIN 
                        # Compute average over all batches in the epoch
                        epoch_train_loss = np.mean(epoch_train_losses)
                        epoch_train_predictions = torch.vstack(epoch_train_predictions) # of shape (num_samples, num_classes)
                        epoch_train_labels = torch.cat(epoch_train_labels)
                        

                        if n_classes == 2:
                            epoch_train_auroc = roc_auc_score(epoch_train_labels, epoch_train_predictions[:, 1])
                        else:
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

                        if n_classes == 2:
                            epoch_validation_auroc = roc_auc_score(epoch_validation_labels, epoch_validation_predictions[:, 1])
                        else:
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
                        epoch_TEST_predictions = torch.vstack(epoch_TEST_predictions) # of shape (num_samples, num_classes)
                        epoch_TEST_labels = torch.cat(epoch_TEST_labels)

                        if n_classes == 2:
                            epoch_TEST_auroc = roc_auc_score(epoch_TEST_labels, epoch_TEST_predictions[:, 1])
                            precision, recall, _ = precision_recall_curve(epoch_TEST_labels, epoch_TEST_predictions[:, 1])
                            # Compute AUPRC
                            epoch_TEST_auprc = auc(recall, precision)
                        else:
                            epoch_TEST_auroc = roc_auc_score(epoch_TEST_labels, epoch_TEST_predictions, multi_class='ovr')
                        # Compute precision recall curve
                        #TEST_precision, TEST_recall, _ = precision_recall_curve(epoch_TEST_labels, epoch_TEST_predictions)
                        # Compute AUPRC
                        #epoch_TEST_auprc = auc(TEST_recall, TEST_precision) # precision is the y axis, recall is the x axis, computes AUC of this curve
                        
                        epoch_TEST_predictions_argmax = torch.argmax(epoch_TEST_predictions, dim=1)
                        epoch_TEST_accuracy = len(torch.where(epoch_TEST_labels==epoch_TEST_predictions_argmax)[0])/epoch_TEST_labels.shape[0]


                        if epoch%10==0 or detect_incr_loss(valid_losses, 5):
                            print('Epoch %d Classifier Loss =====> Training Loss: %.5f \t Training AUROC: %.5f \t Training Accuracy: %.5f\t Validation Loss: %.5f \t Validation AUROC: %.5f \t Validation Accuracy %.5f\t TEST Loss: %.5f \t TEST AUROC: %.5f \t TEST Accuracy %.5f'
                                                % (epoch, epoch_train_loss, epoch_train_auroc, epoch_train_accuracy, epoch_validation_loss, epoch_validation_auroc, epoch_validation_accuracy, epoch_TEST_loss, epoch_TEST_auroc, epoch_TEST_accuracy))
                            
                            
                            


                            # Checkpointing the classifier model

                            state = {
                                    'epoch': epoch,
                                    'classifier_state_dict': classifier.state_dict(),
                                }

                            torch.save(state, os.path.join(ckpt_path, classifier_name + '.tar'))
                            if detect_incr_loss(valid_losses, 5):
                                break
                    
                    
                    # Plot Loss curves
                    plt.figure()
                    plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label="Train")
                    plt.plot(np.arange(1, len(valid_losses) + 1), valid_losses, label="Validation")
                    plt.title("Loss")
                    plt.legend()
                    plt.savefig(os.path.join(plt_path, "e2e_Classifier_loss_%d_%d.pdf"%(encoder_cv, classification_cv)))
                    
                    cv_auroc.append(epoch_TEST_auroc)
                    if n_classes == 2:
                        cv_auprc.append(epoch_TEST_auprc)

                    print("Train classification report: ")
                    print('epoch_train_labels shape: ', epoch_train_labels.shape, 'epoch_train_predictions_argmax shape: ', epoch_train_predictions_argmax.shape)
                    print(classification_report(epoch_train_labels.to('cpu'), epoch_validation_predictions_argmax, target_names=target_names))
                    print("Validation classification report: ")
                    print(classification_report(epoch_validation_labels.to('cpu'), epoch_validation_predictions_argmax, target_names=target_names))
                    print()
                    print("TEST classification report: ")
                    print(classification_report(epoch_TEST_labels.to('cpu'), epoch_TEST_predictions_argmax, target_names=target_names))
                    print()
                
                print('TEST RESULTS OVER CV')
                print('AUC: %.2f +- %.2f'%(np.mean(cv_auroc), np.std(cv_auroc)))
                if n_classes == 2:
                    print('AUPRC: %.2f +- %.2f'%(np.mean(cv_auprc), np.std(cv_auprc)))


    if return_models and return_scores:
        return (classifier, epoch_validation_auroc, epoch_validation_accuracy, epoch_TEST_auroc, epoch_TEST_accuracy)
    if return_models:
        return (classifier)
    if return_scores:
        return (epoch_validation_auroc, epoch_validation_auroc)


def apache_prediction(encoder, data_path, device):
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

    
    
    
    y_train = torch.Tensor([1 in label for label in train_circulatory_labels]).to(device)
    y_TEST = torch.Tensor([1 in label for label in TEST_circulatory_labels]).to(device)

    class_weights = torch.Tensor([len(torch.where(y_train==0)[0]), len(torch.where(y_train==1)[0])]).to(device)
    print('class counts: ', class_weights)
    class_weights = torch.max(class_weights)/class_weights
    print('class_weights: ', class_weights)
    print('y_train shape: ', y_train.shape)
    print('y_TEST shape: ', y_TEST.shape)




    train_linear_classifier(X_train=train_circulatory_data_maps, y_train=y_train, 
    X_validation=train_circulatory_data_maps, y_validation=y_train, # We don't optimize hyper parameters so just train and TEST is used.
    X_TEST=TEST_circulatory_data_maps, y_TEST=y_TEST, encoding_size=encoder.pruned_encoding_size,
    encoder=encoder, window_size=12, target_names=apache_names, ckpt_path='./ckpt', plt_path='./DONTCOMMITplots/HiRID_apache_classification', 
    classifier_name='e2e_apache_classifier', class_weights=class_weights, device=device, lr_list = [.002],
    weight_decay_list = [.0004], n_epochs_list = [100])


    


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    encoder = CausalCNNEncoder(in_channels=36, channels=4, depth=1, reduced_size=2, encoding_size=10, kernel_size=2, window_size=12, device=device)
    
    data_path = '../DONTCOMMITdata/hirid_numpy'
    
    print('encoder pruned_encoding_size: ', encoder.pruned_encoding_size)
    print('encoder pruning_mask: ', encoder.pruning_mask)

    apache_prediction(encoder=encoder, data_path=data_path, device=device)

    
    








