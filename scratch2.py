from pandas.core.indexes.base import Index
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import argparse
import math
import seaborn as sns; sns.set()
import sys
#import statsmodels.api as sm
import numpy as np
import pickle
import os
import random
os.environ['MKL_THREADING_LAYER'] = 'GNU' # Set this value to allow grid_search.py to work.
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from datetime import datetime
from tnc.models import CNN_Transformer_Encoder, EncoderMultiSignalMIMIC, GRUDEncoder, RnnEncoder, WFEncoder, TST, EncoderMultiSignal, LinearClassifier, RnnPredictor, EncoderMultiSignalMIMIC, CausalCNNEncoder
from tnc.utils import plot_distribution, plot_heatmap, PCA_valid_dataset_kmeans_labels, plot_normal_and_mortality, plot_pca_trajectory
from tnc.evaluations import WFClassificationExperiment, ClassificationPerformanceExperiment
#from statsmodels.tsa.stattools import adfuller, acf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
import hdbscan

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



x = torch.randn(15, 2, 6, 49)
window_size = 7

encoder = CausalCNNEncoder(in_channels=12, channels=3, 
                            depth=4, reduced_size=4, out_channels=10, 
                            kernel_size=3, device=device, 
                            window_size=window_size)

classifier_encodings = []
for ind in range(len(x)):
    sample = x[ind]

    windows = []
    i = 0
    while i * window_size < sample.shape[-1]:
        windows.append(sample[:, :, i*window_size: (i+1)*window_size])
        i += 1
    
    windows = torch.stack(windows)
    classifier_encodings.append(encoder(windows))

classifier_encodings = torch.stack(classifier_encodings)

encodings = encoder.forward_seq(x)

print(classifier_encodings[14, 4, :])
print(encodings[14, 4, :])

print(encodings.shape)

