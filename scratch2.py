from pandas.core.indexes.base import Index
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import argparse
import math
import seaborn as sns; sns.set()
import sys
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

import torch.nn.functional as F
x = torch.randn(10, 2, 10, 5040)
print(x.shape)


encoder = CausalCNNEncoder(in_channels=20, channels=8, depth=2, reduced_size=60, encoding_size=16, kernel_size=3, window_size=120, device=device)


out, encoding_mask = encoder.forward_seq(x, sliding_gap=20, return_encoding_mask=True)
out = out.reshape(-1, 16)
print('encodings shape:')
print(out.shape)
print('encodings mask shape:')
print(encoding_mask.shape)

encodings = []
for i in range(0, 5040-120, 20):
    encodings.append(encoder(x[:, :, :, i: i+120]))

encodings = torch.stack(encodings).permute(1, 0, 2).reshape(-1, 16)

print(encodings.shape)

print(encodings - out)

