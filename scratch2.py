from pandas.core.indexes.base import Index
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import argparse
import math
import seaborn as sns; sns.set()
import random
from sklearn.cluster import AgglomerativeClustering
random.seed(0)


clustering_encodings = torch.Tensor([[[ -1,  -7],
         [  5,  -4],
         [ -4,   0],
         [  8,   9],
         [ -1,  -4]],

        [[ -4,  -7],
         [  0,   0],
         [  0,   2],
         [ -1,  11],
         [ -1,  -4]],

        [[ -7,   7],
         [ -3,   1],
         [ -2,  -5],
         [ -9,  -8],
         [  4,   9]],

        [[  7,   2],
         [ 12,   0],
         [  7,   2],
         [-10,  -6],
         [ -7,   4]]])

clustering_encodings = clustering_encodings.reshape(5, -1)
print(clustering_encodings.shape)

clustering_model = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='cosine', compute_distances=True).fit(clustering_encodings)

print(clustering_model.labels_)


'''
encoding_mask = torch.ones(4, 5)
encoding_mask[0, 0:2] = -1
encoding_mask[1, 0:3] = -1
encoding_mask[2, 0] = -1
encoding_mask[3, 0:3] = -1

print('Encodings: ')
print(clustering_encodings)

print('Encoding Mask: ')

print(encoding_mask)


labels = torch.zeros(4, 5)

labels[0, 4] = 1
labels[3, 4] = 1
labels[3, 3] = 1

print('Labels: ')
print(labels)

pos_inds = torch.Tensor([1 in labels[ind] for ind in range(labels.shape[0])]).nonzero()
neg_inds = torch.Tensor([1 not in labels[ind] for ind in range(labels.shape[0])]).nonzero()

print(pos_inds)
print(neg_inds)

print('========================================================')

num_sliding_windows_per_sample = 5





pos_clustering_encodings = clustering_encodings[pos_inds] # shape (num_pos_samples, num_sliding_windows_per_sample, encoding_size)
neg_clustering_encodings = clustering_encodings[neg_inds] # shape (num_neg_samples, num_sliding_windows_per_sample, encoding_size
pos_encoding_mask = encoding_mask[pos_inds] # shape (num_pos_samples, num_sliding_windows_per_sample)
neg_encoding_mask = encoding_mask[neg_inds] # shape (num_neg_samples, num_sliding_windows_per_sample)


clustering_encodings = clustering_encodings.reshape(-1, clustering_encodings.shape[-1]) # of shape (num_samples*num_sliding_windows_per_sample, encoding_size)
pos_clustering_encodings = pos_clustering_encodings.reshape(-1, pos_clustering_encodings.shape[-1])
neg_clustering_encodings = neg_clustering_encodings.reshape(-1, neg_clustering_encodings.shape[-1])



encoding_mask = encoding_mask.reshape(-1,) # of shape (num_samples*num_sliding_windows_per_sample)
pos_encoding_mask = pos_encoding_mask.reshape(-1,) # of shape (num_pos_samples*num_sliding_windows_per_sample)
neg_encoding_mask = neg_encoding_mask.reshape(-1,)

pos_inds = torch.cat([torch.arange(int(ind)*num_sliding_windows_per_sample, int(ind)*num_sliding_windows_per_sample + num_sliding_windows_per_sample) for ind in pos_inds]) # Now the indices are ready for the reshaped encodings
neg_inds = torch.cat([torch.arange(int(ind)*num_sliding_windows_per_sample, int(ind)*num_sliding_windows_per_sample + num_sliding_windows_per_sample) for ind in neg_inds])



# Only keep encodings that were not created from fully imputed data
masked_clustering_encodings = clustering_encodings[encoding_mask!=-1].detach().to('cpu')
pos_clustering_encodings = pos_clustering_encodings[pos_encoding_mask!=-1].detach().to('cpu')
neg_clustering_encodings = neg_clustering_encodings[neg_encoding_mask!=-1].detach().to('cpu')
pos_inds = pos_inds[pos_encoding_mask!=-1]
neg_inds = neg_inds[neg_encoding_mask!=-1]

all_inds = torch.sort(torch.cat([pos_inds, neg_inds]))[0]
print(all_inds)

masked_pos_inds = torch.nonzero(pos_inds[:, None] == all_inds[None, :])[:, 1]
masked_neg_inds = torch.nonzero(neg_inds[:, None] == all_inds[None, :])[:, 1] # Now masked_pos_inds and masked_neg_inds can index masked_clustering_encodings

print(masked_clustering_encodings)
print(pos_clustering_encodings)
print(neg_clustering_encodings)
print(pos_inds)
print(neg_inds)
print()

print(masked_clustering_encodings[masked_pos_inds])
print(masked_clustering_encodings[masked_neg_inds])


'''