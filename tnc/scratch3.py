from os import pread
import torch
import numpy as np
from tnc.models import CausalCNNEncoder
import hdbscan
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from sklearn.datasets import make_blobs
print(device)
encoder = CausalCNNEncoder(in_channels=20,
                           channels=8,
                           depth=2,
                           reduced_size=60,
                           out_channels=16,
                           kernel_size=4,
                           device=device)

clustering_model = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True, core_dist_n_jobs=4)
data = torch.randn(50, 2, 10, 120)

encodings = encoder(data).detach().cpu().numpy()

clustering_model.fit(encodings)
predictions = clustering_model.labels_
print(predictions.shape)
print(predictions.min())
print(predictions.max())
print(type(predictions))

print('===')
TEST_mixed_labels = torch.from_numpy(np.load('/datasets/sickkids/TNC_ICU_data/test_mixed_labels.npy'))


thing = torch.Tensor([1, 2, 4]).reshape(-1, 1)
print(thing.repeat(1, 5))
print('yo')
print(clustering_model.n_clusters)
