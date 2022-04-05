
import matplotlib.pyplot as plt
#import seaborn as sns
#import numpy as np
import torch
from models import CausalCNNEncoder
#from tnc.utils import dim_reduction
#from tnc.models import RnnPredictor
#f, ax = plt.subplots(1)  #
#f.set_figheight(7)
#f.set_figwidth(23)
#ax.set_facecolor('w')




#data = torch.randn(50, 8)
##labels = torch.randint(low=0, high=9, size=(50,))
#names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
#dim_reduction(data, labels, 'ICU', '9999', '9999', 'test', names)




class RnnPredictor(torch.nn.Module):
    def __init__(self, encoding_size, hidden_size):
        super(RnnPredictor, self).__init__()
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.rnn = torch.nn.LSTM(input_size=encoding_size,  hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, x_lengths=None):
        # x is of shape (num_samples, seq_len, num_features)
        output, (h_n, _) = self.rnn(x)
        preds = []
        for hidden_states in output:
            preds.append(self.linear(hidden_states).squeeze())
        
        # of shape (num_samples, seq_len). The ij'th element is the prediction of risk for the jth time step of the ith sample.
        preds = torch.vstack(preds) # Note this must be passed through sigmoid after output
        
        return preds

'''
rnn = RnnPredictor(encoding_size=7, hidden_size=15)
# (num_samples, seq_len, num_features)
data = torch.randn(100, 50, 7)
output = rnn(data)
print(output.shape)
print(output[10])
'''


'''
import tnc.alluvial as alluvial
import matplotlib.pyplot as plt

input_data = {'cluster 1': {'Cardiovascular': 96, 'Pulmonary': 40,},
              'cluster 2': {'Cardiovascular': 2, 'Gastrointestinal': 0.5,},
              'cluster 3': {'Cardiovascular': 0.5, 'Pulmonary': 0.5, 'Gastrointestinal': 1.5,}}

ax = alluvial.plot(input_data)
fig = ax.get_figure()
ax.set_title('Utility display', fontsize=14)
fig.set_size_inches(10,5)
plt.savefig('./test.pdf')
'''


device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoding_size=8
window_size=60
encoder = CausalCNNEncoder(in_channels=20, channels=8, depth=2, reduced_size=30, encoding_size=encoding_size, kernel_size=3, window_size=window_size, device=device)
encoder = encoder.to(device)

print('encoder.encoding_size: ', encoder.encoding_size)
print('encoder.pruned_encoding_size: ', encoder.pruned_encoding_size)
print(encoder.pruning_mask)
checkpoint = torch.load('../../ckpt/ICU/0703_CausalCNNEncoder_ICU_checkpoint_0.tar')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
encoder.pruning_mask = checkpoint['pruning_mask']
encoder.pruned_encoding_size = int(torch.sum(encoder.pruning_mask))

print('encoder.encoding_size: ', encoder.encoding_size)
print('encoder.pruned_encoding_size: ', encoder.pruned_encoding_size)
print(encoder.pruning_mask)



