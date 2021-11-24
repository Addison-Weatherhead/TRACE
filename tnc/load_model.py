import torch
from tnc.models import CausalCNNEncoder, RnnPredictor, LinearClassifier
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


encoder_checkpoint_file = 'ckpt/ICU/0655_CausalCNNEncoder_ICU_checkpoint_0.tar'
encoder_checkpoint = torch.load(encoder_checkpoint_file, map_location=torch.device('cpu'))
encoder = encoder_checkpoint['encoder'] # The entire model is simply saved in the checkpoint, no need to load a state dict or anything

classifier_checkpoint_file = 'ckpt/ICU/0655_encoder_checkpoint_0_sampleClassifier_checkpoint_0.tar'
classifier_checkpoint = torch.load(classifier_checkpoint_file, map_location=torch.device('cpu'))
classifier = classifier_checkpoint['classifier']
rnn = classifier_checkpoint['rnn']
'''


encoder = CausalCNNEncoder(20, 8, 2, 60, 16, 4, device)
rnn = RnnPredictor(16, 32)
classifier = LinearClassifier(32)
'''


class Model(torch.nn.Module):
    def __init__(self, encoder, rnn, classifier) -> None:
        super(Model, self).__init__()
        self.encoder = encoder
        self.rnn = rnn
        self.classifier = classifier

    def forward(self, X, return_encodings=True):
        # X is of shape (num_windows, 2, num_features, window_size) 
        # where window_size is 120 (10 minutes), num_features is 10 (order is given by ["Pulse", "HR", "SpO2", "etCO2", "NBPm", "NBPd", "NBPs", "RR", "CVPm", "awRR"])
        # the second dimension (2) is channels for mask vs data. X[:, 0, :, :] is the normalized data
        # X[:, 1, :, :] is the mask. 1's should indicate observed values, 0's should indicate missing.

        #

        encodings = self.encoder(X)
        encodings = torch.unsqueeze(encodings, 0)
        
        output, _ = self.rnn(encodings) # output contains hidden state for each time step. Shape is (batch_size, seq_len, hidden_size). batch_size=1
        output = output.squeeze(0) # now of shape (seq_len, hidden_size). Can be thought of as seq_len encodings, each of size hidden_size
        # Apply sigmoid because the classifier doesn't apply this because we use BCEWITHLOGITSLOSS in train_linear_classifier
        risk_scores_over_time = torch.nn.Sigmoid()(self.classifier(output).to('cpu')).detach() 

        if return_encodings:
            return encodings.squeeze(), risk_scores_over_time
        else:
            return risk_scores_over_time




full_model = Model(encoder, rnn, classifier)
full_model = full_model.to(device)




torch.save(full_model.state_dict(), './tnc/model.tar')

full_model.load_state_dict(torch.load('tnc/model.tar'))

X = torch.randn(1, 2, 10, 120)


encodings, risk_scores = full_model(X)


print(encodings.shape)
print(risk_scores.shape)
