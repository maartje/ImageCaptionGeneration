import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, pad_index, drop_out):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pad_index = pad_index

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        
        self.dropout_embedding = nn.Dropout(p = drop_out)
        self.dropout_lstm = nn.Dropout(p = drop_out)

    def forward(self, hidden, input_data, seq_lengths):
        output = self.dropout_embedding(self.embedding(input_data))
        packed = pack_padded_sequence (
            output, seq_lengths, batch_first=True)
        output, hidden = self.lstm(packed, hidden)
        unpacked = pad_packed_sequence(
            output, batch_first=True, padding_value=self.pad_index, total_length=None)
        output = self.out(self.dropout_lstm(unpacked[0]))
        output = self.logsoftmax(output)
        return output, hidden
        
class ShowTell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pad_index, drop_out = 0.3):
        super(ShowTell, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, output_size, pad_index, drop_out)

        self.dropout_hidden = nn.Dropout(p = drop_out)
    
    def forward(self, features, input_data, seq_lengths, state = None, device=None):
        features = features.unsqueeze(0)
        if state is None:
            h_0 = self.dropout_hidden(F.relu(self.encoder(features)))
            c_0 = torch.zeros(h_0.size()) # TODO GPU
            if not (device is None):
                c_0 = c_0.to(device)
            state = (h_0, c_0)
        return self.decoder(state, input_data, seq_lengths) # output, (h_n, c_n)

