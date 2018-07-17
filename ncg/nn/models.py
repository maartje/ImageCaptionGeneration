import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, pad_index):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pad_index = pad_index

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, hidden, input_data, seq_lengths):
        output = self.embedding(input_data)
        output = F.relu(output)
        packed = pack_padded_sequence (
            output, seq_lengths, batch_first=True)
        output, hidden = self.gru(packed, hidden)
        unpacked = pad_packed_sequence(
            output, batch_first=True, padding_value=self.pad_index, total_length=None)
        output = self.out(unpacked[0])
        output = self.logsoftmax(output)
        return output, hidden
        
class ShowTell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pad_index):
        super(ShowTell, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, output_size, pad_index)
    
    def forward(self, features, input_data, seq_lengths, hidden = None):
        if hidden is None:
            hidden = F.relu(self.encoder(features))
        output, hidden = self.decoder(hidden, input_data, seq_lengths)
        return output, hidden

