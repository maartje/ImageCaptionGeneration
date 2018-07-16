import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, hidden, input_data, seq_lengths):
        output = self.embedding(input_data)
        output = F.relu(output)
        packed = pack_padded_sequence (
            output, seq_lengths, batch_first=True)
        output, hidden = self.gru(packed, hidden.unsqueeze(0))
        unpacked = pad_packed_sequence(output, batch_first=True, padding_value=-1, total_length=None)
        output = self.out(unpacked[0])
        output = self.logsoftmax(output)
        return output
        

