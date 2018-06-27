import torch

from ncg.nn.models import DecoderRNN

class ShowAndTell():

    def __init__(self, hidden_size, output_size):
        self.decoder = DecoderRNN(hidden_size, output_size)
            
    def calculate_output_probabilities(self, source_encoding, inputs, device):
        hidden = source_encoding.view(1,1,-1)
        output_probs = torch.zeros(inputs.size(1), self.decoder.output_size, device=device)
        for i, input_token in enumerate(inputs.view(-1, 1)):
            output, hidden = self.decoder(input_token, hidden)
            output_probs[i, :] = output
        return output_probs



