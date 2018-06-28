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


    def predict(self, source_encoding, SOS_token, EOS_token, max_length, device):
        with torch.no_grad():
            hidden = source_encoding.view(1,1,-1)
            decoded_tokens = [SOS_token]
            input_token = torch.LongTensor([SOS_token], device=device)
            for di in range(max_length):
                output, hidden = self.decoder(input_token, hidden)
                topv, topi = output.data.topk(1)
                decoded_tokens.append(topi.item())
                if topi.item() == EOS_token:
                    break
                input_token = topi.squeeze().detach()
        return decoded_tokens



