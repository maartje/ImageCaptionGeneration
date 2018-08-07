import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(decoder, predict_data, SOS_index, max_length):
    results = []
    decoder.to(device)
    decoder.eval()
    with torch.no_grad():
        for batch in predict_data:
            image_features = batch.to(device)
            predicted_tokens = []
            batch_size = image_features.size()[0]
            image_features = image_features.unsqueeze(0)
            inputs = torch.LongTensor([[SOS_index]]*batch_size)
            lengths = torch.ones([batch_size], dtype=torch.long)
            hidden = None
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            for i in range(max_length):
                output, hidden = decoder(image_features, inputs, lengths, hidden, device)
                _, topi = output.topk(1)
                predicted_tokens.append(topi.squeeze(1))
                inputs = topi.squeeze(1)
            result = torch.cat(predicted_tokens, 1)
            results.append(result)
        return torch.cat(results).tolist()

