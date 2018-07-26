import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(decoder, predict_data, SOS_index, max_length):
    results = []
    decoder.to(device)
    with torch.no_grad():
        for batch in predict_data:
            image_features = batch
            predicted_tokens = []
            image_features = image_features.unsqueeze(0)
            batch_size = batch.size()[0]
            inputs = torch.LongTensor([[SOS_index]]*batch_size, device = device)
            lengths = torch.ones([batch_size], dtype=torch.long, device = device)
            hidden = None
            for i in range(max_length):
                output, hidden = decoder(image_features, inputs, lengths, hidden)
                _, topi = output.topk(1)
                predicted_tokens.append(topi.squeeze(1))
                inputs = topi.squeeze(1)
            result = torch.cat(predicted_tokens, 1)
            results.append(result)
        return torch.cat(results).tolist()

