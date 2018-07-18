import torch

def predict(decoder, predict_data, SOS_index, max_length):
    results = []
    with torch.no_grad():
        for batch in predict_data:
            image_features = batch
            predicted_tokens = []
            image_features = image_features.unsqueeze(0)
            batch_size = batch.size()[0]
            inputs = torch.LongTensor([[SOS_index]]*batch_size)
            lengths = torch.ones([batch_size], dtype=torch.long)
            hidden = None
            for i in range(max_length):
                output, hidden = decoder(image_features, inputs, lengths, hidden)
                _, topi = output.topk(1)
                predicted_tokens.append(topi.squeeze(1))
                inputs = topi.squeeze(1)
            result = torch.cat(predicted_tokens, 1)
            results.append(result)
        return torch.cat(results).tolist()

