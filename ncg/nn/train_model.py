import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_iter(decoder, train_data, loss_criterion, optimizer, 
               max_epochs = 5, val_data = [],
               fn_batch_listeners = [], fn_epoch_listeners = []):
    decoder.to(device)

    for epoch in range(max_epochs):
        for i, (source_encodings, targets) in enumerate(train_data):
            source_encodings, targets = source_encodings.to(device), targets.to(device)
            token_loss = train(decoder, source_encodings, targets, 
                               loss_criterion, optimizer)
            for fn_on_update in fn_batch_listeners:
                fn_on_update(epoch, i, token_loss)
        for fn_on_epoch_completed in fn_epoch_listeners:
            val_loss = 0
            if val_data:
                val_loss = calculate_validation_loss(decoder, val_data, loss_criterion)
            fn_on_epoch_completed(epoch, i, val_loss)
    
def train(decoder, source_encodings, targets, loss_criterion, optimizer):
    optimizer.zero_grad()
    token_loss = calculate_loss(decoder, source_encodings, targets, loss_criterion)
    token_loss.backward()
    optimizer.step()    
    return token_loss.item() 

def calculate_loss(decoder, source_encodings, targets, loss_criterion):
    inputs = targets[:,:-1] # remove EOS token    
    output_probs = calculate_output_probabilities(decoder, source_encodings, inputs)
    target_outputs = targets.view(-1)[1:] # remove EOS token

    loss = loss_criterion(output_probs, target_outputs)
    return loss

def calculate_output_probabilities(decoder, source_encoding, inputs):
    hidden = source_encoding.view(1,1,-1)
    output_probs = torch.zeros(inputs.size(1), decoder.output_size, device=device)
    for i, input_token in enumerate(inputs.view(-1, 1)):
        output, hidden = decoder(input_token, hidden)
        output_probs[i, :] = output
    return output_probs

def calculate_validation_loss(decoder, val_data, loss_criterion):
    # TODO: use teacher forcing or not?
    # TODO: take token length into account?
    with torch.no_grad():
        total_loss = 0
        total_tokens = 0
        for source_encodings, targets in val_data:
            source_encodings, targets = source_encodings.to(device), targets.to(device)
            token_loss = calculate_loss(decoder, source_encodings, targets, loss_criterion)
            token_length = 1 #targets.size()[1] - 1
            total_loss += token_length * token_loss.item()
            total_tokens += token_length
    return total_loss / total_tokens

def predict(decoder, source_encoding, SOS_token, EOS_token, max_length):
    with torch.no_grad():
        hidden = source_encoding.view(1,1,-1)
        decoded_tokens = [SOS_token]
        input_token = torch.LongTensor([SOS_token], device=device)
        for di in range(max_length):
            output, hidden = decoder(input_token, hidden)
            topv, topi = output.data.topk(1)
            decoded_tokens.append(topi.item())
            if topi.item() == EOS_token:
                return decoded_tokens
            input_token = topi.squeeze().detach()
    decoded_tokens.append(EOS_token)
    return decoded_tokens

