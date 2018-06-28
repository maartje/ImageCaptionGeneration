import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_iter(decoder, train_data, loss_criterion, optimizer, 
               max_epochs = 5, fns_on_update = []):
    decoder.decoder.to(device)

    for epoch in range(max_epochs):
        for i, (source_encodings, targets) in enumerate(train_data):
            source_encodings, targets = source_encodings.to(device), targets.to(device)
            token_loss = train(decoder, source_encodings, targets, 
                               loss_criterion, optimizer)
            for fn_on_update in fns_on_update:
                epoch_finished = (i + 1 == len(train_data))
                fn_on_update(epoch, i, token_loss, epoch_finished)
    
def train(decoder, source_encodings, targets, loss_criterion, optimizer):
    optimizer.zero_grad()
    token_loss = calculate_loss(decoder, source_encodings, targets, loss_criterion)
    token_loss.backward()
    optimizer.step()    
    return token_loss.item() 

def calculate_loss(decoder, source_encodings, targets, loss_criterion):
    inputs = targets[:,:-1] # remove EOS token    
    output_probs = decoder.calculate_output_probabilities(source_encodings, inputs, device)
    target_outputs = targets.view(-1)[1:] # remove EOS token

    loss = loss_criterion(output_probs, target_outputs)
    return loss

