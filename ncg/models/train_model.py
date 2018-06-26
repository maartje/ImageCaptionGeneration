def train_iter(decoder, train_data, loss_criterion, optimizer, 
               device, max_epochs = 5, fn_on_update = None):
    losses = []
    for epoch in range(max_epochs):
        for i, (source_encodings, targets) in enumerate(train_data):
            token_loss = train(decoder, source_encodings, targets, 
                               loss_criterion, optimizer)
            losses.append(token_loss)
            if fn_on_update:
                fn_on_update(epoch + 1, i + 1, token_loss)
    return losses
    
def train(decoder, source_encodings, targets, loss_criterion, optimizer):
    optimizer.zero_grad()
    token_loss = calculate_loss(decoder, source_encodings, targets, loss_criterion)
    token_loss.backward()
    optimizer.step()    
    return token_loss.item() 

def calculate_loss(decoder, source_encodings, targets, loss_criterion):
    inputs = targets[:,:-1] # remove EOS token    
    output_probs = decoder.calculate_output_probabilities(source_encodings, inputs)
    target_outputs = targets.view(-1)[1:] # remove EOS token

    loss = loss_criterion(output_probs, target_outputs)
    return loss

