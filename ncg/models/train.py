def train_iter(decoder, batches, loss_criterion, optimizer, epochs = 5):
    batch_losses = []
    for epoch in range(epochs):
        for training_pairs in batches:
            token_loss = train(decoder, training_pairs, 
                               loss_criterion, optimizer)
            batch_losses.append(token_loss)
    return batch_losses
    
def train(decoder, training_pairs, loss_criterion, optimizer):
    optimizer.zero_grad()

    loss = 0
    total_tokens = 0
    for source_encoding, target in training_pairs:
        loss_per_token = calculate_loss(decoder, source_encoding, target, loss_criterion)
        nr_of_tokens = (target.size(0) - 1)
        loss += loss_per_token * nr_of_tokens
        total_tokens += nr_of_tokens
    token_loss = loss / total_tokens
    token_loss.backward()
    optimizer.step()
    
    return token_loss.item() 

def calculate_loss(decoder, source_encoding, target, loss_criterion):
    inputs = target[:-1] # remove EOS token
    output_probs = decoder.calculate_output_probabilities(source_encoding, inputs)
    
    target_outputs = target[1:].view(-1) # remove EOS token
    loss = loss_criterion(output_probs, target_outputs)
    return loss

