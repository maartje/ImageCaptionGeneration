import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_iter(decoder, train_data, loss_criterion, optimizer, 
               fn_stop_criterion, val_data = [],
               fn_batch_listeners = [], fn_epoch_listeners = []):
    decoder.to(device)
    loss_criterion.to(device)
    
    epoch = 0
    val_loss = None
    while not fn_stop_criterion(epoch, val_loss):
        for batch_index, batch in enumerate(train_data):
            token_loss = train(decoder, batch, loss_criterion, optimizer)
            for fn_on_update in fn_batch_listeners:
                batch_size = batch[0].size() # may differ for last batch
                fn_on_update(epoch, batch_index, batch_size, token_loss)
        if val_data:
            val_loss = calculate_validation_loss(decoder, val_data, loss_criterion)
        for fn_on_epoch_completed in fn_epoch_listeners:
            fn_on_epoch_completed(epoch, batch_index, val_loss)
        epoch += 1
    
def train(decoder, batch, loss_criterion, optimizer):
    optimizer.zero_grad()
    token_loss = calculate_loss(decoder, batch, loss_criterion)
    token_loss.backward()
    optimizer.step()    
    return token_loss.item() 

def calculate_loss(decoder, batch, loss_criterion):
    (image_features, caption_inputs, caption_targets, caption_lengths) = batch
    image_features = image_features.to(device) 
    caption_inputs = caption_inputs.to(device)
    caption_targets = caption_targets.to(device)
    caption_lengths = caption_lengths.to(device)

    output_probs = decoder(
        image_features, caption_inputs, caption_lengths
    )    
    loss = loss_criterion(output_probs.permute(0, 2, 1), caption_targets)
    return loss

def calculate_validation_loss(decoder, val_data, loss_criterion):
    # TODO: use teacher forcing or not?
    # TODO: take token length into account?
    with torch.no_grad():
        total_loss = 0.
        total_tokens = 0
        for batch in val_data:
            token_loss = calculate_loss(decoder, batch, loss_criterion)
            token_length = 1 #targets.size()[1] - 1
            total_loss += token_length * token_loss.item()
            total_tokens += token_length
    return total_loss / total_tokens

# TODO FIX
# def predict(decoder, source_encoding, SOS_token, EOS_token, max_length):
#     with torch.no_grad():
#         hidden = source_encoding.view(1,1,-1)
#         decoded_tokens = [SOS_token]
#         input_token = torch.LongTensor([SOS_token], device=device)
#         for di in range(max_length):
#             output, hidden = decoder(input_token, hidden)
#             topv, topi = output.data.topk(1)
#             decoded_tokens.append(topi.item())
#             if topi.item() == EOS_token:
#                 return decoded_tokens
#             input_token = topi.squeeze().detach()
#     decoded_tokens.append(EOS_token)
#     return decoded_tokens

