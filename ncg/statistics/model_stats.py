from torchsummary import summary

def printDecoderInfo(decoder):
    print(decoder)
    
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f'Number of parameters: {total_params}')

    total_params_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {total_params_trainable}')

def printEncoderInfo(encoder, input_size = (3,224,224)):
    summary(encoder, input_size)

