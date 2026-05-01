import torch
checkpoint = torch.load('/Users/devariwala/Desktop/Development/no10/models/lstm_rul.pth', map_location='cpu')
if isinstance(checkpoint, dict) and 'config' in checkpoint:
    print(checkpoint['config'])
else:
    print("No config in checkpoint")
