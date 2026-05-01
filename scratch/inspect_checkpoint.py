import torch
checkpoint = torch.load('C:\\Users\\kunwa\\AI Sem 4\\EST PROJECT\\AEROPREDICT\\models\\lstm_rul.pth', map_location='cpu')
if isinstance(checkpoint, dict) and 'config' in checkpoint:
    print(checkpoint['config'])
else:
    print("No config in checkpoint")
