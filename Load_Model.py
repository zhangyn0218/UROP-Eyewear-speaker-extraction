import torch
ckpt=torch.load('pretrain_model/backend.pt')
for i in ckpt:
    print(i)