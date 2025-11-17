import torch

'''
result = torch.tensor([[0, 0, 0],
                 [0, 0, 0],
                [0, 0, 0]])

subframe_signal = torch.tensor([[10, 11, 12],
                          [20, 21, 22],
                          [30, 31, 32],
                          [40, 41, 42]])
frame=torch.tensor([0,1,1,2])
#result.index_add_(-2, frame, subframe_signal)
#print(result)
dimension=torch.Size([1])
subframe_signal = subframe_signal.view(*dimension, -1)
print(subframe_signal)
'''
import torch
import math
torch.set_printoptions(profile="full")
def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

max_len = 10000  # 最大序列长度
d_model = 512  # 模型的维度大小

pos_encoding = positional_encoding(max_len, d_model)

print(pos_encoding.shape)
import csv
sinfile=open('data/positional_encoding_sin.csv', 'w')
cosfile=open('data/positional_encoding_ocs.csv', 'w')
sinwriter = csv.writer(sinfile)
coswriter = csv.writer(cosfile)
input =pos_encoding.numpy()
input = input.tolist()
count=0
for i in input:
        #string_data = [str(k) for k in i]
        #sinwriter.writerows(zip(string_data))
        if count%2:
            string_data = [str(k) for k in i]
            sinwriter.writerows(zip(string_data))
        else:
            string_data = [str(k) for k in i]
            coswriter.writerows(zip(string_data))
        count=count+1
sinfile.close()
cosfile.close()