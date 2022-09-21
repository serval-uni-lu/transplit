import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import FullAttention, AttentionLayer


class SVS(nn.Module):
    def __init__(self, c_in, d_model, c_out, period, n_filters, seq_len):
        super(SVS, self).__init__()
        
        self.c_in = c_in
        self.d_model = d_model
        self.c_out = c_out
        self.period = period
        self.n_filters = n_filters
        
        self.conv_layers = [
            nn.Conv1d(1, n_filters, kernel_size=period, stride=period, bias=False)
            for _ in range(c_in)
        ]
        
        for layer in self.conv_layers:
            layer.to('cuda')
            
        self.embedding = nn.Linear(n_filters * c_in, d_model)
        self.activation = nn.ReLU()
    
    def encode(self, x):
        x = x.transpose(1, 2)
        c = []
        for i in range(self.c_in):
            c.append(self.conv_layers[i](x[:, i:i+1, :]))
        x = torch.concat(c, dim=1)
        x = x.transpose(1, 2)
        x = self.embedding(x)
        return self.activation(x)
    
    def forward(self, x):
        return self.encode(x)
    
    def decode(self, x, x_input):
        # x.shape = (batch_size, y_seq_len // period, n_filters * c_out)
        c = []
        for i in range(self.c_out):
            W = self.conv_layers[i].weight.squeeze()
            b1, b2 = i*self.n_filters, (i+1)*self.n_filters
            xi = torch.matmul(x[..., b1:b2], W)
            xi = xi.flatten(start_dim=1)
            c.append(xi)
        return torch.stack(c, dim=-1)

