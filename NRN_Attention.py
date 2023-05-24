import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Transformer_EncDec import Encoder, EncoderLayer
from transformer.SelfAttention import FullAttention, AttentionLayer
from NRN_MLP import distance_encoder
import numpy as np
from einops.layers.torch import Rearrange

class NRN_a(nn.Module):
    def __init__(self, features, K_neighbor, depth, d_model, n_heads, drop_out=0.4):
        super(NRN_a, self).__init__()

        self.features = features
        self.K_neighbor = K_neighbor
        self.inputRearrange = Rearrange('b x -> b 1 x')
        self.drop_out = nn.Dropout(p=drop_out)

        self.enc_embedding = distance_encoder(features, d_model)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False), d_model, n_heads),
                    d_model
                ) for l in range(depth)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            Rearrange(' b n d -> b d n'),
            nn.Linear(K_neighbor, 1),
            Rearrange(' b d n -> b (d n)')
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, features)
        )
    def forward(self, x):

        x = self.inputRearrange(x)
        x_k_split = torch.chunk(x, self.K_neighbor, dim=2)
        cat_list =[]
        for i in range(self.K_neighbor):
            k_i = x_k_split[i]
            if i==0:
                _x_0 = x_k_split[i]
                k_i = self.drop_out(k_i)
            cat_list.append(k_i)
        x = torch.cat(cat_list, dim=1)
        
        # [B, N, D]
        
        enc_out = self.enc_embedding(x)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(self.fc(enc_out))


        _x_0 = _x_0.squeeze(1)
        missing_pos = (_x_0 != 0)
        _x0_masked = torch.masked_select(_x_0, missing_pos)
        _reconst_masked = torch.masked_select(dec_out, missing_pos).reshape(-1)
        mse = torch.mean(torch.square(_x0_masked - _reconst_masked))
        return dec_out, mse  # [B, D]
    
if __name__ == "__main__":

    K = 5
    features = 10

    in_x = torch.randn(2, K * features)


    print(in_x.shape)
    x_ = torch.chunk(in_x, K, dim=1)
    print(x_[0].shape)
    m = NRN_a(features=features, K_neighbor=K, depth=2, d_model=128, n_heads=4)

    out_x, loss = m(in_x)
    print(out_x.shape)
    o = out_x.squeeze(1).reshape(-1, features)
    print(o.shape)
    