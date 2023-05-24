import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.net(x)

class NRN_block(nn.Module):
    def __init__(self, features, K_neighbor, feature_dim, K_dim):
        super().__init__()
        self.ntr_block = nn.Sequential(
            nn.LayerNorm(features),
            Rearrange('b n d -> b d n'),
            FeedForward(K_neighbor, K_dim),
            Rearrange('b d n -> b n d')
        )
        self.nar_block = nn.Sequential(
            nn.LayerNorm(features),
            FeedForward(features, feature_dim)
        )

    def forward(self, x):
        x = x + self.ntr_block(x)
        x = x + self.nar_block(x)
        return x
    
class NRN_block_adapter(nn.Module):
    def __init__(self, features, K_neighbor, feature_dim, K_dim):
        super().__init__()
        self.ntr_block = nn.Sequential(
            nn.LayerNorm(features),
            Rearrange('b n d -> b d n'),
            FeedForward(K_neighbor, K_dim),
            Rearrange('b d n -> b n d')
        )
        self.adapter = nn.ModuleList([])
        for _ in range(K_neighbor):
            self.adapter.append(nn.LayerNorm(features))
        
        self.nar_block = FeedForward(features, feature_dim)
        

    def forward(self, x):
        B, N, D = x.shape
        x = x + self.ntr_block(x)
        x_list = [torch.unsqueeze(t, dim=1) for t in torch.unbind(x, dim=1)]
        cat_list =[]
        for i in range(N):
            cat_list.append(self.adapter[i](x_list[i]))
        x = torch.cat(cat_list, dim=1)
        x = x + self.nar_block(x)
        return x


class distance_encoder(nn.Module):
    def __init__(self, features, d_model) -> None:
        super().__init__()
        distance_hidden_dim = 2 * features
        attr_hidden_dim = d_model - distance_hidden_dim
        self.attr_hidden_dim = attr_hidden_dim
        self.attr_encoder = nn.Sequential(
            nn.Linear(features, attr_hidden_dim),
            nn.GELU(),
            nn.Linear(attr_hidden_dim, attr_hidden_dim)
        )
        self.distance_embedding = nn.Sequential(
            nn.Linear(features, distance_hidden_dim, bias=False),
            nn.LayerNorm(distance_hidden_dim),
            nn.GELU(),
            nn.Linear(distance_hidden_dim, distance_hidden_dim, bias=False)
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False)
        )
        #init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
        # [B, P, D]
        B, N, D = x.shape
        attr_enc = self.attr_encoder(x)

        x_0_N = x[:, :1, :].repeat(1, N, 1)
        x_N_dist = torch.sub(x, x_0_N)
        #return enc_out + self.positional_embed(x)
        dist_emb = self.distance_embedding(x_N_dist)
        enc_out = torch.cat((attr_enc, dist_emb), dim=-1)

        enc_out = self.fc(enc_out)
        return enc_out

class NRN_m(nn.Module):
    def __init__(self, features, K_neighbor, depth, theta, drop_out=0.4, d_model=128):
        super(NRN_m, self).__init__()

        self.features = features
        self.K_neighbor = K_neighbor
        self.inputRearrange = Rearrange('b x -> b 1 x')

        # mixer
        self.drop_out = nn.Dropout(p=drop_out)
        self.encoder = distance_encoder(features, d_model)

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(NRN_block_adapter(d_model, K_neighbor, d_model * theta, K_neighbor * theta))

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            Rearrange('b n d -> b d n'),
            nn.Linear(K_neighbor, 1),
            Rearrange('b d n -> b (d n)')
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, features)
        )
        
    def forward(self, x):
        # b n*d -> b 1 n*d
        r_x = x
        x = self.inputRearrange(x)
        # b 1 n*d -> b n d
        x_k_split = torch.chunk(x, self.K_neighbor, dim=2)

        cat_list =[]
        for i in range(self.K_neighbor):
            k_i = x_k_split[i]
            if i==0:
                _x_0 = x_k_split[i]
                k_i = self.drop_out(k_i)
            cat_list.append(k_i)
        x = torch.cat(cat_list, dim=1)

        x = self.encoder(x)
        for block in self.mixer_blocks:
            x = block(x)
        # b n d -> b d

        # x = self.fc(x)
        out = self.decoder(x)

        _x_0 = _x_0.squeeze(1)
        out = out[:, :1, :].squeeze(1)
        missing_pos = (_x_0 != 0)
        _x0_masked = torch.masked_select(_x_0, missing_pos)
        _reconst_masked = torch.masked_select(out, missing_pos).reshape(-1)
        mse = torch.mean(torch.square(_x0_masked - _reconst_masked))
        return out, mse
    
if __name__ == "__main__":

    K = 5
    features = 10
    m = NRN_m(features=features, K_neighbor=K, depth=2, theta=7)
    print(m)
    in_x = torch.randn(2, K * features+K)
    print(in_x.shape)
    x_ = torch.chunk(in_x, K, dim=1)
    print(x_[0].shape)
    m = NRN_m(features=features, K_neighbor=K, depth=2, feature_dim=features*2, K_dim=2*K, theta=7)
    print(m.modules())
    out_x, loss = m(in_x)
    print(out_x.shape)
    o = out_x.squeeze(1).reshape(-1, features)
    print(o.shape)

    loss = nn.MSELoss()
    cost = loss(o, x_[0])
    