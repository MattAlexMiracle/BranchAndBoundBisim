import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, List, Dict, Any, Callable


class FeatureEmbedder(nn.Module):
    def __init__(self, feature_in, feature_embed_out: int):
        super().__init__()
        self.embd = nn.Linear(feature_in, feature_embed_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.embd(x))
        return x


class CombineEmbedder(nn.Module):
    def __init__(self, node_emb_sz: int,early_stop:float):
        super().__init__()
        self.node_emb_sz = node_emb_sz
        #self.embd = nn.Sequential(nn.Linear(feature_emb_in+2*node_emb_sz, node_emb_sz),
        #                          nn.LeakyReLU(),
        #                          nn.BatchNorm1d(node_emb_sz)
        #                          )
        self.node_emb = nn.Linear(node_emb_sz*2,node_emb_sz)
        self.feat_emb = FeatureEmbedder(128,node_emb_sz)
        self.weight = nn.Linear(node_emb_sz, 1)
        self.early_stop = early_stop

    def forward(self, x: torch.Tensor, raw_feats : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # linear layer with pseudo-skip connection
        #nodeL, nodeR = torch.chunk(x.detach(), 2, -1)
        #nodeL = nodeL.detach()
        #nodeR = nodeR.detach()
        #featT = feat.detach()
        # x = x.detach()
        #x = torch.cat([feat, nodeL, nodeR],-1)
        x = F.leaky_relu(self.feat_emb(raw_feats) + self.node_emb(x))/3
        #x = (self.embd(x) + nodeL + nodeR+feat)/4
        w = self.weight(x)
        return x, w


class NaiveCombineEmbedder(nn.Module):
    def __init__(self, node_emb_sz: int,early_stop:float):
        super().__init__()
        self.node_emb_sz = node_emb_sz
        #self.embd = nn.Sequential(nn.Linear(feature_emb_in+2*node_emb_sz, node_emb_sz),nn.BatchNorm1d(node_emb_sz))
        self.prob = nn.Linear(node_emb_sz, 1)
        self.early_stop = early_stop

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # linear layer with pseudo-skip connection
        if torch.rand(1)< self.early_stop:
            x = x.detach()
        feat, nodeL, nodeR = torch.chunk(x, 3, -1)
        #x = (feat + nodeL + nodeR)/3
        feat = torch.sigmoid(feat)
        x = feat*nodeL + (1-feat)*nodeR
        p =self.prob(x)
        return x, p