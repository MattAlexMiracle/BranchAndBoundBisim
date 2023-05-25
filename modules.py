import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, List, Dict, Any, Callable
import numpy as  np

class FeatureEmbedder(nn.Module):
    def __init__(self, feature_in, feature_embed_out: int):
        super().__init__()
        self.embd = nn.Sequential(
            nn.BatchNorm1d(feature_in),
            nn.Linear(feature_in, feature_embed_out),
            nn.LeakyReLU(),
            nn.Linear(feature_embed_out, feature_embed_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # since at the beginning each node is independent,
        # we can still use batchnorm.
        # we can't do this later in the combiner due to
        # the correlations between nodes
        x = F.leaky_relu(self.embd(x))
        return x

def mapping(probs, steps):
    s = torch.softmax(probs, -1)
    i = torch.argmax(s,-1,keepdim=True)
    i1 = torch.minimum(i+1,torch.ones_like(i)*(s.shape[-1]-2))
    i2 = torch.maximum(i-1,torch.ones_like(i))

    v0 = torch.gather(s,1,i)
    v1 = torch.gather(s,1,i1)
    v2 = torch.gather(s,1,i2)
    vout = torch.where(v1>v2, v1, v2)
    iout = torch.where(v1>v2, i1, i2)
    weight = torch.softmax(torch.cat([vout,v0],-1),-1)
    values = torch.cat([iout,steps[i]],-1)
    weighted_sum =weight*values
    return weighted_sum.sum(-1,keepdim=True)

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
    

@torch.no_grad()
def init(x : nn.Module):
    if type(x) ==  nn.Linear:
        torch.nn.init.normal_(x.weight,0,0.01)
        torch.nn.init.constant_(x.bias, 0)
class CombineEmbedder(nn.Module):
    def __init__(self,feat_emb_sz:int, node_emb_sz: int):
        super().__init__()
        self.node_emb_sz = node_emb_sz
        self.feat_emb_sz = feat_emb_sz
        #self.embd = nn.Sequential(nn.Linear(feature_emb_in+2*node_emb_sz, node_emb_sz),
        #                          nn.LeakyReLU(),
        #                          nn.BatchNorm1d(node_emb_sz)
        #                          )
        self.node_emb = nn.Sequential(
            nn.LayerNorm(node_emb_sz),
            nn.Linear(node_emb_sz,2*node_emb_sz),
            SwiGLU(),
            nn.Linear(node_emb_sz,node_emb_sz),
            nn.LeakyReLU(),
            )
        #self.feat_emb_v = FeatureEmbedder(self.feat_emb_sz,node_emb_sz)
        #self.node_emb_v = nn.Sequential(
        #    nn.Linear(node_emb_sz,node_emb_sz),
        #    nn.LeakyReLU(),
        #    nn.Linear(node_emb_sz,node_emb_sz),
        #    nn.LeakyReLU(),
        #    )
        self.feat_emb = FeatureEmbedder(self.feat_emb_sz,node_emb_sz)
        self.weight = nn.Sequential(
            nn.LayerNorm(node_emb_sz),
            nn.Linear(node_emb_sz,2*node_emb_sz),
            SwiGLU(),
            nn.Linear(self.node_emb_sz,1)
        )
        #self.bn_v = nn.LayerNorm(2*self.node_emb_sz)
        self.value_head = nn.Sequential(
            nn.LayerNorm(node_emb_sz),
            nn.Linear(node_emb_sz,node_emb_sz*2),
            SwiGLU(),
            nn.Linear(node_emb_sz,1),
            nn.Tanh()
        )
        t = np.geomspace(0.01,10.0,64)
        self.codebook = torch.from_numpy(np.concatenate([-t[::-1],t]))
        #torch.nn.init.normal_(self.weight.weight, 0,0.01)
        #torch.nn.init.constant_(self.weight.bias, 0)
        #self.node_emb.apply(init)
        self.weight.apply(init)
        #self.feat_emb.apply(init)
        #self.value_head.apply(init)

    def forward(self, raw_feats : torch.Tensor, uids : torch.LongTensor, id_map : torch.LongTensor) -> Tuple[torch.Tensor,torch.Tensor, torch.Tensor]:
        indices_sorted = torch.argsort(uids, dim=0)
        
        raws = raw_feats[indices_sorted]
        sorted_feats = torch.cat([raws, torch.zeros((1,self.feat_emb_sz), device=raw_feats.device)])
        # now embedd them:
        sorted_feats = self.feat_emb(sorted_feats)
        
        # this is also used for the fixed input features, but not with the extra "no neighbor" feature
        inital_feat = sorted_feats[:-1].clone()

        ## value stuff
        """feats_v = self.feat_emb_v(sorted_feats)
        feats_v_init = feats_v[:-1].clone()"""

        for _ in range(6):
            # 1 retrieve the relevant features using id_map
            feats_l,feats_r = torch.chunk(sorted_feats[id_map].reshape(raw_feats.shape[0],-1),2,-1)
            feats_l = self.node_emb(feats_l)
            feats_r = self.node_emb(feats_r)
            feats = (feats_r + feats_l+ inital_feat)/3
            # print("feats",feats.shape, id_map.shape, inital_feat.shape)
            new = sorted_feats[:-1] + feats 
            # we ignore the first entry since that is simply the "no neighbor" case
            sorted_feats[:-1] = new
            #####################
            ##### value stuff ###
            #####################
            """feats = feats_v[id_map].reshape(raw_feats.shape[0],-1)
            feats = self.node_emb_v(self.bn_v(feats))
            # print("feats",feats.shape, id_map.shape, inital_feat.shape)
            new = (feats_v_init + feats + feats_v_init[:-1])/3
            # we ignore the first entry since that is simply the "no neighbor" case
            feats_v[:-1] = new"""

        # undo the sorting and remove the synthetic "no neighbor" node
        x = sorted_feats[:-1][uids]
        w = self.weight(x)
        v  = self.value_head(x)
        #v  = mapping(self.value_head(x),self.codebook)
        return x, w, v


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