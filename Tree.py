from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Tuple, List, Dict,Any
from enum import Enum
from torch import nn
from tqdm import trange
from time import time

class FeatureType(Enum):
    Feature = 0
    embeddedFeature = 1
    combinedEmbedding = 2
    log_p = 3
    weight = 4

def _build_index_list(nested_list):
    result = []
    index = 0
    for lst in nested_list:
        result.append(list(range(index, index+len(lst))))
        index += len(lst)
    return result


@dataclass
class BinaryNetworkTree:
    leftNode: BinaryNetworkTree | None
    rightNode: BinaryNetworkTree | None
    features: torch.Tensor
    info: Dict[str, Any]
    embeddedFeatures: torch.Tensor
    combinedEmbeddings: torch.Tensor
    log_p: torch.Tensor
    uid: int
    tree_id: int
    weight : torch.Tensor
    embedded: bool = False
    device : str = "cpu"
    sum_cache : torch.Tensor | None = None

    def get_embeddable(self) -> Tuple[torch.Tensor, List[int]]:
        if self.embedded:
            return torch.tensor([]).to(self.device), []
        lE = None
        rE = None
        if self.leftNode is None or self.leftNode.embedded:
            lE = self.leftNode.combinedEmbeddings if self.leftNode is not None else torch.zeros_like(
                self.embeddedFeatures)
        if self.rightNode is None or self.rightNode.embedded:
            rE = self.rightNode.combinedEmbeddings if self.rightNode is not None else torch.zeros_like(
                self.embeddedFeatures)
        if lE is not None and rE is not None:
            # we should get here, iff either both children have embeddings, or are nonexistent
            feat = torch.cat([lE.to(self.device), rE.to(self.device), self.embeddedFeatures.to(self.device)])
            return feat.unsqueeze(0), [self.uid]

        ls = []
        lsF = []
        if self.leftNode is not None:
            left_feat, left_uid = self.leftNode.get_embeddable()
            lsF.append(left_feat.to(self.device))
            ls.extend(left_uid)
        if self.rightNode is not None:
            right_feat, right_uid = self.rightNode.get_embeddable()
            lsF.append(right_feat.to(self.device))
            ls.extend(right_uid)
        return torch.cat(lsF), ls

    def assign_embeddings(self, embeddings: torch.Tensor, uids: List[int], indices:List[int], type: FeatureType) -> None:
        if len(uids) == 0:
            # shortcut
            return
        if self.uid in uids:
            idx = indices[uids.index(self.uid)]
            if type == FeatureType.embeddedFeature:
                self.embeddedFeatures = embeddings[idx]
            if type == FeatureType.combinedEmbedding:
                self.combinedEmbeddings = embeddings[idx]
                self.embedded = True
            if type == FeatureType.log_p:
                self.log_p = embeddings[idx]
            if type == FeatureType.weight:
                self.weight = embeddings[idx]
            # now make the lookup a little bit smaller:
            i = uids.index(self.uid)
            indices = indices[:i] + indices[i+1:]
            uids = uids[:i] + uids[i+1:]
            #uids.pop(i)
        if self.leftNode is not None:
            self.leftNode.assign_embeddings(embeddings, uids,indices, type,)
        if self.rightNode is not None:
            self.rightNode.assign_embeddings(embeddings, uids,indices, type,)

    def get_flat(self, type: FeatureType) -> Tuple[torch.Tensor, List[int]]:
        own_id = [self.uid]
        own_embed: List[torch.Tensor] = list()
        if type == FeatureType.embeddedFeature:
            own_embed.append(self.embeddedFeatures.unsqueeze(0))
        elif type == FeatureType.combinedEmbedding:
            own_embed.append(self.combinedEmbeddings.unsqueeze(0))
        elif type == FeatureType.Feature:
            own_embed.append(self.features.unsqueeze(0))
        elif type == FeatureType.weight:
            own_embed.append(self.weight.unsqueeze(0))
        else:
            print("PANIC NO OPTION FITS")

        if self.leftNode is not None:
            l_embed, lid = self.leftNode.get_flat(type)
            own_id.extend(lid)
            own_embed.append(l_embed)

        if self.rightNode is not None:
            r_embed, rid = self.rightNode.get_flat(type)
            own_id.extend(rid)
            own_embed.append(r_embed)
        return torch.cat(own_embed), own_id

    def set_uid(self, uid: int) -> int:
        self.uid = uid
        if self.leftNode is not None:
            uid = self.leftNode.set_uid(uid+1)
        if self.rightNode is not None:
            uid = self.rightNode.set_uid(uid+1)
        return uid

    def get_log_action(self, action: List[str]) -> torch.Tensor:
        if action[0] == 'l':
            l = self.leftNode.get_log_action(
                action[1:]) if self.leftNode is not None else 1.0
            return self.log_p[0]+l
        elif action[0] == 'r':
            r = self.rightNode.get_log_action(
                action[1:]) if self.rightNode is not None else 1.0
            return self.log_p[1]+r
        # in case we want to consider truncated paths?
        return torch.ones(1)

    def traverse(self, fun):
        fun(self)
        if self.leftNode is not None:
            self.leftNode.traverse(fun)
        if self.rightNode is not None:
            self.rightNode.traverse(fun)

    def sample(self) -> Tuple[List[str],torch.Tensor]:
        #if torch.any(self.log_p > 0):
        #    print("warning logp misformed", self.log_p, self.uid, self.leftNode, self.rightNode)
        if torch.rand(1,device=self.log_p.device) < self.log_p[0].exp():
            if self.leftNode is not None:
                path,logprob = self.leftNode.sample()
                log = self.log_p[0]
                return ["l"]+path, log+logprob
            return [], self.log_p[0]
        else:
            if self.rightNode is not None:
                path,logprob = self.rightNode.sample()
                log = self.log_p[1]
                return ["r"]+path, log+logprob
            return [], self.log_p[1]
    
    def prepare_logprob(self) -> None:
        l,r, lsz, rsz = torch.ones(1,device=self.device),torch.ones(1,device=self.device), 1,1
        if self.leftNode is not None:
            l = self.leftNode.sum_logprob()
            lsz = self.leftNode.size()+1
            self.leftNode.prepare_logprob
        if self.rightNode is not None:
            r = self.rightNode.sum_logprob()
            rsz = self.rightNode.size()+1
            self.rightNode.prepare_logprob()
        self.log_p = torch.log_softmax(torch.cat([l/lsz,r/rsz]),-1)
        
    def sum_logprob(self) -> torch.Tensor:
        if self.sum_cache is not None:
            return self.sum_cache
        l,r =torch.zeros(1,device=self.device),torch.zeros(1,device=self.device)
        if self.leftNode is not None:
            l = self.leftNode.sum_logprob()
        if self.rightNode is not None:
            r = self.rightNode.sum_logprob()
        s=l+r+self.weight
        self.sum_cache = s
        return l+r

    def reset_caches(self) -> None:
        self.embeddedFeatures = torch.zeros_like(self.embeddedFeatures)
        self.combinedEmbeddings = torch.zeros_like(self.combinedEmbeddings)
        self.log_p = torch.zeros_like(self.log_p)
        self.weight = torch.zeros_like(self.weight)
        self.sum_cache = None
        self.embedded = False
        if self.leftNode is not None:
            self.leftNode.reset_caches()
        if self.rightNode is not None:
            self.rightNode.reset_caches()

    def size(self) -> int:
        left_size = self.leftNode.size() if self.leftNode is not None else 0
        right_size = self.rightNode.size() if self.rightNode is not None else 0
        return left_size+right_size+1

class TreeBatch:
    def __init__(self, trees: List[BinaryNetworkTree], device="cpu") -> None:
        self.trees = trees
        for tree in self.trees:
            tree.device = device
        self.device = device

    def assign_uids(self) -> None:
        uid = 0
        for tree in self.trees:
            uid = tree.set_uid(uid)+1

    def get_flat(self, type: FeatureType) -> Tuple[torch.Tensor, List[List[int]]]:
        embs = []
        ids = []
        for tree in self.trees:
            e, id = tree.get_flat(type)
            embs.append(e)
            ids.append(id)
        return torch.cat(embs), ids

    def assign_embeddings(self, embeddings: torch.Tensor, uids: List[List[int]], type: FeatureType) -> None:
        # this allows for splitting of the index space
        indices = _build_index_list(uids)
        for uid,ind,tree in zip(uids,indices,self.trees):
            tree.assign_embeddings(embeddings, uid,ind, type)

    def get_embeddable(self) -> Tuple[torch.Tensor, List[List[int]]]:
        embs = []
        ids = []
        for tree in self.trees:
            e, id = tree.get_embeddable()
            embs.append(e)
            ids.append(id)
        return torch.cat(embs), ids

    def batch_action(self, action: List[List[str]]) -> torch.Tensor:
        probs = torch.zeros(len(self.trees))
        for idx, (tree, a) in enumerate(zip(self.trees, action)):
            probs[idx] = tree.get_log_action(a)
        return probs

    def __getitem__(self,idx)->BinaryNetworkTree:
        return self.trees[idx]

    def traverse(self, fun):
        for tree in self.trees:
            tree.traverse(fun)
    
    def embeddings(self, featEmb : nn.Module, combineEmb:nn.Module) -> None:
        # make sure to reset all buffers before doing this:
        self.reset_caches()
        feat,uids = self.get_flat(type=FeatureType.Feature)
        #print("embedding features" )
        embs = featEmb(feat.to(self.device))
        #print("assigning feature embed",time()-t0)
        self.assign_embeddings(embs,uids, FeatureType.embeddedFeature)
        #print("done",time()-t0)
        #print("got feature embeddings", time()-t0)
        emb,uids = self.get_embeddable()
        while emb.shape[0] !=0:
            #print("emb shape", emb.shape)
            comb,w = combineEmb(emb)
            #print("combining emb")
            self.assign_embeddings(comb, uids.copy(), FeatureType.combinedEmbedding,)
            self.assign_embeddings(w,uids.copy(), FeatureType.weight,)
            emb,uids = self.get_embeddable()
            #print("got level", time()-t0)
        self.prepare_logprob()

    def prepare_logprob(self):
        for tree in self.trees:
            tree.prepare_logprob()
    
    def sample_batch(self) -> Tuple[List[List[str]],torch.Tensor]:
        paths:List[List[str]]= []
        log_ps:List[torch.Tensor] = []
        for tree in self.trees:
            p,l = tree.sample()
            paths.append(p)
            log_ps.append(l)
        return paths, torch.stack(log_ps)
    
    def reset_caches(self) -> None:
        for tree in self.trees:
            tree.reset_caches()

    def get_sizes(self) -> List[int]:
        ls = []
        for tree in self.trees:
            ls.append(tree.size())
        return ls

def __make_random_tree(tree_chance=0.2,maxdepth=16,device="cpu") -> BinaryNetworkTree | None:
    if torch.rand(1) < tree_chance and maxdepth >1:
        bt = BinaryNetworkTree(leftNode=__make_random_tree(tree_chance,maxdepth-1,device),
                               rightNode=__make_random_tree(tree_chance,maxdepth-1,device),
                               features=torch.ones(128,device=device)*maxdepth,
                               info=dict(),
                               embeddedFeatures=torch.zeros(1024,device=device),
                               combinedEmbeddings=torch.zeros(1024,device=device),
                               log_p=torch.zeros(2,device=device),
                               uid=0,
                               tree_id=0,
                               weight=torch.zeros(1,device=device),
                               device=device
                               )
        return bt
    return BinaryNetworkTree(leftNode=None,
                             rightNode=None,
                             features=torch.ones(128,device=device)*maxdepth,
                             info=dict(),
                             embeddedFeatures=torch.zeros(1024,device=device),
                             combinedEmbeddings=torch.zeros(1024,device=device),
                             log_p=torch.zeros(2,device=device),
                             uid=0,
                             tree_id=0,
                             weight=torch.zeros(1,device=device),
                             device=device
                             )

if __name__ == '__main__':
    device = 'cuda'
    trees = []
    for _ in range(64):
        trees.append(__make_random_tree(0.75,device=device))
    trees = TreeBatch(trees, device=device)
    trees.assign_uids()
    feat,uids = trees.get_flat(type=FeatureType.Feature)
    print(feat.shape)
    #print(uids)
    #print("as a test, write features directly to our feature embeddings")
    #trees.assign_embeddings(feat,uids, FeatureType.embeddedFeature)
    #print("tree zeros top level embeddings",trees[0].embeddedFeatures)
    #print("This should be zero",trees[0].embeddedFeatures-trees[0].features)
    # now get the embeddables:
    #emb,indices = trees.get_embeddable()
    #print("emb",emb.shape)
    #print("ind",indices)
    
    # now run some test with NNs:
    from modules import FeatureEmbedder, CombineEmbedder, NaiveCombineEmbedder
    featureEmbedder = FeatureEmbedder(128,1024)
    featureEmbedder.to(device)
    #combineEmbedder = NaiveCombineEmbedder(1024,0.99)
    combineEmbedder = torch.jit.script(CombineEmbedder(1024,1024,0.95))
    print(combineEmbedder.forward)
    combineEmbedder.to(device)
    
    optim = torch.optim.Adam(list(featureEmbedder.parameters())+list(combineEmbedder.parameters()),3e-4)
    # 1. embedd all features:
    #trees.embeddings(featureEmbedder,combineEmbedder)
    #print([x.log_p for x in trees])

    # try to optimize "find longest path" policy as a test
    tsz = trees.get_sizes()
    print("tree sizes", tsz)
    print("expected size of trees", torch.mean(torch.tensor(tsz).float()))
    reward = []
    for t in trange(40):
        trees.embeddings(featureEmbedder,combineEmbedder)
        t0=time()
        #print("start sample")
        paths, log_ps = trees.sample_batch()
        #print("end sample",time()-t0)
        #print(f"round {t}: paths",paths)
        #print(f"round {t}: log p", log_ps)
        # this is our "reward"
        counts = torch.tensor([len(x) for x in paths]).float()
        reward.append(counts)
        counts= counts.to(device)
        #print(t,"current reward", counts.mean(),"±", counts.std())
        print("mean p", log_ps.mean())
        # use policy gradient
        t0 = time()
        loss = -torch.mean(log_ps * counts)
        loss.backward()
        #print("backward done", time()-t0)
        optim.step()
        optim.zero_grad()
        #print("updated params",time()-t0)
    reward = torch.stack(reward)
    print(reward.mean(-1))
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    data = pd.DataFrame([(i,r) for i in range(len(reward)) for r in reward[i].cpu().numpy()])
    data.columns = ["index","reward"]
    sns.relplot(data=data,x="index",y="reward", kind="line",)
    plt.show()


