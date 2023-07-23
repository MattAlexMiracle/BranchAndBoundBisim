from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Tuple, List, Dict,Any, Set
from enum import Enum
from torch import nn
from tqdm import trange
from time import time
import numpy as np
from time import sleep
import torch.multiprocessing as mp
from copy import deepcopy
from numba import njit, int32
from collections import deque

def _build_index_list(nested_list):
    result = []
    index = 0
    for lst in nested_list:
        result.append(list(range(index, index+len(lst))))
        index += len(lst)
    return result



@torch.no_grad()
def get_embeddable(tree) -> Tuple[List[List[List[int]]], List[torch.Tensor], List[int]]:
    stack = deque([tree])
    lsID = []
    raw_feats = []
    ls = []

    while stack:
        node = stack.popleft()
        li = node.leftNode.uid if node.leftNode else -1
        ri = node.rightNode.uid if node.rightNode else -1
        lsID.append([[li,ri]]) 
        raw_feats.append(node.features)
        ls.append(node.uid)

        if node.leftNode is not None:
            stack.insert(0,node.leftNode)

        if node.rightNode is not None:
            stack.insert(0,node.rightNode)
    return lsID, raw_feats, ls


def add_node(tree, new_node : BinaryNetworkTree, parent_treeid : int):
    stack = deque([tree])
    while stack:
        # explore breadth-first to maximize pruning!
        node = stack.popleft()

        if node.leftNode is not None:
            stack.append(node.leftNode)
        if node.rightNode is not None:
            stack.append(node.rightNode)
        if node.tree_id == parent_treeid:
            if node.leftNode is None:
                node.leftNode = new_node
            else:
                node.rightNode = new_node
            return

def assign_embeddings(tree, weight: torch.Tensor,values: torch.Tensor, uids: Dict[int,int]) -> None:
    stack = deque([tree])
    while stack:
        node = stack.popleft()
        if node.leftNode is not None:
            stack.insert(0,node.leftNode)
        if node.rightNode is not None:
            stack.insert(0,node.rightNode)
        # this maps the uid to the actual index in the <weight> and <value> tensor
        idx = uids[node.uid]
        #if idx is not None:
        node.weight = weight[idx]
        node.value = values[idx]

@torch.jit.script
def propergate_tree(nodes:torch.LongTensor,all_weights:torch.Tensor,all_parents:torch.LongTensor,all_values:torch.Tensor,all_sizes:torch.Tensor):
    ps = torch.zeros_like(nodes,dtype=torch.float32)
    vs = torch.zeros_like(nodes,dtype=torch.float32)
    pathlengths = torch.ones_like(nodes)
    while torch.any(nodes>0):
        mask = nodes > 0
        #print(nodes[mask],nodes, all_weights.shape, all_values.shape)
        ps[mask] = ps[mask] + all_weights[nodes[mask]]
        vs[mask] = vs[mask] + all_values[nodes[mask]]
        pathlengths[mask] = pathlengths[mask]+1
        nodes[mask] = all_parents[nodes[mask]]
    ps = ps/pathlengths    
    vs = vs/pathlengths
    return ps,vs

def get_prob(tree:BinaryNetworkTree,open_nodes: List[int]) -> Tuple[Dict[int,torch.Tensor],Dict[int,torch.Tensor]]:
    nodes = torch.tensor(open_nodes)
    # use dictionaries already padded to the maximum size:
    # if we don't do this, we get into index-trouble as soon as a subtree is pruned
    z = torch.zeros(1)
    all_weights = {k:z for k in range(max(open_nodes))}
    all_parents = {k:0 for k in range(max(open_nodes))}
    all_values = {k:z for k in range(max(open_nodes))}
    all_sizes = {k:0 for k in range(max(open_nodes))}
    all_parents[1] = 1
    stack = deque([tree])
    while stack:
        node = stack.popleft()
        all_weights[node.tree_id] = node.weight
        all_values[node.tree_id] = node.value
        all_sizes[node.tree_id] = node.size()
        if node.leftNode is not None:
            all_parents[node.leftNode.tree_id] =  node.tree_id
        if node.rightNode is not None:
            all_parents[node.rightNode.tree_id] =  node.tree_id
        
        if node.leftNode is not None:
            stack.insert(0,node.leftNode)
        if node.rightNode is not None:
            stack.insert(0,node.rightNode)
    #print("w0", all_weights)
    all_weights = torch.cat([v for key, v in sorted(all_weights.items())])
    
    all_parents = torch.tensor([v for key, v in sorted(all_parents.items())])
    all_values = torch.cat([v for key, v in sorted(all_values.items())])
    all_sizes = torch.tensor([v for key, v in sorted(all_sizes.items())])
    # subtract one since SCIP starts counting at one for the tree-ids
    ps, vs = propergate_tree(nodes-1, all_weights, all_parents-1, all_values,all_sizes)
    """while out:
        parent, node = out.popleft()
        t00 = time()
        mask = nodes == node.tree_id
        l = node.leftNode.sum_cache if node.leftNode else 0
        r = node.rightNode.sum_cache if node.rightNode else 0
        node.sum_cache = node.weight + l + r
        ps[mask] = ps[mask] + node.sum_cache/node.size()
        nodes[mask] = parent.tree_id
        vs[mask] = vs[mask] + node.value
        t0 +=time()-t00"""
    #print("full iteration",time()-t0)
    probdict = dict(zip(open_nodes,torch.log_softmax(ps,-1)))
    vdict = dict(zip(open_nodes,vs))
    return probdict, vdict


@dataclass
class BinaryNetworkTree:
    leftNode: BinaryNetworkTree | None
    rightNode: BinaryNetworkTree | None
    features: torch.Tensor
    info: Dict[str, Any]
    value: torch.Tensor
    uid: int
    tree_id: int
    weight : torch.Tensor
    log_p: torch.Tensor | None = None
    device : str = "cpu"
    sum_cache : torch.Tensor | None = None
    size_cache: int | None = None
    
    """@torch.no_grad()
    def get_embeddable(self) -> Tuple[List[List[List[int]]], List[torch.Tensor], List[int]]:
        li = self.leftNode.uid if self.leftNode else -1
        ri = self.rightNode.uid if self.rightNode else -1

        ls = [self.uid]
        raw_feats = [self.features]
        lsID = [[[li, ri]]]
        if self.leftNode is not None:
            left_id, raw_f, left_uid = self.leftNode.get_embeddable()
            lsID.extend(left_id)
            ls.extend(left_uid)
            raw_feats.extend(raw_f)
        if self.rightNode is not None:
            right_id, raw_f, right_uid = self.rightNode.get_embeddable()
            lsID.extend(right_id)
            ls.extend(right_uid)
            raw_feats.extend(raw_f)
        return lsID, raw_feats, ls"""

    def assign_embeddings(self, embeddings: torch.Tensor,values: torch.Tensor, uids: List[int], indices:List[int]) -> None:
        if self.leftNode is not None:
            self.leftNode.assign_embeddings(embeddings,values, uids,indices,)
        if self.rightNode is not None:
            self.rightNode.assign_embeddings(embeddings,values, uids,indices,)

        if self.uid in uids:
            idx = indices[uids.index(self.uid)]
            self.weight = embeddings[idx]
            self.value = values[idx]

    def get_value(self) -> torch.Tensor:
        if self.leftNode is None and self.rightNode is None:
            return self.value
        v = torch.ones(1, device=self.device) * (-1e8)
        if self.leftNode is not None:
            v = torch.maximum(self.leftNode.get_value(),v)
        if self.rightNode is not None:
            v = torch.maximum(self.rightNode.get_value(),v)
        return v
        
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

    def sample(self) -> Tuple[List[str], BinaryNetworkTree,torch.Tensor]:
        if self.leftNode is None and self.rightNode is None:
            return [], self, torch.zeros(1, device=self.device)
        o = [0,1]
        ps = [torch.zeros(1,device=self.device), torch.zeros(1,device=self.device)]
        if self.leftNode is not None:
            ps[0] = self.leftNode.log_p[0].exp() # type: ignore
        if self.rightNode is not None:
            ps[1] = self.rightNode.log_p[1].exp() # type: ignore
        ps = torch.tensor(ps)
        ps = ps/ps.sum()
        i = np.random.choice(o, p=ps.cpu().numpy())
        if i==0:
            path, node,logprob = self.leftNode.sample() # type: ignore
            log = ps[0].log()
            return ["l"]+path,node, log+logprob
        else:
            path, node,logprob = self.rightNode.sample() # type: ignore
            log = ps[1].log()
            return ["r"]+path, node, log+logprob

    def get_prob_old(self,open_nodes: List[int]|None= None) -> Tuple[Dict[int,torch.Tensor],Dict[int,torch.Tensor]]:
        if open_nodes is not None and self.leftNode is None and self.rightNode is None and self.tree_id not in open_nodes:
            return dict(),dict()
        dct = {self.tree_id: torch.zeros(1,device=self.device)}
        dct_v = {self.tree_id: self.value}
        l, lv = self.leftNode.get_prob(open_nodes) if self.leftNode is not None else (dict(), dict())
        r, rv = self.rightNode.get_prob(open_nodes) if self.rightNode is not None else (dict(), dict())

        l = {k:v + self.log_p[0] for k,v in l.items()} # type: ignore
        r = {k:v + self.log_p[1] for k,v in r.items()} # type: ignore

        lv = {k:v + self.value for k,v in lv.items()} # type: ignore
        rv = {k:v + self.value for k,v in rv.items()} # type: ignore
        dct.update(r)
        dct.update(l)
        dct_v.update(lv)
        dct_v.update(rv)
        return dct, dct_v

    def add_node(self, node : BinaryNetworkTree, parent_treeid : int) -> bool:
        if self.tree_id == node.tree_id:
            print("\n\n\nPANIC THIS SHOULD NEVER HAPPEN!!\n\n\n")
            sleep(100)
        if self.tree_id != parent_treeid:
            found = False
            if self.leftNode is not None:
                found = self.leftNode.add_node(node, parent_treeid)
            if not found and self.rightNode is not None:
                found = self.rightNode.add_node(node, parent_treeid)
            return found
        else:
            if self.leftNode is None:
                self.leftNode = node
            else:
                self.rightNode = node
            return True
        
    def get_all_numbers(self) -> List[int]:
        ls = [self.tree_id]
        if self.leftNode is not None:
            ls += self.leftNode.get_all_numbers()
        if self.rightNode is not None:
            ls += self.rightNode.get_all_numbers()
        return ls
    def contains_id(self, id: int) -> bool:
        if id == self.tree_id:
            return True
        if self.leftNode is not None and self.leftNode.contains_id(id):
            return True
        if self.rightNode is not None and self.rightNode.contains_id(id):
            return True
        return False

    def prepare_logprob(self,temperature : float, legal_ids:List[int]) -> None:
        l,r, lsz, rsz = None,None, 1,1
        if self.log_p is not None:
            # in this case we are already done!
            return
        if self.leftNode is not None:
            l = self.leftNode.sum_logprob(legal_ids)
            lsz = self.leftNode.size()+1
            self.leftNode.prepare_logprob(temperature, legal_ids)
        if self.rightNode is not None:
            r = self.rightNode.sum_logprob(legal_ids)
            rsz = self.rightNode.size()+1
            self.rightNode.prepare_logprob(temperature, legal_ids)
        # if l is None and r is None:
        #    l = self.weight
        #    r = self.weight
        if l is None:
            l = torch.ones(1,device=self.device)
        if r is None:
            r = torch.ones(1,device=self.device)
        self.log_p = torch.log_softmax(torch.cat([l/lsz,r/rsz])/temperature,-1)
        
    def sum_logprob(self, legal_ids:List[int]) -> torch.Tensor:
        if self.sum_cache is not None:
            return self.sum_cache
        # remove useless leaves
        if self.rightNode is None and self.leftNode is None and self.tree_id not in legal_ids:
            s = torch.tensor([-float("inf")],device=self.device)
            self.sum_cache = s.to(self.device)
            return s
            
        l = self.leftNode.sum_logprob(legal_ids) if self.leftNode else torch.zeros(1, device=self.device)
        r = self.rightNode.sum_logprob(legal_ids) if self.rightNode else torch.zeros(1, device=self.device)
        if l == -float("inf"):
            s = r + self.weight
        elif r == -float("inf"):
            s = l + self.weight
        else:
            s=l+r+self.weight
        self.sum_cache = s
        return s
    
    def prune_closed_branches(self,open_nodes: List[int]) -> bool:
        if self.leftNode is not None:
            if self.leftNode.prune_closed_branches(open_nodes):
                #print("pruned successfully!")
                self.leftNode = None            
        if self.rightNode is not None:
            if self.rightNode.prune_closed_branches(open_nodes):
                #print("pruned successfully!")
                self.rightNode = None
        # now check if self is in open nodes
        if self.tree_id in open_nodes:
            return False
        elif self.leftNode is None and self.rightNode is None:
            return True
        return False

    @torch.no_grad()
    def reset_caches(self) -> None:
        stack = deque([self])
        #self.embeddedFeatures = torch.zeros_like(self.embeddedFeatures)
        #self.combinedEmbeddings = torch.zeros_like(self.combinedEmbeddings)
        while stack:
            node = stack.popleft()
            node.log_p = None#torch.zeros(2,device=self.device)
            node.weight = torch.zeros(1)
            node.value = torch.zeros(1)
            node.sum_cache = None
            node.size_cache = None
            if node.leftNode is not None:
                #self.leftNode.reset_caches()
                stack.insert(0, node.leftNode)
            if node.rightNode is not None:
                stack.insert(0, node.rightNode)

    def size(self) -> int:
        if self.size_cache != None:
            return self.size_cache
        left_size = self.leftNode.size() if self.leftNode is not None else 0
        right_size = self.rightNode.size() if self.rightNode is not None else 0
        self.size_cache= left_size+right_size+1
        return self.size_cache
    
    def set_device(self,device):
        self.device = device
        if self.leftNode is not None:
            self.leftNode.set_device(device)
        if self.rightNode is not None:
            self.rightNode.set_device(device)

def to_dict(node) -> Dict[str,Any]:
    d = dict()
    d["features"] = node.features.numpy()
    #d["info"] = self.info
    d["tree_id"] = node.tree_id
    d["left"] = to_dict(node.leftNode) if node.leftNode is not None else None
    d["right"] = to_dict(node.rightNode) if node.rightNode is not None else None
    return d

def from_dict(d: Dict[str, Any]) -> BinaryNetworkTree|None:
    if d is None:
        return None
    tree = BinaryNetworkTree(None,None,torch.zeros(1),dict(),torch.zeros(0), uid=0,tree_id=0, weight=torch.zeros(1))
    tree.features = torch.from_numpy(d["features"])
    #tree.info = d["info"]
    tree.tree_id = d["tree_id"]
    tree.leftNode = from_dict(d["left"])
    tree.rightNode = from_dict(d["right"])
    return tree
    
def tree_from_indices(tree:BinaryNetworkTree | None, indices:List[int]) -> BinaryNetworkTree | None:
    if tree is None:
        return None
    new_tree = None
    if tree.tree_id in indices:
        new_tree = deepcopy(tree)
        new_tree.leftNode = tree_from_indices(new_tree.leftNode, indices)
        new_tree.rightNode = tree_from_indices(new_tree.rightNode, indices)
    else:
        return None



class TreeBatch:
    def __init__(self, trees: List[BinaryNetworkTree], device="cpu") -> None:
        self.trees = trees
        #for tree in self.trees:
        #    tree.set_device(device)
        self.device = device
        self.assign_uids()

    def assign_uids(self) -> None:
        uid = 0
        for tree in self.trees:
            uid = tree.set_uid(uid)+1

    def assign_embeddings(self, embeddings: torch.Tensor, values:torch.Tensor, uids: List[List[int]]) -> None:
        # this allows for splitting of the index space
        indices = _build_index_list(uids)
        for uid,ind,tree in zip(uids,indices,self.trees):
            transl = dict(zip(uid,ind))
            assign_embeddings(tree, embeddings,values, transl)

    def get_embeddable(self) -> Tuple[torch.Tensor,torch.Tensor, List[List[int]]]:
        id_map : List[np.ndarray] = []
        uids = []
        raw_feats = []
        for tree in self.trees:
            i_map, r, id = get_embeddable(tree)
            id_map.extend(i_map)
            uids.append(id)
            raw_feats.extend(r)
        id_map = torch.tensor(id_map,dtype=int)
        return id_map, torch.stack(raw_feats), uids

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
    
    def embeddings(self, combineEmb:nn.Module, temperature, legal_ids : List[List[int]]):
        # make sure to reset all buffers before doing this:
        #t = time()
        self.reset_caches()
        id_map, raw_feats, uids = self.get_embeddable()
        #print("got embeddables",time()-t)
        #t=time()
        uids_flat = torch.LongTensor([item for sublist in uids for item in sublist])
        _,w,values = combineEmb(raw_feats.to(self.device), uids_flat.to(self.device), id_map.to(self.device))
        #print("embeddings done",time()-t)
        #print("values mean",values.mean(), "w mean", w.mean(), "w std", w.std())
        #self.assign_embeddings(comb, uids.copy(), FeatureType.combinedEmbedding,)
        #t = time()

        self.assign_embeddings(w,values,uids)
        # indices = _build_index_list(uids)
        # for tree, uid, ind, op in zip(self.trees, uids, indices, legal_ids):
        #     transl = dict(zip(uid,ind))
        #     get_prob_assign(tree, op, w, values, transl)

        #print("assign time",time()-t)
        #t=time()
        #self.prepare_logprob(temperature,legal_ids)
        #print("prepare_logprob time",time()-t)
        

    def get_value(self,):
        t  = []
        for tree in self.trees:
            t.append(tree.get_value())
        return torch.stack(t)

    def prepare_logprob(self, temperature : float, legal_ids : List[List[int]]) -> None:
        for tree, l in zip(self.trees,legal_ids):
            prepare_logprob(tree,temperature, l)
    
    def __len__(self) -> int:
        return len(self.trees)
    
    def sample_batch(self) -> Tuple[List[List[str]],List[BinaryNetworkTree],torch.Tensor]:
        paths:List[List[str]]= []
        log_ps:List[torch.Tensor] = []
        nodes = []
        for tree in self.trees:
            p,node,l = tree.sample()
            paths.append(p)
            log_ps.append(l)
            nodes.append(node)
        return paths, nodes, torch.stack(log_ps)
    
    def reset_caches(self) -> None:
        for tree in self.trees:
            tree.reset_caches()

    def get_sizes(self) -> List[int]:
        ls = []
        for tree in self.trees:
            ls.append(tree.size())
        return ls

    def get_logprob(self, actions: List[int], open_nodes : List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out_log = []
        out_q = []
        out_v = []
        entropy = []
        for i,tree,o in zip(actions,self.trees, open_nodes):
            logit, q = get_prob(tree,o)
            out_log.append(logit[i])
            l = torch.stack(list(logit.values()))
            entropy.append(-(l*l.exp()).sum())
            out_q.append(q[i])
            out_v.append(torch.max(torch.stack(list(q.values()))))
        return torch.stack(out_log), torch.stack(out_q), torch.stack(out_v), torch.stack(entropy)

if __name__ == '__main__':
    # torch.set_float32_matmul_precision('high')
    device = 'cuda'
    sz = 2_000
    # trees = []
    # for _ in range(128):
    #     trees.append(__make_random_tree(0.75,device=device))
    # trees = TreeBatch(trees, device=device)
    #feat,uids = trees.get_flat(type=FeatureType.Feature)
    #print(feat.shape)
    
    # now run some test with NNs:
    from modules import FeatureEmbedder, CombineEmbedder, NaiveCombineEmbedder
    #featureEmbedder = FeatureEmbedder(128,1024)
    #featureEmbedder.to(device)
    #combineEmbedder = NaiveCombineEmbedder(1024,0.99)
    combineEmbedder = CombineEmbedder(1024)
    print(combineEmbedder.forward)
    combineEmbedder.to(device)
    # combineEmbedder = torch.compile(combineEmbedder, mode="reduce-overhead")
    
    optim = torch.optim.AdamW(combineEmbedder.parameters(),3e-4, weight_decay=0.01)
    # 1. embedd all features:
    #trees.embeddings(featureEmbedder,combineEmbedder)
    #print([x.log_p for x in trees])

    # try to optimize "find longest path" policy as a test
    #tsz = trees.get_sizes()
    #print("tree sizes", tsz)
    #print("expected size of trees", torch.mean(torch.tensor(tsz).float()))
    reward = []
    for t in trange(100):
        trees = []
        for _ in range(1):
            tmp = grow_tree(sz,device=device)
            trees.extend(tmp)
        indices = torch.randperm(len(trees))[:128]
        trees = [trees[i] for i in indices]
        trees = TreeBatch(trees, device=device)


        print("start embed and sample")
        t0=time()
        with torch.autocast("cuda"):
            trees.embeddings(combineEmbedder)
            paths,nodes, log_ps = trees.sample_batch()
            print("end embed and sample",time()-t0)
            #print(f"round {t}: paths",paths)
            #print(f"round {t}: log p", log_ps)
            # this is our "reward"
            counts = torch.tensor([len(x) for x in paths]).float()
            reward.append(counts)
            counts= counts.to(device)
            #print(t,"current reward", counts.mean(),"±", counts.std())
            print("mean p", log_ps.mean(),"current reward", counts.mean(),"±", counts.std())
            # use policy gradient
            t0 = time()
            loss = -torch.mean(log_ps * (counts-trees.get_value().detach())) + torch.mean((trees.get_value() - counts)**2)
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


