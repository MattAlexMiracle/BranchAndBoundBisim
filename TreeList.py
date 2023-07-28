import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from time import time
from numba import njit

@dataclass
class Parent_Feature_Map:
    uids : List[int]
    tree_ids : List[int]
    parent_ids : List[int]
    features: List[torch.Tensor]

    def tid_parents_uid_parents(self):
        ls = []
        tid = self.tree_ids#.tolist()
        for i in self.parent_ids:
            if i < 0:
                ls.append(-1)
            else:
                ls.append(self.uids[tid.index(i)])
        return ls

def add_parent_map(pm : Parent_Feature_Map, uid, tid, pid, feats):
    pm.uids.append(uid)
    pm.tree_ids.append(tid)
    pm.parent_ids.append(pid)
    pm.features.append(feats)
    #print("pmff",pm.features)
    return pm

def prune_elements(tree : Parent_Feature_Map, open_nodes:List[int]):
    initial = len(tree.features)
    counter=0
    last_change = 0
    while (counter-last_change)<len(tree.tree_ids):
        counter +=1
        i = counter % len(tree.tree_ids)
        tid = tree.tree_ids[i]
        if tid in open_nodes:
            continue
        if tid not in tree.parent_ids:
            #print(tid in tree.parent_ids,tree.parent_ids,tid)
            tree.uids.pop(i) #= np.delete(tree.uids,i)
            tree.tree_ids.pop(i) #= np.delete(tree.tree_ids,i)
            tree.parent_ids.pop(i) #= np.delete(tree.parent_ids,i)
            tree.features.pop(i)
            last_change = counter
    #print("pruned", initial - len(tree.features), "nodes")

class TreeList:
    def __init__(self, trees: List[Parent_Feature_Map]):
        self.trees = trees
        start = 0
        for tree in self.trees:
            l = len(tree.tree_ids)
            tree.uids = list(range(start, start + l))
            start = start + l
            

    def get_prob(self,combineEmb:nn.Module, open_nodes: List[List[int]]):
        children, feats, uids = [], [], []
        for t in self.trees:
            c,f,u = get_embeddable(t)
            children.extend(c)
            feats.extend(f)
            uids.extend(u)
        #print(feats)
        feats = torch.cat(feats).half()
        uids = torch.LongTensor(uids)
        children = torch.LongTensor(children)
        #print(feats.shape, children.shape, uids.shape)
        _,weights,values = combineEmb(feats, uids, children)
        pds, vds, entropy = [], [], []

        for tree, o in zip(self.trees,open_nodes):
            w1, v1 = retrieve_valuables(tree.uids, uids.tolist(), weights, values)
            #print("w1",w1.shape)
            probdict, vdict = get_prob(tree, w1.squeeze(-1), v1.squeeze(-1), o)
            l = torch.stack(list(probdict.values()))
            entropy.append(-(l*l.exp()).sum())
            #tmp = probdict[a] if a>=0 else list(probdict.values())
            pds.append(probdict)
            vds.append(max(vdict.values()))
        #print(pds)
        #print(vds)
        #print(entropy)
        #pds = torch.tensor(pds)
        vds = torch.stack(vds)
        entropy = torch.stack(entropy)
        return pds, vds, entropy
    
    def get_log_action(self,combineEmb:nn.Module, open_nodes: List[List[int]], actions: List[int]):
        pds, vds, entropy = self.get_prob(combineEmb,open_nodes)
        ps = []
        for p,a in zip(pds,actions):
            ps.append(p[a])
        return torch.stack(ps),vds,entropy

    def reset_caches(self):
        for tree in self.trees:
            for f in tree.features:
                f.grad = None
        return 


    def __getitem__(self,idx):
        return self.trees[idx]
    def __len__(self):
        return len(self.trees)

torch.jit.script    
def retrieve_valuables(uids_tree:List[int], batch_ids:List[int], weights:torch.Tensor, values:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    vs, ws = [], []
    for uid in uids_tree:
        bid = batch_ids.index(uid)
        vs.append(values[bid])
        ws.append(weights[bid])
    return torch.stack(ws), torch.stack(vs)

def parents_to_children(parent_ids:List[int], node_ids:List[int]):
    child_list = {k : [-1,-1] for k in node_ids}
    child_list[-1] = [-1,-1]
    for p, n in zip(parent_ids, node_ids):
        if child_list[p][0]<0:
            child_list[p][0] = n
        else:
            child_list[p][1] = n
    del child_list[-1]
    #print(child_list)
    return [v for v in child_list.values()]

@torch.no_grad()
def get_embeddable(tree) -> Tuple[List[List[List[int]]], List[torch.Tensor], List[int]]:
    # 1. get uid
    ls = tree.uids
    # 2. get children
    lsID = parents_to_children(tree.tid_parents_uid_parents(), ls)
    # 3. get featues
    raw_feats = tree.features
    return lsID, raw_feats, ls

@torch.jit.script
def orient_padded_tensor(ids:torch.Tensor,ten:torch.Tensor):
    # first create a zero-padded tensor
    tmp = torch.zeros(torch.max(ids).long()+1)
    tmp[ids.long()] = ten
    return tmp

@torch.jit.script
def propergate_tree(nodes:torch.LongTensor,all_weights:torch.Tensor,all_parents:torch.LongTensor,all_values:torch.Tensor):
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

def get_prob(tree:Parent_Feature_Map, weights:torch.Tensor, values:torch.Tensor,open_nodes: List[int]) -> Tuple[Dict[int,torch.Tensor],Dict[int,torch.Tensor]]:
    nodes = torch.tensor(open_nodes)
    #print("nodes", nodes.shape)
    # use dictionaries already padded to the maximum size:
    # if we don't do this, we get into index-trouble as soon as a subtree is pruned
    all_weights = weights
    all_values = values
    all_parents = torch.tensor(tree.parent_ids)
    tree_ids = torch.tensor(tree.tree_ids)-1
    #all_sizes = tree["sizes"]

    all_weights = orient_padded_tensor(tree_ids, all_weights)
    
    all_parents = orient_padded_tensor(tree_ids, all_parents.float()).long()
    all_values = orient_padded_tensor(tree_ids, all_values)
    #print(all_weights.shape, all_parents.shape, all_values.shape)
    #all_sizes = orient_padded_tensor(tree_ids, all_sizes)
    # subtract one since SCIP starts counting at one for the tree-ids
    ps, vs = propergate_tree(nodes-1, all_weights, all_parents-1, all_values)
    probdict = dict(zip(open_nodes,torch.log_softmax(ps,-1)))
    vdict = dict(zip(open_nodes,vs))
    return probdict, vdict