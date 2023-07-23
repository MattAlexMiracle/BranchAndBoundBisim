from pyscipopt import Nodesel
import pyscipopt
import torch
from Tree import BinaryNetworkTree, TreeBatch, to_dict, get_prob, add_node
from copy import deepcopy
from numba import njit
from pyscipopt import Model
from time import sleep, time
from typing import Dict, List, Any, Tuple
import numpy as np
import sys

def sample_open_nodes(nodes,logits :Dict[int,torch.Tensor]):
    ids : List[int] = [node.getNumber() for node in nodes]
    #print(ids)
    relevant_logits = sorted((k,v) for k,v in logits.items() if k in ids)
    just_logits = torch.stack([x[1] for x in relevant_logits])
    sampled = torch.distributions.Categorical(logits=just_logits).sample()
    chosen, chosen_logit = relevant_logits[sampled]
    # print("chose node", chosen, "with log-likelihood", chosen_logit, "from", just_logits.exp().sum())
    for node in nodes:
        if node.getNumber() == chosen:
            return node

@njit        
def powernorm(val : np.ndarray, power : float):
    return np.sign(val) * (np.abs(val)**power)

@njit
def signed_log(val : np.ndarray):
    return np.sign(val) * (np.log(np.abs(val)+1e-3))

@njit
def make_data(vars, slack_cons):
    vars = np.abs(vars  - np.floor(vars))
    slack_cons = np.append(slack_cons[~np.isnan(slack_cons)],0)
    vars = vars[~np.isnan(vars)]

    slack_cons = signed_log(slack_cons[np.logical_and(slack_cons<10**10, slack_cons>-10**10)])

    # range=(0,1.0), no range
    slack_hist = np.histogram(slack_cons, 10,range=(np.min(slack_cons), np.max(slack_cons)+1e-8))[0]
    slack_hist = slack_hist/(slack_hist.sum()+1e-8)
    frac_mean = np.mean(vars)
    hist = np.histogram(vars[vars!=0],10,range=(0,1.0))[0]
    var_hist = hist/(hist.sum()+1e-8)
    already_integral = np.isclose(vars,0).mean()
    return slack_hist, var_hist, frac_mean, already_integral


def get_model_info(model,power=0.5):
    NcutsApp = powernorm(model.getNCutsApplied(),power)
    Nsepa = powernorm(model.getNSepaRounds(),power)
    gap = model.getGap()
    # node properties
    #t0 = time()
    vars = [v.getLPSol() for v in model.getVars() if v.vtype() in ["BINARY", "INTEGER", "IMPLINT"]]
    #n=0
    #vs = 
    slack_cons = [model.getSlack(c) for c in model.getConss() if c.isLinear()]
    #print("iterated",time()-t0)

    
    # you have to be careful with using isclose for values close to zero
    # because atol can give false positives. In this case we accept this here
    cond = np.log10(model.getCondition())
    lpi = powernorm(model.lpiGetIterations(),power)
    slack_cons = np.array(slack_cons)
    vars = np.array(vars).reshape(-1)

    slack_hist, var_hist, frac_mean, already_integral = make_data(vars, slack_cons)

    info = {
            "NcutsApp":NcutsApp,
            "Nsepa":Nsepa,
            "gap": gap,
            "lpi": lpi,
            "cond": cond,
            "mean to integral": frac_mean,
            #"std to integral": frac_std,
            #"max to integral": frac_max,
            #"min to integral": frac_min,
            "already_integral": already_integral
        }

    return info, var_hist, slack_hist

def num_in_range(ranges:List[Tuple[int,int]], mod:List[int]):
    def f(n:int) -> int:
        for idx, (r1,r2) in enumerate(ranges):
            if r1 <= n <= r2:
                return mod[idx]
        return 2**32
    return f

class CustomNodeSelector(Nodesel):
    def __init__(self, comb_model, device :str, temperature :float):
        self.comb_model = comb_model
        self.comb_model.to(device)
        self.sel_counter = 0
        self.comp_counter = 0
        self.device = device
        self.tree : BinaryNetworkTree | None = None
        self.paths = []
        self.nodes = []
        self.open_nodes = []
        self.gaps = []

        self.added_ids = set()
        self.logit_lookup = dict()
        self.temperature = temperature
        self.step = 0
        self.mods = num_in_range([(0,10), (10,100),(100,1000),(1000,10_000), (10_000,2**32)],[1,5,10,100,1000])
    
    def get_tree(self, node, info : Dict[str, Any], var_hist: np.ndarray, slack_hist : np.ndarray,power=0.5):
        #t0 = time()
        self.added_ids.add(node.getNumber())
        #print("Node id", node.getNumber())
        tmp =  node.getDomchg()
        dmch = np.array([x.getNewBound() for x in tmp.getBoundchgs()] if tmp is not None else 0.0).astype(float)
        expDomch = powernorm(dmch.mean(),power)
        depth = powernorm(node.getDepth()/(self.model.getNNodes()+1),power)
        info["expDomch"] = expDomch
        info["depth_normed"] = depth
        info["n_AddedConss"] = powernorm(node.getNAddedConss(),power)

        
        features = torch.tensor(list(info.values())+ var_hist.tolist()+slack_hist.tolist()).clamp(-10,10).detach().half()
        #print("info var slack",list(info.values()),var_hist.tolist(),slack_hist.tolist())
        new_node = BinaryNetworkTree(leftNode=None,
                rightNode=None,
                features=features,#torch.ones(128),
                info=dict(),#info,
                value=torch.zeros(1,device=self.device),
                log_p=None, #torch.zeros(2,device=self.device),
                uid=0,
                tree_id=node.getNumber(),
                weight=torch.zeros(1,device=self.device),
                device=self.device
                )
        #print("constructed features and node", time()-t0)
        if node.getNumber() != 1:
            p = node.getParent().getNumber()
            #t0 = time()
            add_node(self.tree, new_node, p) # type: ignore
            #print("appened tree", time()-t0)
        else:
            self.tree = new_node

    @torch.inference_mode()
    def nodeselect(self):
        self.step+=1
        #t=time()
        #t0=time()
        self.comb_model.eval()
        leaves, children, siblings = self.model.getOpenNodes()
        open_nodes = set(leaves + children + siblings)
        if len(open_nodes)==0:
            print("no open nodes", len(open_nodes))
            return {"selnode":self.model.getBestboundNode()}
        nodes = sorted(list(filter(lambda x: x.getNumber() not in self.added_ids, open_nodes)), key=lambda node: node.getNumber())
        if nodes is None:
            print("dumb selection")
            return {"selnode":self.model.getBestboundNode()}
        power=0.5
        info,var_hist, slack_hist = get_model_info(self.model,power=power)
        for c in nodes:
            self.get_tree(c,info,var_hist, slack_hist,power=power)
        if self.step % self.mods(self.step) != 0:
            t = self.model.getBestChild()
            if t is not None:
                # first try to work on the selected subtree
                return {"selnode":t}
            # if the subtree is solved, continue with the best bound node
            return {"selnode":self.model.getBestboundNode()}
        open_node_ids = [n.getNumber() for n in open_nodes]
        self.tree.prune_closed_branches(open_node_ids)
        
        
        #print("make features", time()-t0,len(open_nodes))
        #t0 = time()
        trees = TreeBatch([self.tree],self.device) # type: ignore
        #self.comb_model.eval()
        trees.embeddings(self.comb_model,self.temperature,[open_node_ids])
        # self.comb_model.train()
        #print("Time taken", time()-t0,"newly added nodes",len(nodes))
        #t0 = time()
        tmp, _ = get_prob(trees[0],open_node_ids)
        self.logit_lookup = tmp #{k:v.cpu() for k,v in tmp.items()}
        node = sample_open_nodes(open_nodes,self.logit_lookup)
        self.paths.append(node.getNumber()) # type: ignore
        self.gaps.append(np.clip(self.model.getGap(),-10,10))
        self.open_nodes.append(open_node_ids)
        #print("Time taken for prob", time()-t0,)        
        trees.reset_caches()
        self.nodes.append(to_dict(self.tree))
        # now cleanup the tree??

        #print("total time",time()-t, sz)
        return {"selnode": node}
    
    def nodecomp(self, node1, node2):
        #n1 = node1.getNumber()
        #n2 = node2.getNumber()
        #p1 = self.logit_lookup # type: ignore
        #p = p1[n1].exp() / (p1[n1].exp() + p1[n2].exp())
        return -1 if node1.getLowerbound() <= node2.getLowerbound() else 1
        #return -1 if torch.rand(1) < 0.5 else 1


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

if __name__ == "__main__":
    pass
    