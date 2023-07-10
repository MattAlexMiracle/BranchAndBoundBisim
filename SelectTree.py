from pyscipopt import Nodesel
import pyscipopt
import torch
from Tree import BinaryNetworkTree, TreeBatch, to_dict, from_dict
from copy import deepcopy
from pyscipopt import Model
from modules import CombineEmbedder
from time import sleep, time
from typing import Dict, List, Any
import numpy as np
import sys

def sample_open_nodes(nodes,logits :Dict[int,torch.Tensor]):
    ids : List[int] = [node.getNumber() for node in nodes]
    #print(ids)
    relevant_logits = sorted((k,v) for k,v in logits.items() if k in ids)
    just_logits = torch.cat([x[1] for x in relevant_logits])
    sampled = torch.distributions.Categorical(logits=just_logits).sample()
    chosen, chosen_logit = relevant_logits[sampled]
    # print("chose node", chosen, "with log-likelihood", chosen_logit, "from", just_logits.exp().sum())
    for node in nodes:
        if node.getNumber() == chosen:
            return node
        
def powernorm(val : np.ndarray, power : float):
    return np.sign(val) * (abs(val)**power)

def signed_log(val : np.ndarray):
    return np.sign(val) * (np.log(abs(val)+1e-3))


def get_model_info(model,power=0.5):
    NcutsApp = powernorm(model.getNCutsApplied(),power)
    Nsepa = powernorm(model.getNSepaRounds(),power)
    gap = model.getGap()
    # node properties
    p = []
    n=0
    vs = model.getVars()
    for v in vs:
        if v.vtype() in ["BINARY", "INTEGER", "IMPLINT"]:
            n+=1
            sol = v.getLPSol()
            p += [abs(sol  - np.floor(sol))]
    p = np.array(p).reshape(-1)
    frac_mean = np.mean(p)
    #frac_std = np.std(p)
    #frac_max = np.max(p)
    #frac_min = np.min(np.concatenate([p[p!=0],np.array([1.0])]))
    hist = np.histogram(np.concatenate([p[p!=0],np.array([1.0])]),10,range=(0,1.0), density=True)[0]
    hist = hist/hist.sum()
    # you have to be careful with using isclose for values close to zero
    # because atol can give false positives. In this case we accept this here
    already_integral = np.isclose(np.array(p),0).mean()
    cond = np.log10(model.getCondition())
    lpi = powernorm(model.lpiGetIterations(),power)
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
    #print(info,hist)

    return info, hist



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
    
    def get_tree(self, node, info : Dict[str, Any], hist: np.ndarray,power=0.5):
        self.added_ids.add(node.getNumber())
        tmp =  node.getDomchg()
        dmch = np.array([x.getNewBound() for x in tmp.getBoundchgs()] if tmp is not None else 0.0).astype(float)
        expDomch = powernorm(dmch.mean(),power)
        depth = powernorm(node.getDepth()/(self.model.getNNodes()+1),power)
        info["expDomch"] = expDomch
        info["depth_normed"] = depth
        info["n_AddedConss"] = powernorm(node.getNAddedConss(),power)
        slack_cons = []
        for c in self.model.getConss():
            if c.isLinear():
                slack_cons.append(self.model.getSlack(c))
        slack_cons = np.array(slack_cons)
        slack_cons = signed_log(slack_cons[np.logical_and(slack_cons<10**20, slack_cons>-10**20)])
        # range=(0,1.0), no range
        slack_hist = np.histogram(np.concatenate([slack_cons,np.array([1.0])]),10,density=True)[0]
        slack_hist = slack_hist/slack_hist.sum()
        
        features = torch.from_numpy(np.array(list(info.values())+ hist.tolist()+slack_hist.tolist()).clip(-10,10)).detach().half()
        new_node = BinaryNetworkTree(leftNode=None,
                rightNode=None,
                features=features,#torch.ones(128),
                info=dict(),#info,
                value=torch.zeros(1,device=self.device),
                log_p=torch.zeros(2,device=self.device),
                uid=0,
                tree_id=node.getNumber(),
                weight=torch.zeros(1,device=self.device),
                device=self.device
                )
        if node.getNumber() != 1:
            p = node.getParent().getNumber()
            self.tree.add_node(new_node, p) # type: ignore
        else:
            self.tree = new_node

    @torch.inference_mode()
    def nodeselect(self):
        t=time()
        t0=time()
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
        info,hist = get_model_info(self.model,power=power)
        for c in nodes:
            self.get_tree(c,info,hist,power=power)
        print("make features", time()-t0)
        #t0 = time()
        trees = TreeBatch([self.tree],self.device) # type: ignore
        #self.comb_model.eval()
        open_node_ids = [n.getNumber() for n in open_nodes]
        trees.embeddings(self.comb_model,self.temperature,[open_node_ids])
        # self.comb_model.train()
        #print("Time taken", time()-t0,"newly added nodes",len(nodes))
        tmp, _ = trees[0].get_prob([open_node_ids])
        t0 = time()
        self.logit_lookup = tmp #{k:v.cpu() for k,v in tmp.items()}
        node = sample_open_nodes(open_nodes,self.logit_lookup)
        self.paths.append(node.getNumber()) # type: ignore
        self.gaps.append(np.clip(self.model.getGap(),-10,10))
        self.open_nodes.append(open_node_ids)
        
        sz = self.tree.size()
        trees.reset_caches()
        self.nodes.append(to_dict(self.tree))
        # now cleanup the tree??
        if sz >100:
            self.tree.prune_closed_branches(open_node_ids)
        print("Time taken for prob", time()-t0,)
        print("total time",time()-t)
        return {"selnode": node}
    
    def nodecomp(self, node1, node2):
        #n1 = node1.getNumber()
        #n2 = node2.getNumber()
        #p1 = self.logit_lookup # type: ignore
        #p = p1[n1].exp() / (p1[n1].exp() + p1[n2].exp())
        return -1 if torch.rand(1) < 0.5 else 1


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
    