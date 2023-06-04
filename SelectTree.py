from pyscipopt import Nodesel
import pyscipopt
import torch
from Tree import BinaryNetworkTree, TreeBatch, to_dict, from_dict
from copy import deepcopy
from pyscipopt import Model
from modules import CombineEmbedder
from time import sleep, time
from typing import Dict, List
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

        self.added_ids = set()
        self.logit_lookup = dict()
        self.temperature = temperature
    
    def get_tree(self, node):
        self.added_ids.add(node.getNumber())
        power=0.5
        #age = powernorm(torch.tensor(self.model.getNNodes()),power)
        primalbound = signed_log(self.model.getPrimalbound()).clip(-10,10)
        NcutsApp = powernorm(self.model.getNCutsApplied(),power)
        Nsepa = powernorm(self.model.getNSepaRounds(),power)
        dualbound = signed_log(node.getLowerbound()).clip(-10,10)
        local_estimate = signed_log(self.model.getLocalEstimate()).clip(-10,10)
        val = signed_log(self.model.getSolObjVal(None)).clip(-10,10)
        gap = np.clip(self.model.getGap(),-10,10)
        # node properties
        depth = powernorm(node.getDepth()/(self.model.getNNodes()+1),power)
        #estimate = powernorm(torch.tensor(node.getEstimate()),power).clamp(-100,100)
        p = []
        n=0
        vs = self.model.getVars()
        for v in vs:
            if v.vtype() in ["BINARY", "INTEGER", "IMPLINT"]:
               n+=1
               sol = v.getLPSol()
               p += [abs(sol  - np.floor(sol))]
        p = np.array(p).reshape(-1)
        frac_mean = 10*np.mean(p)
        frac_std = 10*np.std(p)
        frac_max = 10*np.max(p)
        frac_min = 10*np.min(np.concatenate([p[p!=0],np.array([1.0])]))
        hist = np.histogram(np.concatenate([p[p!=0],np.array([1.0])]),10,range=(0,1.0), density=True)[0]
        # you have to be careful with using isclose for values close to zero
        # because atol can give false positives. In this case we accept this here
        already_integral = np.isclose(np.array(p),0).mean()

        constraints_added = powernorm(node.getNAddedConss(),power)
        nbranch, nconsprop, nprop = node.getNDomchg()
        domchange_branch = powernorm(nbranch,power)
        domchange_consprop = powernorm(nconsprop,power)
        domchange_prop = powernorm(nprop,power)
        cond = np.log10(self.model.getCondition()).clip(0,10)

        tmp =  node.getDomchg()
        dmch = np.array([x.getNewBound() for x in tmp.getBoundchgs()] if tmp is not None else 0.0).astype(float)
        expDomch = powernorm(dmch.mean(),power)
        stdDomch = powernorm(dmch.std(),power)
        lpi = powernorm(self.model.lpiGetIterations(),power)

        info = {
                    "NcutsApp":NcutsApp,
                    "Nsepa":Nsepa,
                    "gap": gap,
                    "lpi": lpi,
                    #"dual": dualbound,
                    #"best_local_sol": local_estimate,
                    #"val":val,
                    "depth_normed":depth,
                    #"constraints_added" : constraints_added,
                    #"domchange_branch":domchange_branch,
                    #"domchange_consprop":domchange_consprop,
                    #"domchange_prop": domchange_prop,
                    "expDomch" : expDomch,
                    #"stdDomch":stdDomch,
                    "cond": cond,
                    #"primalbound":primalbound,
                    # "active ratio" : actives,
                    "mean to integral": frac_mean,
                    "std to integral": frac_std,
                    "max to integral": frac_max,
                    "min to integral": frac_min,
                    "already_integral": already_integral
                }
        print(info,hist)
        
        features = torch.from_numpy(np.array(list(info.values())+ hist.tolist())).detach().half()
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
        leaves, children, siblings = self.model.getOpenNodes()
        open_nodes = set(leaves + children + siblings)
        if len(open_nodes)==0:
            print("no open nodes")
            return {"selnode":None}
        nodes = sorted(list(filter(lambda x: x.getNumber() not in self.added_ids, open_nodes)), key=lambda node: node.getNumber())
        if nodes is None:
            print("dumb selection")
            return {"selnode":self.model.getBestNode()}
        for c in nodes:
            self.get_tree(c)
        trees = TreeBatch([self.tree],self.device) # type: ignore
        self.comb_model.eval()
        open_node_ids = [n.getNumber() for n in open_nodes]
        #t0 = time()
        trees.embeddings(self.comb_model,self.temperature,[open_node_ids])
        self.comb_model.train()
        tmp, _ = trees[0].get_prob()
        #print("Time taken", time()-t0)
        self.logit_lookup = {k:v.cpu() for k,v in tmp.items()}
        node = sample_open_nodes(open_nodes,self.logit_lookup)
        self.paths.append(node.getNumber()) # type: ignore
        self.open_nodes.append(open_node_ids)
        trees.reset_caches()
        self.nodes.append(to_dict(self.tree))
        return {"selnode": node}
    
    def nodecomp(self, node1, node2):
        n1 = node1.getNumber()
        n2 = node2.getNumber()
        p1 = self.logit_lookup # type: ignore
        p = p1[n1].exp() / (p1[n1].exp() + p1[n2].exp())
        return -1 if torch.rand(1) < p.cpu() else 1


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
    