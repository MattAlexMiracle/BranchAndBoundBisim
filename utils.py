from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
#from SelectTree import CustomNodeSelector
from dataclasses import dataclass
from Tree import TreeBatch
from typing import List, Dict, Any
import igraph as ig
from datetime import datetime

@dataclass
class NodeData:
    open_nodes : List[List[int]]
    returns : torch.Tensor
    nodes : TreeBatch
    actions: List[List[int]]
    mask : torch.Tensor
    rewards: torch.Tensor

def get_returns(rewards:torch.Tensor, decay):
    returns = torch.zeros_like(rewards)
    for idx in range(1,len(rewards)+1):
        for r in rewards[-idx:].flip(-1):
            returns[-idx] = r + decay*returns[-idx]
    return returns

def get_data(nodesel, model, baseline_gap=None, baseline_nodes=None):
    open_nodes = nodesel.open_nodes
    if len(nodesel.nodes) == 0:
        return [], torch.Tensor(), [], torch.Tensor(), []
    nodes = nodesel.nodes
    rewards = -torch.tensor([len(i)*0 for i in open_nodes])/100
    gap = model.getGap()*10
    tmp=baseline_gap
    if tmp is None:
        tmp=0
    text = f"""
    ================================================
    Problemname: {model.getProbName()} {datetime.now().strftime("%H:%M:%S")} {model.getNNodes()}
           Baseline               NN
    GAP   {tmp*10}    {gap}
    Nodes {baseline_nodes}    {len(open_nodes)}
    ================================================
    """
    print(text)
    if baseline_gap is not None:
        #gap = gap /(baseline_gap*10 + gap+1e-8) - 0.5
        gap = gap / (10*baseline_gap+1e-8) - 1
    if baseline_nodes is not None:
        n = model.getNNodes()
        n_nodes = n/(baseline_nodes + 1e-8) -1
        gap = 1*gap + (1-1)*n_nodes
    
    rewards[-1] = 10*(rewards[-1] - np.clip(gap,-2,2))
    returns = get_returns(rewards,0.99)
    selecteds = nodesel.paths
    return open_nodes, returns, nodes, rewards, selecteds

def get_data_full_gaps(nodesel, model,):
    open_nodes = nodesel.open_nodes
    if len(nodesel.nodes) == 0:
        return [], torch.Tensor(), [], torch.Tensor(), []
    nodes = nodesel.nodes
    rewards = -torch.tensor(nodesel.gaps)*10
    gap = model.getGap()*10
    text = f"""
    ================================================
    Problemname: {model.getProbName()}
            NN
    GAP   {gap}
    Nodes {len(open_nodes)}
    ================================================
    """
    print(text)
    returns = get_returns(rewards,0.99)
    selecteds = nodesel.paths
    return open_nodes, returns, nodes, rewards, selecteds



def powernorm(val : torch.Tensor, power : float):
    return val.sign() * (val.abs()**power)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def plot_tree(dct : Dict[Any,Any], chosen: List[int],filename:str, gap:float):
    def build_tree(graph, data, parent=None):
        if "tree_id" in data.keys():
            value = str(data["tree_id"])
            graph.add_vertex(name=value)
        else:
            return
        if parent is not None:
            graph.add_edge(parent, value)
        for key, child in data.items():
            if key != "tree_id" and isinstance(child, dict):
                #print(child)
                build_tree(graph, child, parent=value)
    tree = ig.Graph()
    build_tree(tree,dct)
    tree.vs["label"] = tree.vs["name"]
    fig, ax = plt.subplots()
    layout = tree.layout("kamada_kawai")
    visual_style = {}
    visual_style["vertex_size"] = 1
    cs= []
    fig.set_figheight(100)
    fig.set_figwidth(100)
    for i in tree.vs["name"]:
        if int(i) in chosen:
            i = int(i)
            #print("chosen!", chosen.index(i)/len(chosen) )
            cs.append([0.5 + 0.5*chosen.index(i)/len(chosen), 0.0, 0.0])
        else:
            cs.append([0.0, 0.0, 1.0])
    visual_style["vertex_color"] = cs
    visual_style["bbox"] = (300, 300)
    visual_style["margin"] = 30

    ig.plot(tree,target=ax, **visual_style,)
    plt.text(0,0, f"gap: {gap}")
    plt.savefig(filename)
    plt.close(fig)
    plt.close()



def plotting(train_rewards, eval_rewards):
    fig, ((axs1,axs2),(axs3,axs4)) = plt.subplots(2,2)
    data = pd.DataFrame([(i,r) for i in range(len(train_rewards)) for r in train_rewards[i]])
    data_test = pd.DataFrame([(i,r) for i in range(len(eval_rewards)) for r in eval_rewards[i]])
    data.columns = ["index","reward"]
    data_test.columns = ["index","reward"]
    reg_train_output = data.groupby(["index"]).mean().ewm(halflife=4).mean().to_numpy().reshape(-1)
    ###
    reg_eval_output = data_test.groupby(["index"]).mean().ewm(halflife=4).mean().to_numpy().reshape(-1)

    print(data)
    axs1.set_title("pure training reward")
    sns.lineplot(data=data,x="index",y="reward", ax=axs1)
    axs2.set_title("exponential smooth training reward")
    sns.lineplot(x=np.array(range(len(reg_train_output))),y=reg_train_output, ax=axs2)
    ###
    axs3.set_title("pure eval reward")
    sns.lineplot(data=data_test,x="index",y="reward", ax=axs3)
    axs4.set_title("exponential smooth eval reward")
    sns.lineplot(x=np.array(range(len(reg_eval_output))),y=reg_eval_output, ax=axs4)
    plt.show()