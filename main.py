import ray
if __name__ == '__main__':
    ray.init(object_store_memory=0.5*10**9)
from ProblemCreators import subset_sum, make_tsp, create_knapsack_instance, capacitated_facility_location, cutting_stock
import torch
from Tree import BinaryNetworkTree, TreeBatch, to_dict, from_dict
from TreeList import TreeList, Parent_Feature_Map
from utils import get_data, plot_tree
from SelectTree import CustomNodeSelector
from pyscipopt import Model
from torch import nn
from copy import deepcopy
from modules import CombineEmbedder
import time
from ray.util.multiprocessing import Pool
from torch.multiprocessing import Pool as mPool
import ray
from typing import List, Callable, Tuple
import numpy as np
from utils import NodeData
from PPO import train_ppo, get_old_data
import hydra
from tqdm import trange
from omegaconf import DictConfig, OmegaConf
import wandb
import os
import pandas as pd

def launch_models(cfg : DictConfig, pool,NN: nn.Module, csv_info : pd.DataFrame, num_proc: int) -> Tuple[List[List[int]],
                                                                                                           torch.Tensor,
                                                                                                           List[BinaryNetworkTree],
                                                                                                           List[float],
                                                                                                           List[List[int]],
                                                                                                           torch.Tensor]:
    g = torch.Generator()
    arg_list = []
    csv_indices = []
    indices = np.random.choice(len(csv_info),(num_proc,),replace=False)
    # NN_ref = ray.put(NN)
    for it,i in enumerate(indices):
        seed = g.seed()
        i = int(i)
        csv_indices.append(i)
        datum = csv_info.loc[i]
        arg_list.append((it,seed, NN, datum["name"],datum["gap"],datum["open_nodes"]))
    #for it in range(num_proc//2):
    #    seed = g.seed()
    #    i = torch.randint(0, len(model_makers), (1,))
    #    f = model_makers[i]
    #    arg_list.append((it,seed, NN_ref, f))
    result = pool.starmap(__make_and_optimize,arg_list)
    open_nodes, returns, nodes, rewards, selecteds = [], [], [], [], [],
    mask = []
    for csv_ind, res in zip(csv_indices,result):
        op, ret, no, r, select, gap = res
        last_n = no[-1]
        open_nodes.extend(op)
        returns.append(ret)
        nodes.extend(no)
        rewards.append(r)
        if len(op) > 0:
            mask.extend([1 for _ in range(len(r)-1)] + [0.0])
        selecteds.extend(select)
        if gap < csv_info.at[csv_ind,"gap"]:
            print("updated",csv_ind, "from value", csv_info.at[csv_ind,"gap"], "to value", gap)
            csv_info.at[csv_ind,"gap"] = cfg.env.harden_gaps*gap  + (1-cfg.env.harden_gaps)*csv_info.at[csv_ind,"gap"]
                
    returns = torch.cat(returns)
    #pool.terminate()
    #pool.close()
    rewards = rewards
    mask = torch.tensor(mask)
    # plot_tree(last_n, last_sel, f"figs/time-{int(time.time())}.png",sum(rewards[-1]))
    return open_nodes, returns, nodes, rewards, selecteds, mask

def __make_and_optimize(it, seed, NN, f, baseline_gap=None,baseline_nodes=None):
    #print("started")
    torch.manual_seed(seed)
    nodesel = CustomNodeSelector(NN, "cpu", 1.0)
    if isinstance(f, str):
        model = Model()
        model.readProblem(f)
    else:
        model = f()
    if baseline_gap is None:
        model.setRealParam("limits/time", 45)
        model.hideOutput()
        model.optimize()
        baseline_gap=model.getGap()
        baseline_nodes = sum([len(x) for x in model.getOpenNodes()])

    model.freeTransform()
    if not isinstance(f, str):
        model.writeProblem(f"cache/model-{it}.cip")
    with torch.inference_mode():
        model.includeNodesel(nodesel, "custom test",
                            "just a test", 1000000, 100000000)
        model.setRealParam("limits/time", 45)
        model.hideOutput()
        model.optimize()
    #print("done snd")

    op, ret, no, r, select = get_data(nodesel, model, baseline_gap=baseline_gap,baseline_nodes=baseline_nodes)
    gap = model.getGap()
    model.freeProb()
    # op, ret, no, r, select
    #no = [to_dict(n) for n in no]
    print("done converting, starting to send to main process")
    return (op, ret, no, r, select, gap)

def eval_model(pool, model, data: pd.DataFrame):
    # NN_ref = ray.put(model)
    tmp = []
    ls = list(range(len(data)))
    ls = [ls[x:x+8] for x in range(0, len(ls), 8)]
    for l in ls:
        args = []
        for i in l:
            print("RUNNING", i)
            datum = data.loc[i]
            args.append((i,-1, model, datum["name"],datum["gap"],datum["open_nodes"]))
        d = pool.starmap(__make_and_optimize,args)
        for t1, t2, t3, r, t4, t5 in d:
            tmp.append(torch.tensor(r).sum().item())
            del t1, t2, t3, t4, t5

    wandb.log({"eval reward mean": torch.tensor(tmp).mean(),"eval reward std": torch.tensor(tmp).std(), "eval reward median": np.median(np.array(tmp))})


def naive_optim(NN: nn.Module, optim: torch.optim.Optimizer, data: NodeData):
    with torch.autocast("cuda"):
        logprob = data.nodes.embeddings(NN, 1.0, data.open_nodes)
        logprob, qs, vs, ent = data.nodes.get_logprob(
            data.actions, data.open_nodes)
        q_loss = (qs[:-1] - data.returns.to(device))**2
        adv = data.returns.to(device) - vs.detach()
        value_loss = q_loss.mean()
        policy_loss = - torch.mean(adv*logprob)
        loss = policy_loss + value_loss - 0.1*ent.mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        NN.parameters(), 10, error_if_nonfinite=True)
    optim.step()
    print("value loss", value_loss.detach().cpu(), "policy loss", policy_loss.detach().cpu(
    ), "expected advantage", torch.mean(adv.detach().cpu()), "entropy", ent.detach().cpu().mean())
    optim.zero_grad()
    return loss.detach().cpu().item()

def make_test_data(num, functions):
    num = num * len(functions)
    x = np.split(np.arange(num),len(functions))
    out = []
    for idx,ns in enumerate(x):
        for seed in ns:
            print("seed",seed)
            if not os.path.exists(f"model-{seed}.cip"):
                functions[idx](int(seed))
            out.append(f"model-{seed}.cip")
    return out

def fit(cfg, NN,optim, open_nodes, returns, nodes, rewards, selecteds, mask):
    print("getting old model base")
    batch = TreeList(nodes)
    data = NodeData(open_nodes=open_nodes, returns=returns.to(cfg.device),
            nodes=batch, actions=selecteds, mask=mask, rewards=rewards)
    old_logprob, _, old_vs, _, adv = get_old_data(cfg, NN, batch, data)
    print("fitting....")
    NN.train()
    r= trange(cfg.training_scheme.update_epochs)
    for _ in r:
        with torch.no_grad():
            idx = np.random.choice(len(nodes), size=min(
                len(nodes), cfg.optimization.batchsize), replace=False)
            batch = TreeList([nodes[i] for i in idx])
            rs = torch.tensor([returns[i] for i in idx])
            act = [selecteds[i] for i in idx]
            o_batch = [open_nodes[i] for i in idx]
            o_logp = torch.tensor([old_logprob[i] for i in idx],device=cfg.device)
            o_vs = torch.tensor([old_vs[i] for i in idx],device=cfg.device)
            data = NodeData(open_nodes=o_batch, returns=rs.to(cfg.device),
                            nodes=batch, actions=act, mask=mask, rewards=rewards)
            ad_o = torch.tensor([adv[i] for i in idx],device=cfg.device)
        
        loss, kldiv = train_ppo(NN,optim,batch,data,o_logp,o_vs,cfg.training_scheme, mb_advantages=ad_o)
        batch.reset_caches()
        r.set_description(f"loss {loss}", True)
        if kldiv > 0.03:
            break
    del open_nodes, returns, nodes, rewards, selecteds, mask, batch

def train(cfg: DictConfig, NN, optim):
    #torch.multiprocessing.set_start_method("spawn")
    pool = Pool(8)
    df = pd.read_csv("training_data/info.csv")
    df_eval = pd.read_csv("eval_data/info.csv")
    #total_rewards = []
    #eval_rewards = []
    funs=[make_tsp, ]
    #test_data = make_test_data(32,funs)
    for it in range(cfg.env.num_steps):
        print("starting launch round",it)
        NN.eval()
        open_nodes, returns, nodes, rewards, selecteds, mask = launch_models(cfg,
            pool,NN, df, 8)
        r_tmp = [r.sum().item() for r in rewards]
        print(it,"rewards:", r_tmp,)
        rewards = torch.cat(rewards).detach()
        wandb.log({"train reward mean": torch.tensor(r_tmp).mean(), "train reward median": np.median(np.array(r_tmp))})
        if len(selecteds) < 10:
            print("not enough steps!!!")
            continue
        fit(cfg, NN, optim, open_nodes, returns, nodes, rewards, selecteds, mask)
        if it % 25 == 0:
            torch.save({"weights": NN.state_dict(), "config": cfg}, f"models/model-{wandb.run.id}.pt")
            eval_model(pool,NN, df_eval)

    eval_model(pool,NN, df_eval)
    torch.save({"weights": NN.state_dict(), "config": cfg}, f"models/model-{wandb.run.id}.pt")
    pool.close()
    pool.join()


#if __name__ == "__main__":
@hydra.main(version_base=None,config_path="confs", config_name="config")
def main(cfg: DictConfig):
    #ray.init(num_cpus=12)
    print(cfg)
    wandb.init(project="BnBBisim", config=OmegaConf.to_container(cfg))
    device = cfg.device
    NN = CombineEmbedder(cfg.model.features, cfg.model.hidden_dim,depth=cfg.model.depth)
    NN.to(device)
    optim = torch.optim.AdamW(NN.parameters(), cfg.optimization.lr)
    print(f"""
          =============================================================
          Num trainable parameters 
          {sum(p.numel() for p in NN.parameters() if p.requires_grad)}
          ==============================================================
          """)
    train(cfg,NN, optim)

if __name__ == "__main__":
    main()