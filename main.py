import ray
if __name__ == '__main__':
    ray.init(object_store_memory=0.5*10**9)
from ProblemCreators import subset_sum, make_tsp, create_knapsack_instance, capacitated_facility_location, cutting_stock
import torch
from Tree import BinaryNetworkTree, TreeBatch, to_dict, from_dict
from utils import get_data
from SelectTree import CustomNodeSelector
from pyscipopt import Model
from torch import nn
from copy import deepcopy
from modules import CombineEmbedder
from ray.util.multiprocessing import Pool
import ray
from typing import List, Callable, Tuple
import numpy as np
from utils import NodeData
from PPO import train_ppo, get_old_data
import hydra
from tqdm import trange
from omegaconf import DictConfig, OmegaConf
import wandb
import gc

def launch_models(pool,NN: nn.Module, temperature: float, num_proc: int, model_makers: List[Callable]) -> Tuple[List[List[int]],
                                                                                                           torch.Tensor,
                                                                                                           List[BinaryNetworkTree],
                                                                                                           List[float],
                                                                                                           List[List[int]],
                                                                                                           torch.Tensor]:
    g = torch.Generator()
    arg_list = []
    NN_ref = ray.put(NN)
    for it in range(num_proc//2):
        seed = g.seed()
        i = torch.randint(0, len(model_makers), (1,))           
        arg_list.append((it,seed, NN_ref, f"cache/model-{it}.cip"))
    for it in range(num_proc//2):
        seed = g.seed()
        i = torch.randint(0, len(model_makers), (1,))
        f = model_makers[i]
        arg_list.append((it,seed, NN_ref, f))
    result = pool.starmap(__make_and_optimize,arg_list)
    open_nodes, returns, nodes, rewards, selecteds = [], [], [], [], [],
    mask = []
    for res in result:
        # x = res.get()
        op, ret, no, r, select = deepcopy(res)
        open_nodes.extend(op)
        returns.append(ret)
        nodes.extend([from_dict(n) for n in no])
        rewards.append(r)
        if len(op) > 0:
            mask.extend([1 for _ in range(len(r)-1)] + [0.0])
        selecteds.extend(select)
    returns = torch.cat(returns)
    #pool.terminate()
    #pool.close()
    rewards = rewards
    mask = torch.tensor(mask)
    return open_nodes, returns, nodes, rewards, selecteds, mask

def __make_and_optimize(it, seed, NN, f):
    #print("started")
    torch.manual_seed(seed)
    NN = ray.get(NN)
    nodesel = CustomNodeSelector(NN, "cpu", 1.0)
    if isinstance(f, str):
        model = Model()
        model.readProblem(f)
    else:
        model = f()
    model.setRealParam("limits/time", 45)
    model.hideOutput()
    model.optimize()
    baseline_gap=model.getGap()
    baseline_nodes = sum([len(x) for x in model.getOpenNodes()])

    model.freeTransform()
    if not isinstance(f, str):
        model.writeProblem(f"cache/model-{it}.cip")

    model.includeNodesel(nodesel, "custom test",
                         "just a test", 1000000, 100000000)
    model.setRealParam("limits/time", 30)
    model.hideOutput()
    model.optimize()
    #print("done snd")

    op, ret, no, r, select = get_data(nodesel, model, baseline_gap=baseline_gap,baseline_nodes=baseline_nodes)
    model.freeProb()
    # op, ret, no, r, select
    #no = [to_dict(n) for n in no]
    print("done converting, starting to send to main process")
    return (op, ret, no, r, select)


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
            functions[idx](int(seed))
            out.append(f"model-{seed}.cip")
    return out

def fit(cfg, NN,optim, open_nodes, returns, nodes, rewards, selecteds, mask):
    print("getting old model base")
    batch = TreeBatch(nodes, device=cfg.device)
    data = NodeData(open_nodes=open_nodes, returns=returns.to(cfg.device),
            nodes=batch, actions=selecteds, mask=mask, rewards=rewards)
    old_logprob, _, old_vs, _, adv = get_old_data(cfg, NN, batch, data)
    print("fitting....")
    r= trange(cfg.training_scheme.update_epochs)
    for _ in r:
        with torch.no_grad():
            idx = np.random.choice(len(nodes), size=min(
                len(nodes), cfg.optimization.batchsize), replace=False)
            batch = TreeBatch([nodes[i] for i in idx], device=cfg.device)
            rs = torch.tensor([returns[i] for i in idx])
            act = [selecteds[i] for i in idx]
            o_batch = [open_nodes[i] for i in idx]
            o_logp = torch.tensor([old_logprob[i] for i in idx],device=cfg.device)
            o_vs = torch.tensor([old_vs[i] for i in idx],device=cfg.device)
            data = NodeData(open_nodes=o_batch, returns=rs.to(cfg.device),
                            nodes=batch, actions=act, mask=mask, rewards=rewards)
            ad_o = torch.tensor([adv[i] for i in idx],device=cfg.device)
        
        loss = train_ppo(NN,optim,batch,data,o_logp,o_vs,cfg.training_scheme, mb_advantages=ad_o)
        batch.reset_caches()
        del data
        r.set_description(f"loss {loss}", True)
    del open_nodes, returns, nodes, rewards, selecteds, mask

def train(cfg: DictConfig, NN, optim):
    #torch.multiprocessing.set_start_method("spawn")
    pool = Pool(12)
    #total_rewards = []
    #eval_rewards = []
    funs=[make_tsp, ]
    test_data = make_test_data(32,funs)
    for it in range(cfg.env.num_steps):
        print("starting launch round",it)
        open_nodes, returns, nodes, rewards, selecteds, mask = launch_models(
            pool,NN, 1.0, 12, funs)
        r_tmp = [r.sum().item() for r in rewards]
        print(it,"rewards:", r_tmp,)
        rewards = torch.cat(rewards).detach()
        wandb.log({"train reward mean": torch.tensor(r_tmp).mean()})
        if len(selecteds) < 10:
            print("not enough steps!!!")
            continue
        fit(cfg, NN, optim, open_nodes, returns, nodes, rewards, selecteds, mask)
        gc.collect()
        if it % 25 == 0:
            tmp = []
            NN_ref = ray.put(NN)
            result = pool.starmap(__make_and_optimize,[(0,0,NN_ref,t) for t in test_data])
            for d in result:
                _, _, _, r, _  = deepcopy(d)
                tmp.append(torch.tensor(r).sum().item())
            wandb.log({"eval reward mean": torch.tensor(tmp).mean(),"eval reward std": torch.tensor(tmp).std()})
    tmp = []
    NN_ref = ray.put(NN)
    for test in test_data:
        tmp.append((0,0,NN_ref, test))
    tmp = [d[-2].sum().item() for d in pool.starmap(__make_and_optimize, tmp)]
    wandb.log({"eval reward mean": torch.tensor(tmp).mean(),"eval reward std": torch.tensor(tmp).std()})


#if __name__ == "__main__":
@hydra.main(version_base=None,config_path="confs", config_name="config")
def main(cfg: DictConfig):
    #ray.init(num_cpus=12)
    wandb.init(project="BnBBisim", config=OmegaConf.to_container(cfg))
    device = cfg.device
    NN = CombineEmbedder(12, 512)
    NN.to(device)
    optim = torch.optim.AdamW(NN.parameters(), cfg.optimization.lr)
    train(cfg,NN, optim)

if __name__ == "__main__":
    main()