# modified from cleanrl PPO
import numpy as np
import torch
from torch import nn
from SelectTree import TreeBatch
from utils import NodeData
import wandb


def split_list_by_mask(lst, mask):
    result = []
    temp = []
    for i, val in enumerate(lst):
        temp.append(val)
        if mask[i] == 0:
            result.append(temp)
            temp = []
    result.append(temp)
    return result


@torch.inference_mode()
def calculate_advantages(rewards, values, discount_factor, gae_lambda):
    advantages = []
    advantage = 0
    next_value = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        td_error = r + next_value * discount_factor - v
        advantage = td_error + advantage * discount_factor * gae_lambda
        next_value = v
        advantages.insert(0, advantage)

    advantages = torch.tensor(advantages)

    return advantages

@torch.no_grad()
def advantages_from_list(rewards, values, mask, discount_factor, gae_lambda):
    rs = split_list_by_mask(rewards, mask)
    vs = split_list_by_mask(values, mask)
    advantages = []
    for r, v in zip(rs, vs):
        if len(r) == 0:
            continue
        advantages.append(calculate_advantages(
            r, v, discount_factor, gae_lambda))
    return torch.cat(advantages)


@torch.no_grad()
def compute_advantage(returns, vs):
    return returns - vs.detach()


@torch.inference_mode()
def get_old_data(conf, NN_old: nn.Module, batch: TreeBatch, data: NodeData):
    NN_old.eval()
    # with torch.autocast("cuda"):
    old_logprob, old_qs, old_vs, entropy_old = [],[],[],[]
    for idx,b in enumerate(batch):
        tree = TreeBatch([b])
        tree.embeddings(NN_old, 1.0, [data.open_nodes[idx]])
        o_lp, o_q, o_v, o_ent = tree.get_logprob(
            [data.actions[idx]], [data.open_nodes[idx]])
        old_logprob.append(o_lp)
        old_qs.append(o_q)
        old_vs.append(o_v)
        entropy_old.append(o_ent)
    old_logprob = torch.cat(old_logprob)
    old_qs = torch.cat(old_qs)
    old_vs = torch.cat(old_vs)
    entropy_old = torch.cat(entropy_old)
    adv = advantages_from_list(
        data.rewards, old_vs, data.mask, conf.env.decay, conf.optimization.gae)
    batch.reset_caches()
    NN_old.train()
    return old_logprob, old_qs, old_vs, entropy_old, adv


def train_ppo(NN: nn.Module, optimizer: torch.optim.Optimizer, batch: TreeBatch, data: NodeData, old_logprob, old_vs, conf,  mb_advantages=None):
    data_size = len(batch)
    b_inds = np.arange(data_size)
    # with torch.autocast("cuda"):
    batch.embeddings(NN, 1.0, data.open_nodes)

    logprob, qs, vs, entropy = batch.get_logprob(data.actions, data.open_nodes)

    logratio = logprob - old_logprob.detach()
    ratio = logratio.exp()
    print("old_vs", old_vs.mean(), "vs", vs.mean(), "logratio",ratio.mean(), "±", ratio.std(), entropy.mean())



    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
    if mb_advantages is None:
        mb_advantages = compute_advantage(data.returns, vs)
    if conf.norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()
                         ) / (mb_advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * \
        torch.clamp(ratio, 1 - conf.clip_coef, 1 + conf.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = vs
    if conf.clip_vloss:
        v_loss_unclipped = (newvalue - data.returns) ** 2
        v_clipped = old_vs.detach() + torch.clamp(
            newvalue - old_vs.detach(),
            -conf.clip_coef,
            conf.clip_coef,
        )
        v_loss_clipped = (v_clipped - data.returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - data.returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - conf.ent_coef * entropy_loss + v_loss * conf.vf_coef
    wandb.log({"train loss": loss.detach().item(), "entropy": entropy_loss.detach().item(), "v_loss": v_loss.detach().item(),
              "mean_logprob": logprob.mean().detach().item(), "std_logprob": logprob.std().detach().item(),
               "approx_kl": approx_kl.detach().item(),
               "old_approx_kl": old_approx_kl.detach().item(),
               "pg_loss" : pg_loss.detach().item(),
               "mean ratio": ratio.mean().detach().item()
               })
    if approx_kl > 0.03:
        print("emergency skip due to large kl-div", approx_kl)
        optimizer.zero_grad()
        return loss.detach().item(), approx_kl.detach().item()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(NN.parameters(), conf.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    return loss.detach().cpu().item(), approx_kl.detach().item()
