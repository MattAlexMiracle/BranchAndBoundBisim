from ProblemCreators import make_tsplib, read_lp
from pathlib import Path
import torch
from modules import CombineEmbedder
from SelectTree import CustomNodeSelector
from utils import get_data
import pandas as pd
from argparse import ArgumentParser
import os
import numpy as np


def node_efficiency(our_nodes, our_gap, their_nodes, their_gap):
    # for 1/gap bigger is better
    # for nodes smaller is better 
    # try nodes/(1/gap) = nodes*gap
    # our_gap, our_nodes, their_gap, their_nodes
    ours  = (our_gap)/our_nodes
    theirs= (their_gap)/their_nodes
    return (theirs-ours)/(np.maximum(theirs,ours)+1e-8) #(e1-e2) / np.maximum(e1, e2)


def relative_utility(base, ours):
    s = base - ours
    mx = np.maximum(ours.values, base.values) + 1e-5
    return s / mx


def relative_utility_per_node(base, ours, nnodes_ours, nnodes_base):
    # this flips the preference: higher values here are better, meaning we need to flip the subtraction
    s = base / nnodes_base - ours / nnodes_ours
    mx = np.maximum(ours.values / nnodes_ours, base.values / nnodes_base) + 1e-5
    return s / mx


def make_additional_metrics(df):
    df["Utility"] = relative_utility(base=df["Gap Base"], ours=df["Gap Ours"])
    df["Utility/Node"] = node_efficiency(
        their_gap=df["Gap Base"],
        our_gap=df["Gap Ours"],
        our_nodes=df["Nodes Base"],
        their_nodes=df["Nodes Ours"],
    )
    return df


def add_mean(df, decimals=3):
    t = df.round(decimals=decimals).mean()
    t["Gap Ours"] = np.nan
    t["Gap Base"] = np.nan
    t["Name"] = "Mean"
    df = df.append(t, ignore_index=True)
    return df


@torch.inference_mode()
def main(checkpoint_file: str, problem_dir: str, cont_csv: bool):
    dct = torch.load(checkpoint_file)
    cfg = dct["config"]
    params = dct["weights"]
    NN = CombineEmbedder(
        cfg.model.features, cfg.model.hidden_dim, depth=cfg.model.depth, n_layers=cfg.model.n_layers
    )
    NN.load_state_dict(params)
    df = None
    if os.path.exists("results.csv") and cont_csv:
        df = pd.read_csv("results.csv")
    else:
        df = pd.DataFrame(
            columns=["Name", "Gap Ours", "Gap Base", "Nodes Ours", "Nodes Base", "Reward"]
        )
    # skip = "tsplib_data/swiss42.tsp"
    start = True
    for item in Path(problem_dir).iterdir():
        if str(item.stem) in df["Name"].values:
            print("already done", str(item), "skipping now")
            continue
        if not start:
            print(str(item))
            continue
        if "tour" in str(item) or not (
            str(item).endswith("tsp") or str(item).endswith("lp") or str(item).endswith("mps")
        ):
            continue
        print(f"\n\n\n\n\n\nRUNNING THE BASELINE {item}\n\n\n\n\n\n")
        if str(item).endswith("tsp"):
            model = make_tsplib(str(item))
        else:
            model = read_lp(str(item))
        if model is None:
            continue
        model.setRealParam("limits/time", 45)
        model.optimize()
        baseline_gap = model.getGap()
        base_nodes = model.getNTotalNodes()
        if base_nodes < 5:
            df = df.append(
                {
                    "Name": str(item.stem),
                    "Reward": float("NaN"),
                    "Gap Ours": float("NaN"),
                    "Gap Base": float("NaN"),
                    "Nodes Base": float("NaN"),
                    "Nodes Ours": float("NaN"),
                },
                ignore_index=True,
            )
            continue
        del model
        ################################
        print(f"\n\n\n\n\n\nRUNNING THE CUSTOM {item}\n\n\n\n\n\n")
        if str(item).endswith("tsp"):
            model = make_tsplib(str(item))
        else:
            model = read_lp(str(item))
        nodesel = CustomNodeSelector(NN, "cpu", 1.0)
        model.includeNodesel(
            nodesel,
            "learnt_Nodeselector",
            "this is a reinforcement learnt tree-based node selector",
            1000000,
            100000000,
        )
        model.setRealParam("limits/time", 45)
        # model.hideOutput()
        model.optimize()
        print("exited optimization")

        op, ret, no, r, select = get_data(
            nodesel, model, baseline_gap=baseline_gap, baseline_nodes=None
        )
        df = df.append(
            {
                "Name": str(item.stem),
                "Reward": r.sum().item(),
                "Gap Ours": model.getGap(),
                "Gap Base": baseline_gap,
                "Nodes Base": base_nodes,
                "Nodes Ours": model.getNTotalNodes(),
            },
            ignore_index=True,
        )
        del model
        df.to_csv("results.csv", index=False)

    print(df)
    df = make_additional_metrics(df)
    
    # df["Nodes Base"] = pd.to_numeric(df["Nodes Base"].round(),downcast="integer")
    # df["Nodes Ours"] = pd.to_numeric(df["Nodes Ours"].round(),downcast="integer")
    df.to_csv("results.csv", index=False)


def output(filename, outname):
    
    csv = pd.read_csv(filename)
    tmp = csv.loc[csv["Nodes Base"] > 5]
    df = add_mean(make_additional_metrics(tmp).sort_values("Name")).round(3)
    print(df)
    print("percentage winning",np.mean(tmp["Gap Ours"]<=tmp["Gap Base"]))
    fl = tmp.loc[np.logical_and(tmp["Gap Ours"]<1e10,tmp["Gap Base"]<1e10)]
    print("mean without infinites", fl.mean())
    df["Name"] = df["Name"].apply(lambda x: x.replace("_","\\_"))
    df["Nodes Base"] = df["Nodes Base"].astype(int)
    df["Nodes Ours"] = df["Nodes Ours"].astype(int)
    # switch to scientific notation for long tables, i.e. MINLPLIB and MIPLIB
    use_exp = len(df)>=100
    
    if use_exp:
        with pd.option_context('display.float_format',lambda x: "$\\infty$" if x > 10e10 else ("-$\\infty$" if x< -10e10 else "{:,.3f}".format(x))):
            df.to_latex(outname,index=False, longtable=True, escape=False, na_rep="---")
    else:
        pd.options.display.float_format = '{:,.3f}'.format
        df.to_latex(outname,index=False)

if __name__ == "__main__":
    # Create the CLI parser
    parser = ArgumentParser(description="Process checkpoint file and problem directory.")
    parser.add_argument("checkpoint_file", type=str, help="Path to the checkpoint file")
    parser.add_argument("problem_dir", type=str, help="Path to the problem directory")
    parser.add_argument(
        "--continue_csv",
        action="store_true",
        help="continues the current csv, helpful in case of SCIP crash",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.checkpoint_file, args.problem_dir, args.continue_csv)
