from pathlib import Path
import torch
from modules import CombineEmbedder
from SelectTree import CustomNodeSelector
from utils import get_data
import pandas as pd
from argparse import ArgumentParser
import os
import numpy as np
from test_model import make_additional_metrics, add_mean
from pyscipopt import Model


@torch.inference_mode()
def main(checkpoint_file: str, problem_dir: str, cont_csv: bool):
    dct = torch.load(checkpoint_file)
    cfg = dct["config"]
    params = dct["weights"]
    NN = CombineEmbedder(cfg.model.features,
                         cfg.model.hidden_dim, depth=cfg.model.depth,n_layers=cfg.model.n_layers)
    NN.load_state_dict(params)
    df = None
    if os.path.exists("results.csv") and cont_csv:
        df = pd.read_csv("results.csv")
    else:
        df = pd.DataFrame(columns=["Name","Gap Ours","Gap Base", "Nodes Ours", "Nodes Base", "Reward"])
    # skip = "tsplib_data/swiss42.tsp"
    start = True
    ressource_file = pd.read_csv(Path(problem_dir)/"info.csv")
    for i,row in ressource_file.iterrows():
        print(row)
        print(row["name"])
        model = Model()
        nodesel = CustomNodeSelector(NN, "cpu", 1.0)
        model.readProblem(row["name"])
        model.includeNodesel(nodesel, "learnt_Nodeselector",
                            "this is a reinforcement learnt tree-based node selector", 1000000, 100000000)
        model.setRealParam("limits/time", 45)
        #model.hideOutput()
        model.optimize()
        base_nodes=row["nnodes"]
        op, ret, no, r, select = get_data(
            nodesel, model, baseline_gap=row["gap"], baseline_nodes=None)
        df = df.append({"Name": row["name"], "Reward": r.sum().item(), "Gap Ours": model.getGap(
        ), "Gap Base": row["gap"], "Nodes Base": base_nodes, "Nodes Ours": model.getNTotalNodes()}, ignore_index=True)

    print(df)
    df = make_additional_metrics(df)
    #df["Nodes Base"] = pd.to_numeric(df["Nodes Base"].round(),downcast="integer")
    #df["Nodes Ours"] = pd.to_numeric(df["Nodes Ours"].round(),downcast="integer")
    df.to_csv('results.csv', index=False)


if __name__ == "__main__":
    # Create the CLI parser
    parser = ArgumentParser(
        description='Process checkpoint file and problem directory.')
    parser.add_argument('checkpoint_file', type=str,
                        help='Path to the checkpoint file')
    parser.add_argument('problem_dir', type=str,
                        help='Path to the problem directory')
    parser.add_argument('--continue_csv', action="store_true",
                        help='continues the current csv, helpful in case of SCIP crash')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.checkpoint_file, args.problem_dir, args.continue_csv)
