from ProblemCreators import make_tsplib
from pathlib import Path
import torch
from modules import CombineEmbedder
from SelectTree import CustomNodeSelector
from utils import get_data
import pandas as pd
from argparse import ArgumentParser
import os

@torch.inference_mode()
def main(checkpoint_file:str, problem_dir : str, cont_csv:bool):
    dct = torch.load(checkpoint_file)
    cfg = dct["config"]
    params = dct["weights"]
    NN = CombineEmbedder(cfg.model.features, cfg.model.hidden_dim,depth=cfg.model.depth)
    NN.load_state_dict(params)
    df = None
    if os.path.exists("results.csv") and cont_csv:
        df = pd.read_csv("results.csv")
    else:
        df = pd.DataFrame(columns=["name", "normalized_gap", "gap"])
    #skip = "tsplib_data/swiss42.tsp"
    start = True
    for item in Path(problem_dir).iterdir():
        if str(item) in df["name"].values:
            print("already done", str(item),"skipping now")
            continue
        #if skip in str(item):
        #    start = True
        if not start:
            print(str(item))
            continue
        #item = "tsplib_data/ulysses16.tsp"
        if "tour" in str(item) or not str(item).endswith("tsp"):
            continue
        print(f"\n\n\n\n\n\nRUNNING THE BASELINE {item}\n\n\n\n\n\n")
        model = make_tsplib(str(item))
        if model is None:
            continue
        model.setRealParam("limits/time", 45)
        #model.hideOutput()
        model.optimize()
        baseline_gap = model.getGap()
        base_nodes = model.getNNodes()
        del model
        ################################
        print(f"\n\n\n\n\n\nRUNNING THE CUSTOM {item}\n\n\n\n\n\n")
        model = make_tsplib(str(item))
        nodesel = CustomNodeSelector(NN, "cpu", 1.0)
        model.includeNodesel(nodesel, "custom test",
                    "just a test", 1000000, 100000000)
        model.setRealParam("limits/time", 45)
        #model.hideOutput()
        model.optimize()
        print("exited optimization")

        op, ret, no, r, select = get_data(nodesel, model, baseline_gap=baseline_gap,baseline_nodes=None)
        df = df.append({"name": str(item), "normalized_gap": r.sum().item(), "gap": model.getGap(), "baseline gap":baseline_gap, "nnodes baseline":base_nodes, "nnodes bnbisim":model.getNNodes()},ignore_index=True)
        del model
        df.to_csv('results.csv', index=False)


    print(df)
    df.to_csv('results.csv', index=False)

if __name__ == "__main__":
    # Create the CLI parser
    parser = ArgumentParser(description='Process checkpoint file and problem directory.')
    parser.add_argument('checkpoint_file', type=str, help='Path to the checkpoint file')
    parser.add_argument('problem_dir', type=str, help='Path to the problem directory')
    parser.add_argument('--continue_csv', action="store_true", help='continues the current csv, helpful in case of SCIP crash')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.checkpoint_file, args.problem_dir, args.continue_csv)