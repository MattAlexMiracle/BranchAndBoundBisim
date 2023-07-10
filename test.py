from ProblemCreators import make_tsplib
from pathlib import Path
import torch
from modules import CombineEmbedder
from SelectTree import CustomNodeSelector
from utils import get_data
import pandas as pd
from argparse import ArgumentParser

@torch.inference_mode()
def main(checkpoint_file:str, problem_dir : str):
    dct = torch.load(checkpoint_file)
    cfg = dct["config"]
    params = dct["weights"]
    NN = CombineEmbedder(cfg.model.features, cfg.model.hidden_dim)
    NN.load_state_dict(params)
    df = pd.DataFrame(columns=["name", "normalized_gap", "gap"])

    for item in Path(problem_dir).iterdir():
        item = "tsplib_data/swiss42.tsp"
        if "tour" in str(item) and str(item).endswith("tsp"):
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

        op, ret, no, r, select = get_data(nodesel, model, baseline_gap=baseline_gap,baseline_nodes=None)
        df.append({"name": str(item), "normalized_gap": r, "gap": model.getGap(), "nnodes baseline":base_nodes, "nnodes bnbisim":model.getNNodes()},ignore_index=True)
        del model

    print(df)
    df.to_csv('results.csv', index=False)

if __name__ == "__main__":
    # Create the CLI parser
    parser = ArgumentParser(description='Process checkpoint file and problem directory.')
    parser.add_argument('checkpoint_file', type=str, help='Path to the checkpoint file')
    parser.add_argument('problem_dir', type=str, help='Path to the problem directory')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.checkpoint_file, args.problem_dir)