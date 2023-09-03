import pandas as pd
import os
import argparse
from ProblemCreators import *

import ray


@ray.remote(
    num_cpus=2,
)
def get_model_gap(tmp_file):
    model = Model()
    model.readProblem(tmp_file)
    model.setRealParam("limits/time", 45)
    model.optimize()
    return tmp_file, model.getGap(), model.getNTotalNodes()


def make_dataset(functions, number_of_instances, folder_name):
    data = []
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        os.mkdir(f"{folder_name}/data")
        os.mkdir(f"{folder_name}/tmp")
    for f in functions:
        buffer = []
        gap = []
        refs = []
        node_count = []
        for i in range(256):
            model = f()
            model.writeProblem(f"{folder_name}/tmp/model-{f.__name__}-{i}.cip")
            refs.append(get_model_gap.remote(f"{folder_name}/tmp/model-{f.__name__}-{i}.cip"))

        for r in refs:
            print("getting REF")
            name, gp, nnodes = ray.get(r)
            if 0.0 < gp < 1.0 and nnodes > 50:
                buffer.append(name)
                gap.append(gp)
                node_count.append(nnodes)
        for i in range(number_of_instances):
            print("instance pool has size",len(gap))
            refs = []
            if len(gap)<25:
                for j in range(16):
                    model = f()
                    model.writeProblem(f"{folder_name}/tmp/model-{f.__name__}-{j}-{i+64}.cip")
                    refs.append(
                        get_model_gap.remote(f"{folder_name}/tmp/model-{f.__name__}-{j}-{i+64}.cip")
                    )
                for r in refs:
                    name, gp, nnodes = ray.get(r)
                    if 0.0 < gp < 1.0 and nnodes > 50:
                        buffer.append(name)
                        gap.append(gp)
                        node_count.append(nnodes)
            # now get the
            if len(gap) == 0:
                continue
            gap_a = np.array(gap)
            medoid = np.argmin(np.abs(gap_a - gap_a[:, None]).sum(axis=0))  # type: ignore
            model = buffer.pop(medoid)
            gp = gap.pop(medoid)
            nnode = node_count.pop(medoid)
            os.rename(
                model,
                f"{folder_name}/data/model-{f.__name__}-{i}.cip",
            )
            d = {
                "gap": gp,
                "time": 45,
                "type": f.__name__,
                "index": i,
                "name": f"{folder_name}/data/model-{f.__name__}-{i}.cip",
                "nnodes": nnode,
                "open_nodes": -1,
            }
            data.append(d)

    df = pd.DataFrame(data)
    df.to_csv(f"{folder_name}/info.csv")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset using specified functions")
    parser.add_argument("folder_name", help="Folder name for saving the dataset")
    parser.add_argument(
        "number_of_instances", type=int, help="Number of instances to generate for each function"
    )
    parser.add_argument("functions", nargs="+", help="List of functions to generate instances from")

    args = parser.parse_args()

    functions = []
    for function_name in args.functions:
        try:
            function = eval(function_name)
            functions.append(function)
        except NameError:
            print(f"Invalid function name: {function_name}")

    print(make_dataset(functions, args.number_of_instances, args.folder_name))
