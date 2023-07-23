import pandas as pd
import os
import argparse
from ProblemCreators import *

def make_dataset(functions, number_of_instances, folder_name):
    data = []
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        os.mkdir(f"{folder_name}/data")
    for f in functions:
        buffer = []
        gap = []
        for i in range(50):
            model = f()
            model.setRealParam("limits/time", 45)
            #model.setRealParam("limits/gap", 0.01)
            model.optimize()
            if 0.0 < model.getGap()<50 and model.getNNodes()>50:
                buffer.append(model)
                gap.append(model.getGap())
        for i in range(number_of_instances):
            accept = False
            for _ in range(15):
                for _ in range(3):
                    model = f()
                    model.setRealParam("limits/time", 45)
                    #model.setRealParam("limits/gap", 0.01)
                    model.optimize()
                    if 0.0 < model.getGap()<50 and model.getNNodes()>50:
                        print("adding in model, the overall buffer size is now", len(buffer))
                        buffer.append(model)
                        gap.append(model.getGap())
                # now get the medoid
                gap_a = np.array(gap)
                medoid = np.argmin(np.abs(gap_a - gap_a[:,None]).sum(axis=0)) # type: ignore
                model = buffer.pop(medoid)
                gap.pop(medoid)
                model.writeProblem(f"{folder_name}/data/model-{f.__name__}-{i}.cip")
                #if 10.00 < model.getGap():
                #    print("gap too large, rerunning")
                #    continue
                if 0.0 < model.getGap() < 50.0 and model.getNNodes()>50:
                    d = {"gap":model.getGap(), "time" : 45, "type":f.__name__, "index":i, "name":f"{folder_name}/data/model-{f.__name__}-{i}.cip", "open_nodes":sum([len(x) for x in model.getOpenNodes()])}
                    data.append(d)
                    break
    df = pd.DataFrame(data)
    df.to_csv(f"{folder_name}/info.csv")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset using specified functions")
    parser.add_argument("folder_name", help="Folder name for saving the dataset")
    parser.add_argument("number_of_instances", type=int, help="Number of instances to generate for each function")
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
