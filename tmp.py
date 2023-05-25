from ray.util.multiprocessing import Pool as rayPool
import ray; ray.init()
import torch
from pyscipopt import Model
import multiprocessing as mp

import pyscipopt as scip
from scipy.spatial.distance import cdist

def make_tsp():
    """
    USE MTZ formulation
    """
    #g_cpu = torch.Generator()
    #if seed is not None:
    #    g_cpu.manual_seed(seed)
    # Define a distance matrix for the cities
    size = 75
    d = torch.randn(size,2,).numpy()*2
    dist_matrix = cdist(d,d)
    #print("TSP size",size)
    # Create a SCIP model
    model = Model("TSP")

    # Define variables
    num_cities = dist_matrix.shape[0]
    x = {}

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x[i,j] = model.addVar(vtype="B", name=f"x_{i}_{j}")
    u={}
    for i in range(1,num_cities):
        u[i] = model.addVar(vtype="I", name=f"u_{i}")
        model.addCons(1<=(u[i]<= num_cities-1), name=f"u_{i}_constraint")

    # Define constraints
    # Each city must be visited exactly once
    for i in range(num_cities):
        model.addCons(scip.quicksum(x[i,j] for j in range(num_cities) if j != i) == 1, name=f"city_{i}_visited_origin")
    for j in range(num_cities):
        model.addCons(scip.quicksum(x[i,j] for i in range(num_cities) if j != i) == 1, name=f"city_{j}_visited_dest")
    # There should be no subtours
    for i in range(1,num_cities):
        for j in range(1,num_cities):
            if i != j:
                model.addCons(u[i] - u[j] + (num_cities - 1)*x[i,j]<= num_cities-2, name=f"no_subtour_{i}_{j}")
    

    # Define objective
    model.setObjective(scip.quicksum(dist_matrix[i,j] * x[i,j] for i in range(num_cities) for j in range(num_cities) if j != i), sense="minimize")

    return model


def launch_models(pool, num_proc: int):
    g = torch.Generator()
    arg_list = []
    for _ in range(num_proc):
        seed = g.seed()
        f = make_tsp
        arg_list.append((seed,  f))
    result = pool.map(__make_and_optimize, arg_list)
    g, bg = [], []
    for res in result.get():
        (gap, baseline_gap) = res.get()
        g.append(gap)
        bg.append(baseline_gap)
        print("retrieved")
    return g, bg


def __make_and_optimize(t):
    seed,  f = t
    print("started")
    torch.manual_seed(seed)
    model = f()
    model.setRealParam("limits/time", 60)
    model.hideOutput()
    model.optimize()
    baseline_gap=model.getGap()
    baseline_nodes = model.getNNodes()

    model.freeTransform()
    #model.freeProb()
    model.setRealParam("limits/time", 60)
    model.hideOutput()
    model.optimize()
    gap = model.getGap()
    print("done converting, starting to send to main process")
    return (gap, baseline_gap)




def main():
    mp.set_start_method("spawn")
    pool = rayPool(processes=16)
    g,bg = launch_models(pool, 16)
    print(g)
    print(bg)

if __name__ == "__main__":
    main()