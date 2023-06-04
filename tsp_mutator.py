import numpy as np
from numpy.random import default_rng
"""
simplified version of tspgen:
Bossek, J., Kerschke, P., Neumann, A., Wagner, M., Neumann, F., & Trautmann, H. (2019). Evolving Diverse TSP Instances by Means of Novel and Creative Mutation Operators. 
In Proceedings of the 15th ACM/SIGEVO Workshop on Foundations of Genetic Algorithms (FOGA XV), Potsdam, Germany. (Accepted)
"""


def mutator_axisprojection(coords, pm=0.1, p_jitter=0, jitter_sd=0.1):
    assert coords.ndim == 2, "coords must be a 2D matrix"
    assert 0 <= pm <= 1, "pm must be between 0 and 1"
    assert 0 <= p_jitter <= 1, "p_jitter must be between 0 and 1"
    assert jitter_sd >= 0.001, "jitter_sd must be greater than or equal to 0.001"

    to_mutate = np.random.choice(range(coords.shape[0]), size=int(pm * coords.shape[0]), replace=False)

    if len(to_mutate) < 2:
        return coords

    # sample axis
    axis = np.random.choice(range(coords.shape[1]))

    # get bounds of selected points
    rng = np.min(coords[to_mutate, axis]), np.max(coords[to_mutate, axis])

    # sample "constant axis" within range
    line = np.random.uniform(low=rng[0], high=rng[1])

    # jitter points around projected axis
    if np.random.rand() < p_jitter:
        rng_len = len(to_mutate)
        jitter = np.random.normal(loc=0, scale=jitter_sd, size=(rng_len, coords.shape[1]))
        line = line + jitter[:, axis]

    coords[to_mutate, axis] = line

    return coords

def mutator_cluster(coords, pm=0.1):
    assert coords.ndim == 2, "coords must be a 2D matrix"
    assert 0 <= pm <= 1, "pm must be between 0 and 1"

    to_mutate = np.random.choice(range(coords.shape[0]), size=int(pm * coords.shape[0]), replace=False)
    n_mutants = len(to_mutate)

    if n_mutants <= 1:
        return coords

    # generate cluster center
    cl_center = np.random.uniform(size=coords.shape[1])

    # sample sdev for mutation
    sdev = np.random.uniform(low=0.001, high=0.3)

    # generate new coordinates with random noise
    noise = np.random.normal(loc=0, scale=sdev, size=(n_mutants, coords.shape[1]))
    new_coords = noise + cl_center
    new_coords = np.clip(new_coords, 0, 1)

    coords[to_mutate, :] = new_coords

    return coords

# This is pretty dumb, but I'll keep it to maintain some parallelity with the original implementation
def get_random_linear_function():
    intercept = np.random.uniform()
    slope = np.random.uniform(low=0, high=3) if intercept < 0.5 else np.random.uniform(low=-3, high=0)

    def lin_fun(x):
        return intercept + slope * x

    return {'intercept': intercept, 'slope': slope, 'linFun': lin_fun}

def mutator_expansion(coords, min_eps=0.1, max_eps=0.3):
    assert coords.ndim == 2, "coords must be a 2D matrix"
    assert 0.05 <= min_eps <= 0.5, "min_eps must be between 0.05 and 0.5"
    assert 0.05 <= max_eps <= 0.5, "max_eps must be between 0.05 and 0.5"
    assert min_eps <= max_eps, "min_eps must not be greater than max_eps"

    eps = np.random.uniform(low=min_eps, high=max_eps)

    linear = get_random_linear_function()
    intercept = linear["intercept"]
    slope = linear["slope"]

    # Calculate the orthogonal projections of points on the linear function
    bb = np.array([0, intercept])
    cc = np.array([1, intercept + slope])
    uu = cc - bb

    projs = np.zeros_like(coords)
    projs = ((np.sum((coords[:, None] - bb) * uu, axis=2) / np.sum(uu**2))[:, None] * uu) + bb
    #for i in range(coords.shape[0]):
    #     point = coords[i]
    #     projs[i] = ((np.sum((point - bb) * uu) / np.sum(uu**2)) * uu) + bb

    dists = np.sqrt(np.sum((projs - coords)**2, axis=1))
    to_mutate = np.where(dists < eps)[0]

    if len(to_mutate) < 2:
        return coords

    norm_dir_vecs = (coords[to_mutate] - projs[to_mutate]) / np.sqrt(np.sum((coords[to_mutate] - projs[to_mutate])**2, axis=1, keepdims=True))

    if type == "Expansion":
        mutants = projs[to_mutate] + norm_dir_vecs * (eps + np.random.exponential(scale=10, size=(len(to_mutate), 1)))
    else:
        mutants = coords[to_mutate] - norm_dir_vecs * dists[to_mutate].reshape(-1, 1) * np.clip(np.abs(np.random.normal(size=len(to_mutate))), None, 1)

    coords[to_mutate] = mutants

    return coords

def mutator_implosion(coords, min_eps=0.1, max_eps=0.3):
    assert coords.shape[1] > 1, "Invalid number of coordinates in `coords`."
    assert min_eps > 0 and max_eps > 0, "Implosion radius must be positive."
    assert min_eps <= max_eps, "min_eps must not be greater than max_eps."

    num_coords, num_dims = coords.shape

    # Get implosion center
    blackhole = np.random.uniform(size=num_dims)

    # Sample implosion radius
    eps = np.random.uniform(min_eps, max_eps)

    dists = np.linalg.norm(coords - blackhole, axis=1)
    to_mutate = np.where(dists < eps)[0]

    if len(to_mutate) < 2:
        return coords

    mutants = coords[to_mutate] + (coords[to_mutate] - blackhole) * np.minimum(np.abs(np.random.normal()), eps)
    coords[to_mutate] = mutants

    return coords

def mutator_grid(coords, box_min=0.1, box_max=0.3, p_rot=0, p_jitter=0, jitter_sd=0):
    assert coords.shape[1] > 1, "Invalid number of coordinates in `coords`."
    assert box_min >= 0.05 and box_max >= 0.05, "Box width and height must be at least 0.05."
    assert box_min <= box_max, "box_min must not be greater than box_max."
    assert 0 <= p_rot <= 1, "p_rot must be between 0 and 1."
    assert 0 <= p_jitter <= 1, "p_jitter must be between 0 and 1."
    assert jitter_sd >= 0, "jitter_sd must be non-negative."

    num_coords, num_dims = coords.shape

    box_width = np.random.uniform(box_min, box_max)
    box_height = np.random.uniform(box_min, box_max)

    anchor = np.random.uniform([0, 0], [1 - box_width, 1 - box_height])

    to_mutate = np.where(
        (coords[:, 0] > anchor[0])
        & (coords[:, 0] <= anchor[0] + box_width)
        & (coords[:, 1] > anchor[1])
        & (coords[:, 1] <= anchor[1] + box_height)
    )[0]

    n_mutants = len(to_mutate)
    if n_mutants < 2:
        return coords

    k_dim = int(np.floor(np.sqrt(n_mutants)))

    k_dim_sq = k_dim ** 2
    if k_dim_sq < n_mutants:
        to_mutate = np.random.choice(to_mutate, size=k_dim_sq, replace=False)

    grid_x = np.linspace(anchor[0], anchor[0] + box_width, k_dim)
    grid_y = np.linspace(anchor[1], anchor[1] + box_height, k_dim)
    grid = np.transpose([np.tile(grid_x, k_dim), np.repeat(grid_y, k_dim)])

    if np.random.uniform() < p_rot:
        angle = np.random.uniform(0, 90)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        grid_mean = np.mean(grid, axis=0)
        grid = np.dot(grid - grid_mean, rotation_matrix.T) + grid_mean

    if np.random.uniform() < p_jitter and jitter_sd > 0:
        jitter = np.random.normal(scale=jitter_sd, size=(k_dim_sq, num_dims))
        grid += jitter

    coords[to_mutate] = grid

    return coords

def mutator_linear_projection(coords, pm=0.1, p_jitter=0, jitter_sd=0):
    assert coords.shape[1] > 1, "Invalid number of coordinates in `coords`."
    assert 0 <= pm <= 1, "pm must be between 0 and 1."
    assert 0 <= p_jitter <= 1, "p_jitter must be between 0 and 1."
    assert jitter_sd >= 0, "jitter_sd must be non-negative."

    num_coords, num_dims = coords.shape

    to_mutate = np.random.choice(np.arange(num_coords), size=int(pm * num_coords), replace=False)

    n_mutants = len(to_mutate)
    if n_mutants < 2:
        return coords

    beta_0 = np.random.uniform()

    if beta_0 < 0.5:
        beta_1 = np.random.uniform(0, 3)
    else:
        beta_1 = np.random.uniform(-3, 0)

    coords[to_mutate, 1] = beta_0 + beta_1 * coords[to_mutate, 0]

    if np.random.uniform() < p_jitter and jitter_sd > 0:
        coords[to_mutate, 1] += np.random.normal(scale=jitter_sd, size=n_mutants)

    return coords

def mutator_add_col(coords, std=(1,50),mu=(-10,10)):
    mu = mu[0]+np.random.rand(coords.shape[0])*(mu[1]-mu[0])
    std = std[0]+np.random.rand(coords.shape[0])*(std[1]-std[0])
    col = np.random.randn(coords.shape[0])*std + mu
    coords = np.concatenate([coords,col.reshape(-1,1)],1)
    print(coords.shape)
    return coords

muts = [mutator_axisprojection,mutator_cluster,mutator_expansion,mutator_linear_projection,mutator_implosion]

def do_mutation(coords, funs=muts, n_muts=15):
    idx = np.random.choice(len(funs),size=n_muts,replace=True)
    for i in idx:
        coords = funs[i](coords)
    for f in funs:
        coords = f(coords)
    return coords