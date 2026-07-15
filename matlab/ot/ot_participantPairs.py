import os
import pickle as pkl
import sys
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr, spearmanr
from scipy.io import savemat
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import ot
import time

##=== Load data

# Load distance matrix from .mat file
# Contains:
#   rating_mats: stim x stim x participants
source_file = 'preference_distance.mat'

with h5py.File(source_file, 'r') as f:
    rating_mats = np.array(f['rating_mats'])

# loading mat files with h5py reverses the dimension order
rating_mats = rating_mats.transpose(np.arange(rating_mats.ndim)[::-1])

# rating_mats shape: stim x stim x participants
n_stim, _, n_participants = rating_mats.shape

##=== Preprocessing

# Generate participant pairs
p_pairs = np.array(list(combinations(range(n_participants), 2)))
n_pairs = np.size(p_pairs, 0)

##=== Select epsilon values/range

#epsilons = [0.02, 0.2, 2.0, 20.0]
epsilons = np.logspace(-5, 1, 20, base=10)
n_eps = len(epsilons)

##=== GWOT

# OT.plan gives transport plan
# OT.dist gives transport distance
# OT.time gives computation time
OT = {
    'plan': np.full((n_stim, n_stim, n_pairs, n_eps), np.nan),
    'dist': np.full((n_pairs, n_eps), np.nan),
    'time': np.full((n_pairs, n_eps), np.nan)
    }

for pair in range(n_pairs):
    for e, epsilon in enumerate(epsilons):
        start_time = time.time()
        
        C1 = rating_mats[:, :, p_pairs[pair, 0]]
        C2 = rating_mats[:, :, p_pairs[pair, 1]]
        
        # Get OT plan
        plan, gw_log = ot.gromov.entropic_gromov_wasserstein(
            C1=C1, C2=C2, epsilon=epsilon, loss_fun='square_loss', verbose=False, log=True
        )
        dist = gw_log['gw_dist']
        
        # If the plan is all 0s, then GWOT didn't converge
        if np.all(plan == 0):
            plan[:] = np.nan
            dist = np.nan
        
        end_time = time.time()
            
        OT['plan'][:, :, pair, e] = plan
        OT['dist'][pair, e] = dist
        OT['time'][pair, e] = end_time - start_time
        
        print(f"OT | pair={pair}/{n_pairs-1}, epsilon={epsilon} | duration={end_time - start_time:.3f}s")
    
##=== Save results to .mat
savemat('OT_preference_distance.mat', {
    'OT': OT,
    'epsilons': epsilons,
    'p_pairs': p_pairs
})

"""

# Load two dissimilarity matrices of 93 colors, which are averaged over two non-overlapping color-neurotypical participants
RDM1 = np.load(os.path.join(folder_path, 'matrices/RDM_neutyp_group1.npy'))
RDM2 = np.load(os.path.join(folder_path, 'matrices/RDM_neutyp_group2.npy'))

# Load color labels
color_labels = np.load(os.path.join(folder_path, 'color_label/new_color_order.npy'))



epsilon = 0.2 # set the value of hyperparameter
# find the optimal transportation plan
OT_plan = ot.gromov.entropic_gromov_wasserstein(C1=RDM1,C2=RDM2,epsilon=epsilon,loss_fun="square_loss",verbose=True)


# set the ranges of epsilon
epsilons = [0.02, 0.2, 2.0, 20.0]

# visualize optimal transportation plans depending on epsilon
for epsilon in epsilons:
  OT_plan = ot.gromov.entropic_gromov_wasserstein(C1=RDM1,C2=RDM2,epsilon=epsilon,loss_fun="square_loss",verbose=False)
  show_heatmaps(matrices=[OT_plan], titles=[f'Optimal transportation plan '], color_labels=color_labels)


"""