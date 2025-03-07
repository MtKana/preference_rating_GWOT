import os
import pickle as pkl
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import csv
from scipy.stats import pearsonr, spearmanr
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
import seaborn as sns
import ot
import plotly.graph_objs as go
import plotly.express as px
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math


# Define the function to compute minimum GWD for the range of epsilons
def compute_min_gwd(matrix_1, matrix_2, epsilons):

    OT_plans = []
    gwds = []
    matching_rates = []

    for epsilon in epsilons:
      OT, gw_log = ot.gromov.entropic_gromov_wasserstein(C1=matrix_1, C2=matrix_2 , epsilon=epsilon, loss_fun="square_loss", log=True)  # optimization
      gwd = gw_log['gw_dist']
      matching_rate = comp_matching_rate(OT, k=1)  # calculate the top 1 matching rate
      OT_plans.append(OT)
      gwds.append(gwd)
      matching_rates.append(matching_rate)

    return min(gwds)

def comp_matching_rate(OT_plan, k, order="maximum"):
  # This function computes the matching rate, assuming that in the optimal transportation plan,
  # the items in the i-th row and the j-th column are the same (correct mactch) when i = j.
  # Thus, the diagonal elements of the optimal transportation plan represent the probabilities
  # that the same items (colors) match between the two structures.

  # Get the diagonal elements
  diagonal = np.diag(OT_plan)
  # Get the top k values for each row
  if order == "maximum":
      topk_values = np.partition(OT_plan, -k)[:, -k:]
  elif order == "minimum":
      topk_values = np.partition(OT_plan, k - 1)[:, :k]
  # Count the number of rows where the diagonal is in the top k values and compute the matching rate
  count = np.sum([diagonal[i] in topk_values[i] for i in range(OT_plan.shape[0])])
  matching_rate = count / OT_plan.shape[0] * 100
  return matching_rate

def GWD_and_plot(matrix1, matrix2, epsilons):
    OT_plans = []
    gwds = []
    matching_rates = []
    valid_epsilons = []

    for epsilon in epsilons:
        OT, gw_log = ot.gromov.entropic_gromov_wasserstein(
            C1=matrix1, C2=matrix2, epsilon=epsilon, loss_fun="square_loss", log=True
        )  # optimization
        
        # Check if the transportation matrix is all zeros
        if not OT.any():
            print(f"Skipping epsilon={epsilon} because it results in a zero transportation matrix.")
            continue

        gwd = gw_log['gw_dist']
        matching_rate = comp_matching_rate(OT, k=1)  # calculate the top-1 matching rate

        OT_plans.append(OT)
        gwds.append(gwd)
        matching_rates.append(matching_rate)
        valid_epsilons.append(epsilon)

    if not gwds:
        raise ValueError("No valid epsilon values resulted in a non-zero transportation matrix.")

    plt.scatter(valid_epsilons, gwds, c=matching_rates)
    plt.xlabel("epsilon")
    plt.ylabel("GWD")
    plt.xscale('log')
    plt.grid(True, which='both')
    cbar = plt.colorbar()
    cbar.set_label(label='Matching Rate (%)')
    plt.show()

    # Extract the best epsilon that minimizes the GWD
    min_gwd = min(gwds)
    best_eps_idx = gwds.index(min_gwd)
    best_eps = valid_epsilons[best_eps_idx]
    OT_plan = OT_plans[best_eps_idx]
    matching_rate = matching_rates[best_eps_idx]

    if min_gwd == 0:
        print(f'Optimal transportation plan \n GWD={min_gwd:.3f} \n matching rate : {matching_rate:.1f}%')

    show_heatmaps(0, 0.1, matrices=[OT_plan], titles=[
        f'Optimal transportation plan \n GWD={min_gwd:.3f} \n matching rate : {matching_rate:.1f}%'
    ])

    return OT_plan, gwds, matching_rates


def OT_epsilon(epsilons, OT_plans, gwds, e_ind, matching_rates):

    best_eps_idx = e_ind
    min_gwd = gwds[best_eps_idx]
    best_eps = epsilons[best_eps_idx]
    OT_plan = OT_plans[best_eps_idx]
    matching_rate = matching_rates[best_eps_idx]

    show_heatmaps(0, 0.05,
        matrices=[OT_plan],
        titles=[f'Optimal transportation plan \n GWD={min_gwd:.3f} \n matching rate : {matching_rate:.1f}%'],
        color_labels=unique_colours)
    

# # Function to compute GWD and plot the OT plan as well as GWD values
# def pairwise_csv_GWOT(file1, file2, eps_range, n_eps, response_type):
#     # Get color index dictionary
#     colour_index = getUniqueColours()
#     matrix_size = len(colour_index)

#     # Generate epsilon values
#     epsilons = np.logspace(np.log10(eps_range[0]), np.log10(eps_range[1]), n_eps)

#     # Initialize matrices
#     matrix_1 = np.zeros((matrix_size, matrix_size))
#     matrix_2 = np.zeros((matrix_size, matrix_size))

#     # Load and process the first CSV file
#     df_PM1 = pd.read_csv(file1)
#     df_PM1 = df_PM1[(df_PM1['practice_trial'] != 1) & (df_PM1['response_type'] == response_type)]
#     colour1_1 = df_PM1['colour1']
#     colour2_1 = df_PM1['colour2']
#     target_preference_1 = df_PM1['response']

#     for c1, c2, tp in zip(colour1_1, colour2_1, target_preference_1):
#         I = colour_index[c1]
#         j = colour_index[c2]
#         matrix_1[I, j] = tp

#     matrix_1 = matrix_1.astype(int)

#     # Load and process the second CSV file
#     df_PM2 = pd.read_csv(file2)
#     df_PM2 = df_PM2[(df_PM2['practice_trial'] != 1) & (df_PM2['response_type'] == response_type)]
#     colour1_2 = df_PM2['colour1']
#     colour2_2 = df_PM2['colour2']
#     target_preference_2 = df_PM2['response']

#     for c1, c2, tp in zip(colour1_2, colour2_2, target_preference_2):
#         I = colour_index[c1]
#         j = colour_index[c2]
#         matrix_2[I, j] = tp

#     matrix_2 = matrix_2.astype(int)

#     # Calculate RSA correlation
#     RSA_corr = RSA(matrix_1, matrix_2)
#     print('RSA correlation coefficient : ', RSA_corr)

#     # Calculate GWD and plot
#     OT_plan_as, gwds_as, matching_rates_as = GWD_and_plot(matrix_1, matrix_2, epsilons)
#     print('Minimum GWD: ', min(gwds_as))