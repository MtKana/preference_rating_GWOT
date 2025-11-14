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
from utilityFunctions import show_heatmaps
import time
import pickle

def GWD_and_find_best(matrix1, matrix2, epsilons):

    def comp_matching_rate(OT_plan, k, order="maximum"):
        """Computes the matching rate based on diagonal dominance in the optimal transport matrix."""
        diagonal = np.diag(OT_plan)
        if order == "maximum":
            topk_values = np.partition(OT_plan, -k)[:, -k:]
        elif order == "minimum":
            topk_values = np.partition(OT_plan, k - 1)[:, :k]
        count = np.sum([diagonal[i] in topk_values[i] for i in range(OT_plan.shape[0])])
        return count / OT_plan.shape[0] * 100  # Matching rate in %

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
        matching_rate = comp_matching_rate(OT, k=1)  # Calculate the top-1 matching rate

        OT_plans.append(OT)
        gwds.append(gwd)
        matching_rates.append(matching_rate)
        valid_epsilons.append(epsilon)

    if not gwds:
        raise ValueError("No valid epsilon values resulted in a non-zero transportation matrix.")

    # Identify the best epsilon (corresponding to the minimum GWD)
    min_gwd = min(gwds)
    best_eps_idx = gwds.index(min_gwd)
    best_eps = valid_epsilons[best_eps_idx]
    OT_plan = OT_plans[best_eps_idx]
    matching_rate = matching_rates[best_eps_idx]

    # Print the best epsilon, minimum GWD, and matching rate
    print(f"Best epsilon: {best_eps}, Minimum GWD: {min_gwd:.3f}, Matching Rate: {matching_rate:.1f}%")

    return best_eps, OT_plan, min_gwd, matching_rate


def compute_GWOT_for_all_pairs(matrix_pairs, pair_indices, epsilons, save_filename):
    results = []

    for idx, ((matrix1, matrix2), original_index) in enumerate(zip(matrix_pairs, pair_indices)):
        print(f"\nProcessing matrix pair {idx + 1}/{len(matrix_pairs)} (original index: {original_index})...")
        start_time = time.time()  # Start timer

        try:
            best_eps, OT_plan, min_gwd, matching_rate = GWD_and_find_best(matrix1, matrix2, epsilons)
            # Store original index with the result
            results.append((original_index, best_eps, OT_plan, min_gwd, matching_rate))
        except ValueError as e:
            print(f"Skipping pair {idx + 1} due to error: {e}")

        elapsed_time = time.time() - start_time  # Compute elapsed time
        print(f"Time taken for pair {idx + 1}: {elapsed_time:.2f} seconds")

    # Save results to file
    with open(save_filename, "wb") as file:
        pickle.dump(results, file)

    print(f"Results successfully saved to {save_filename}")
    return results

def GWD_and_plot(matrix1, matrix2, epsilons):

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

    # Identify the best epsilon (corresponding to the minimum GWD)
    min_gwd = min(gwds)
    best_eps_idx = gwds.index(min_gwd)
    best_eps = valid_epsilons[best_eps_idx]
    OT_plan = OT_plans[best_eps_idx]
    matching_rate = matching_rates[best_eps_idx]

    # Print the best epsilon and its corresponding minimum GWD
    print(f"Best epsilon: {best_eps}, Minimum GWD: {min_gwd:.3f}")

    # Plot GWD vs Epsilon
    plt.scatter(valid_epsilons, gwds, c=matching_rates, cmap="viridis")
    plt.xlabel("epsilon")
    plt.ylabel("GWD")
    plt.xscale('log')
    plt.grid(True, which='both')
    cbar = plt.colorbar()
    cbar.set_label(label='Matching Rate (%)')
    plt.scatter([best_eps], [min_gwd], color='red', marker='o', label=f'Best ε={best_eps}')  # Highlight best epsilon
    plt.legend()
    plt.show()

    # Display the optimal transport plan
    show_heatmaps(
        0, 0.1,
        matrices=[OT_plan],
        titles=[f'Optimal transportation plan \n GWD={min_gwd:.3f} \n Best ε={best_eps} \n Matching rate: {matching_rate:.1f}%'],
        nrows=1, ncols=1, cbar_label=None, color_labels=None, cmap_name='viridis'
    )

    return OT_plan, gwds, matching_rates, min_gwd, matching_rate

def GWD_and_plot_rotational_OT(matrix1, matrix2, epsilons,
                 rotinv_method: str = "fro_norm",
                 rotinv_threshold: float | None = None,
                 plot_candidates: bool = True):

    def comp_matching_rate(OT_plan, k, order="maximum"):
        diagonal = np.diag(OT_plan)
        if order == "maximum":
            topk_values = np.partition(OT_plan, -k)[:, -k:]
        elif order == "minimum":
            topk_values = np.partition(OT_plan, k - 1)[:, :k]
        count = np.sum([diagonal[i] in topk_values[i] for i in range(OT_plan.shape[0])])
        return count / OT_plan.shape[0] * 100.0
    
    def make_rotational_candidates(n: int):
        """
        Build 24 candidates for an n x n OT plan (n=12 expected):
        - 12 cyclic row shifts of the identity (down by s = 0..n-1)
        - 12 rotation-90° clockwise of each shifted matrix
        Returns:
        candidates: list[np.ndarray] of shape (n, n)
        labels:     list[str] same length as candidates
        """
        eye = np.eye(n)
        candidates, labels = [], []

        for s in range(n):
            # cyclic row shift (down) by s
            P = np.roll(eye, shift=s, axis=0)
            candidates.append(P)
            labels.append(f"shift_{s}")

            # 90° clockwise rotation of the shift
            # np.rot90(..., k=-1) rotates 90° clockwise
            P_rot = np.rot90(P, k=-1)
            candidates.append(P_rot)
            labels.append(f"shift_{s}_rot90")

        return candidates, labels

    def candidate_matching_scores(OT: np.ndarray,
                                candidates: list[np.ndarray],
                                method: str = "fro_norm",
                                threshold: float | None = None):
        """
        Compute a score for each candidate vs the OT plan.
        Methods:
        - "fro_norm": <OT, P> / (||OT||_F * ||P||_F)  in [0,1]
        - "mass_fraction": sum(OT * P) / sum(OT)      in [0,1]
        - "threshold_hits": fraction of positions where both OT>=threshold and P==1
                            (requires 'threshold' argument)
        Returns: scores (list[float])
        """
        OT = np.asarray(OT, dtype=float)
        scores = []

        if method == "fro_norm":
            ot_norm = np.linalg.norm(OT, ord="fro")
            for P in candidates:
                denom = ot_norm * np.linalg.norm(P, ord="fro")
                # safe guard if denom == 0
                score = 0.0 if denom == 0 else float(np.sum(OT * P) / denom)
                scores.append(score)

        elif method == "mass_fraction":
            ot_sum = float(OT.sum())
            for P in candidates:
                score = 0.0 if ot_sum == 0 else float(np.sum(OT * P) / ot_sum)
                scores.append(score)

        elif method == "threshold_hits":
            if threshold is None:
                raise ValueError("threshold_hits method requires a 'threshold' value.")
            mask = (OT >= threshold)
            denom = mask.sum()
            for P in candidates:
                # count how many 'active' OT entries land on candidate ones
                hits = np.logical_and(mask, (P == 1)).sum()
                score = 0.0 if denom == 0 else float(hits / denom)
                scores.append(score)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from 'fro_norm', 'mass_fraction', 'threshold_hits'.")

        return scores

    def plot_candidate_scores(scores, labels, title="Rotational Candidate Matching Rates"):
        """
        Simple line plot of 24 candidate scores.
        """
        x = np.arange(len(scores))
        plt.figure(figsize=(14, 4))
        plt.plot(x, scores, marker='o')
        plt.xticks(x, labels, rotation=60, ha='right', fontsize=8)
        plt.ylabel("Matching Rate (score)")
        plt.xlabel("Candidate")
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
    print("Computing Gromov-Wasserstein distances and optimal transport plans...")

    OT_plans, gwds, matching_rates, valid_epsilons = [], [], [], []

    for epsilon in epsilons:
        OT, gw_log = ot.gromov.entropic_gromov_wasserstein(
            C1=matrix1, C2=matrix2, epsilon=epsilon, loss_fun="square_loss", log=True
        )
        if not OT.any():
            print(f"Skipping epsilon={epsilon} because it results in a zero transportation matrix.")
            continue
        gwd = gw_log['gw_dist']
        mr = comp_matching_rate(OT, k=1)

        OT_plans.append(OT)
        gwds.append(gwd)
        matching_rates.append(mr)
        valid_epsilons.append(epsilon)

        print(f"Epsilon: {epsilon}, GWD: {gwd}, Matching Rate: {mr}")

    if not gwds:
        raise ValueError("No valid epsilon values resulted in a non-zero transportation matrix.")

    # Identify best epsilon
    min_gwd = min(gwds)
    best_idx = int(np.argmin(gwds))
    best_eps = valid_epsilons[best_idx]
    OT_plan = OT_plans[best_idx]
    matching_rate = matching_rates[best_idx]

    print(f"Best epsilon: {best_eps}, Minimum GWD: {min_gwd:.3f}")

    # Plot GWD vs epsilon with color by (standard) matching rate
    plt.figure()
    sc = plt.scatter(valid_epsilons, gwds, c=matching_rates, cmap="viridis")
    plt.xlabel("epsilon")
    plt.ylabel("GWD")
    plt.xscale("log")
    plt.grid(True, which="both")
    cbar = plt.colorbar(sc)
    cbar.set_label(label="Matching Rate (%)")
    plt.scatter([best_eps], [min_gwd], color="red", marker="o", label=f"Best ε={best_eps}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # OT plan heatmap
    show_heatmaps(
        0, 0.1,
        matrices=[OT_plan],
        titles=[f'Optimal transportation plan \n GWD={min_gwd:.3f} \n Best ε={best_eps} \n Matching rate: {matching_rate:.1f}%'],
        nrows=1, ncols=1, cbar_label=None, color_labels=None, cmap_name='viridis'
    )

    # ---- Rotation-invariant matching rate & plot ----
    n = OT_plan.shape[0]
    candidates, labels = make_rotational_candidates(n)
    cand_scores = candidate_matching_scores(OT_plan, candidates,
                                            method=rotinv_method,
                                            threshold=rotinv_threshold)
    rotinv_max_score = float(np.max(cand_scores))
    rotinv_best_label = labels[int(np.argmax(cand_scores))]
    print(f"RotInv best candidate: {rotinv_best_label}, score={rotinv_max_score:.3f}")

    if plot_candidates:
        plot_candidate_scores(
            cand_scores,
            labels,
            title="Rotational Candidate Matching Rates (vs. OT plan at best ε)"
        )

    # Return everything your code already expects, plus the rotation-invariant max
    return OT_plan, gwds, matching_rates, min_gwd, matching_rate, candidates, labels, cand_scores, rotinv_best_label, rotinv_max_score

# def compute_GWOT_for_all_pairs(matrix_pairs, epsilons):
#     results = []

#     for idx, (matrix1, matrix2) in enumerate(matrix_pairs):
#         print(f"\nProcessing matrix pair {idx + 1}/{len(matrix_pairs)}...")
#         try:
#             OT_plan, gwds, matching_rates = GWD_and_plot(matrix1, matrix2, epsilons)
#             results.append((OT_plan, gwds, matching_rates))
#         except ValueError as e:
#             print(f"Skipping pair {idx + 1} due to error: {e}")

#     return results

# # Function to compute minimum GWD for the range of epsilons
# def compute_min_gwd(matrix_1, matrix_2, epsilons):

#     OT_plans = []
#     gwds = []
#     matching_rates = []

#     for epsilon in epsilons:
#       OT, gw_log = ot.gromov.entropic_gromov_wasserstein(C1=matrix_1, C2=matrix_2 , epsilon=epsilon, loss_fun="square_loss", log=True)  # optimization
#       gwd = gw_log['gw_dist']
#       matching_rate = comp_matching_rate(OT, k=1)  # calculate the top 1 matching rate
#       OT_plans.append(OT)
#       gwds.append(gwd)
#       matching_rates.append(matching_rate)

#     return min(gwds)

# def OT_epsilon(epsilons, OT_plans, gwds, e_ind, matching_rates):

#     best_eps_idx = e_ind
#     min_gwd = gwds[best_eps_idx]
#     best_eps = epsilons[best_eps_idx]
#     OT_plan = OT_plans[best_eps_idx]
#     matching_rate = matching_rates[best_eps_idx]

#     show_heatmaps(0, 0.05,
#         matrices=[OT_plan],
#         titles=[f'Optimal transportation plan \n GWD={min_gwd:.3f} \n matching rate : {matching_rate:.1f}%'],
#         color_labels=unique_colours)
    
# # Function to plot the embeddings
# def plot_embeddings(embeddings, titles, color_labels, overlay=False):
#     fig = go.Figure()
    
#     if overlay:
#         for i, embedding in enumerate(embeddings):
#             fig.add_trace(go.Scatter3d(
#                 x=embedding[:, 0],
#                 y=embedding[:, 1],
#                 z=embedding[:, 2],
#                 mode='markers+text',
#                 marker=dict(size=10, color=color_labels),
#                 text=color_labels,
#                 textposition="top center",
#                 name=titles[i]
#             ))
#     else:
#         for i, embedding in enumerate(embeddings):
#             fig = go.Figure()
#             fig.add_trace(go.Scatter3d(
#                 x=embedding[:, 0],
#                 y=embedding[:, 1],
#                 z=embedding[:, 2],
#                 mode='markers+text',
#                 marker=dict(size=10, color=color_labels),
#                 text=color_labels,
#                 textposition="top center"
#             ))
#             fig.update_layout(
#                 title=f'MDS Embedding - {titles[i]}',
#                 scene=dict(
#                     xaxis_title='Dimension 1',
#                     yaxis_title='Dimension 2',
#                     zaxis_title='Dimension 3'
#                 ),
#                 height=800
#             )
#     fig.show()
    

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