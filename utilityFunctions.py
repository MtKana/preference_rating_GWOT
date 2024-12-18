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

def load_csv(file_path):
    return pd.read_csv(file_path, usecols=["response", "response_type", "practice_trial", "trials.thisIndex"])


def add_colored_label(ax, x, y, bgcolor, width=1, height=1):
  rect = Rectangle((x, y), width, height, facecolor=bgcolor)
  ax.add_patch(rect)


def sort_files_in_directory(directory_path):
    """
    Sorts and returns a list of file names in a specified directory.

    Args:
    - directory_path (str): Path to the directory containing the files.

    Returns:
    - List[str]: A sorted list of file names in the directory.
    """
    try:
        # List all files in the directory
        files = os.listdir(directory_path)
        
        # Sort files alphabetically (default behavior of sort())
        files.sort()  
        
        return files
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' does not exist.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


# Display multiple matrices as a subplot
#
# INPUTS:
#   vmin_val: number, minimum of colour scale
#   vmin_val: number, maximum of colour scale
#   matrices: list, list of 2D numpy matrices
#   titles: list, list of title strings for each subplot
#   cbar_label: string, title for colour bars
#   color_labels: dictionary, dictionary of colours and their ids (x/y axis position), {colour:id}
# OUTPUTS:
#   Returns nothing, just plots
def show_heatmaps(vmin_val, vmax_val, matrices, rows, cols, titles, cbar_label=None, color_labels=None):
    num_plots = len(matrices)
    grid_size = math.ceil(math.sqrt(num_plots))  # Determine the grid size
    fig, axs = plt.subplots(rows, cols, figsize=(16, 9)) #figsize=(5 * grid_size, 5 * grid_size))

#     # Flatten the axes array if it is 2D
#     if isinstance(axs, np.ndarray):
#         axs = axs.ravel()
#     else:
#         axs = [axs]

#     for i, (matrix, title) in enumerate(zip(matrices, titles)):
#         ax = axs[i]
        
        im = ax.imshow(matrix, aspect='auto', vmin=vmin_val, vmax=vmax_val)
        ax.set_title(title, fontsize=14)

#         # Set axis labels
#         ax.set_xlabel("Right")  # Label for x-axis
#         ax.set_ylabel("Left")   # Label for y-axis

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, fontsize=14)
        cbar.ax.tick_params(labelsize=14)

#         # Adjust the height of the color bar
#         position = cax.get_position()
#         new_position = [position.x0, position.y0 + 0.1, position.width, position.height * 0.8]
#         cax.set_position(new_position)

#         if color_labels is not None:
#             ax.axis('off')
#             for idx, color in enumerate(color_labels):
#                 add_colored_label(ax, -1.5, idx - 0.5, color, width=0.8)
#                 add_colored_label(ax, idx - 0.5, matrix.shape[1] - 0.2, color, height=0.8)

#             ax.set_aspect('equal')
#             ax.set_xlim(-3.0, matrix.shape[1])
#             ax.set_ylim(-1, len(color_labels) + 1.4)
#             ax.invert_yaxis()

#             for spine in ax.spines.values():
#                 spine.set_visible(False)

#     # Hide unused axes
#     for ax in axs[num_plots:]:
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()

def show_heatmaps(vmin_val, vmax_val, matrices, titles, cbar_label=None, color_labels=None):
    num_plots = len(matrices)
    fig, axs = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))

    if num_plots == 1:
        axs = [axs]

    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = axs[i]
        
        im = ax.imshow(matrix, aspect='auto', vmin=vmin_val, vmax=vmax_val)
        ax.set_title(title)

        # Set axis labels
        ax.set_xlabel("Right")  # Label for x-axis
        ax.set_ylabel("Left")   # Label for y-axis

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label)

        # Adjust the height of the color bar
        position = cax.get_position()
        new_position = [position.x0, position.y0 + 0.1, position.width, position.height * 0.8]
        cax.set_position(new_position)

        if color_labels is not None:
            ax.axis('off')
            for idx, color in enumerate(color_labels):
                add_colored_label(ax, -1.5, idx-0.5, color, width=0.8)
                add_colored_label(ax, idx-0.5, matrix.shape[1] - 0.2, color, height=0.8)

            ax.set_aspect('equal')
            ax.set_xlim(-3.0, matrix.shape[1])
            ax.set_ylim(-1, len(color_labels)+1.4)
            ax.invert_yaxis()

            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.show()

def show_heatmap(matrix, title, cbar_label=None, color_labels=None):
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(matrix, aspect='auto', vmin=0, vmax=7)
    ax.set_title(title)

    # Set axis labels
    ax.set_xlabel("Right")  # Label for x-axis
    ax.set_ylabel("Left")   # Label for y-axis

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)

    # Adjust the height of the color bar
    position = cax.get_position()
    new_position = [position.x0, position.y0 + 0.1, position.width, position.height * 0.8]
    cax.set_position(new_position)

    if color_labels is not None:
        ax.axis('off')
        for idx, color in enumerate(color_labels):
            add_colored_label(ax, -1.5, idx - 0.5, color, width=0.8)
            add_colored_label(ax, idx - 0.5, matrix.shape[1] - 0.2, color, height=0.8)

        ax.set_aspect('equal')
        ax.set_xlim(-3.0, matrix.shape[1])
        ax.set_ylim(-1, len(color_labels) + 1.4)
        ax.invert_yaxis()

        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.show()


def transform_values(matrix):
    # Define the transformation mapping
    transform_map = {
        0: 4,
        1: 3,
        2: 2,
        3: 1,
        4: -1,
        5: -2,
        6: -3,
        7: -4
    }

    # Vectorized transformation approach
    # We can use a vectorized operation by creating a look-up array.
   
    # Vectorized transformation using a lookup array
    lookup = np.array([transform_map[i] for i in range(8)], dtype=int)

    # Ensure the matrix contains valid indices (0-7)
    if not np.all((matrix >= 0) & (matrix <= 7)):
        raise ValueError("Matrix contains values outside the range 0-7.")

    # Apply the transformation
    transformed_matrix = lookup[matrix]
    return transformed_matrix

def compute_color_preference_distance_matrix(matrix):
    """
    Computes the color preference distance matrix from the given matrix.
    Diagonal values are directly copied from the original matrix.

    :param matrix: A 12x12 numpy array with integer values ranging from -4 to 4
    :return: A 12x12 numpy array representing the color preference distance matrix
    """
    # Ensure the input is a numpy array
    matrix = np.array(matrix)

    # Initialize the distance matrix with zeros
    distance_matrix = np.zeros_like(matrix, dtype=float)

    # Iterate over each pair of cells (i, j) and (j, i)
    for i in range(matrix.shape[0]):
        for j in range(i, matrix.shape[1]):
            if i == j:  # Diagonal values
                distance_matrix[i, j] = matrix[i, j]
            else:  # Off-diagonal values
                value1 = matrix[i, j]
                value2 = matrix[j, i]

                # Special case: if values are -1 and 1, set distance to 0
                if (value1 == -1 and value2 == 1) or (value1 == 1 and value2 == -1):
                    distance = 0
                else:
                    # Flip one value if both are positive or both are negative
                    if (value1 > 0 and value2 > 0) or (value1 < 0 and value2 < 0):
                        value2 = -value2

                    # Compute the average of the absolute values
                    distance = (abs(value1) + abs(value2)) / 2

                # Assign the distance to the distance matrix
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    return distance_matrix

def compute_color_similarity_distance_matrix(matrix):
    """
    Computes the color similarity distance matrix from the given matrix.
    Diagonal values are directly copied from the original matrix.

    :param matrix: A 12x12 numpy array with integer values ranging from -4 to 4
    :return: A 12x12 numpy array representing the color similarity distance matrix
    """
    # Ensure the input is a numpy array
    matrix = np.array(matrix)

    # Initialize the distance matrix with zeros
    distance_matrix = np.zeros_like(matrix, dtype=float)

    # Iterate over each pair of cells (i, j) and (j, i)
    for i in range(matrix.shape[0]):
        for j in range(i, matrix.shape[1]):
            if i == j:  # Diagonal values
                distance_matrix[i, j] = matrix[i, j]
            else:  # Off-diagonal values
                value1 = matrix[i, j]
                value2 = matrix[j, i]

                # Compute the average of the absolute values
                distance = (value1 + value2) / 2

                # Assign the distance to the distance matrix
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    return distance_matrix


def RSA(matrix1, matrix2, method='pearson'):
  upper_tri_1 = matrix1[np.triu_indices(matrix1.shape[0], k=1)]
  upper_tri_2 = matrix2[np.triu_indices(matrix2.shape[0], k=1)]
  if method == 'pearson':
    corr, _ = pearsonr(upper_tri_1, upper_tri_2)
  elif method == 'spearman':
    corr, _ = spearmanr(upper_tri_1, upper_tri_2)

  return corr

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

# Function to plot the embeddings
## Not used in this code
def plot_embeddings(embeddings, titles, color_labels, overlay=False):
    fig = go.Figure()
    
    if overlay:
        for i, embedding in enumerate(embeddings):
            fig.add_trace(go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode='markers+text',
                marker=dict(size=10, color=color_labels),
                text=color_labels,
                textposition="top center",
                name=titles[i]
            ))
    else:
        for i, embedding in enumerate(embeddings):
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode='markers+text',
                marker=dict(size=10, color=color_labels),
                text=color_labels,
                textposition="top center"
            ))
            fig.update_layout(
                title=f'MDS Embedding - {titles[i]}',
                scene=dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    zaxis_title='Dimension 3'
                ),
                height=800
            )
    fig.show()

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

def GWD_and_plot(matrix1, matrix2, epsilons):
    
    OT_plans = []
    gwds = []
    matching_rates = []

    for epsilon in epsilons:
      OT, gw_log = ot.gromov.entropic_gromov_wasserstein(C1=matrix1, C2=matrix2 , epsilon=epsilon, loss_fun="square_loss", log=True)  # optimization
      gwd = gw_log['gw_dist']
      matching_rate = comp_matching_rate(OT, k=1)  # calculate the top 1 matching rate
      OT_plans.append(OT)
      gwds.append(gwd)
      matching_rates.append(matching_rate)

      
    plt.scatter(epsilons, gwds, c=matching_rates)
    plt.xlabel("epsilon")
    plt.ylabel("GWD")
    plt.xscale('log')
    plt.grid(True, which = 'both')
    cbar = plt.colorbar()
    cbar.set_label(label='Matching Rate (%)')
    plt.show()

    # extract the best epsilon that minimizes the GWD
    min_gwd = min(gwds)
    best_eps_idx = gwds.index(min_gwd)
    best_eps = epsilons[best_eps_idx]
    OT_plan = OT_plans[best_eps_idx]
    matching_rate = matching_rates[best_eps_idx]

    show_heatmaps(0, 0.1, matrices=[OT_plan], titles=[f'Optimal transportation plan \n GWD={min_gwd:.3f} \n matching rate : {matching_rate:.1f}%'])

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
    
# Function to shuffle the upper and lower triangular parts of a matrix
def shuffle_upper_and_lower_triangular(matrix, size):
    # Create a copy of the matrix
    matrix_copy = matrix.copy()
    
    # Set the diagonal elements to zero
    np.fill_diagonal(matrix_copy, 0)

    # Shuffle the upper triangular elements
    upper_tri_indices = np.triu_indices(size, 1)
    upper_tri_values = matrix_copy[upper_tri_indices].copy()
    np.random.shuffle(upper_tri_values)
    matrix_copy[upper_tri_indices] = upper_tri_values
    
    # Shuffle the lower triangular elements
    lower_tri_indices = np.tril_indices(size, -1)
    lower_tri_values = matrix_copy[lower_tri_indices].copy()
    np.random.shuffle(lower_tri_values)
    matrix_copy[lower_tri_indices] = lower_tri_values
    
    # Return the shuffled matrix
    return matrix_copy

# Shuffle elements across rows except the diagonal elements
def shuffle_column_and_asymmetritisize(matrix, size):
    matrix_copy = matrix.copy()
    np.fill_diagonal(matrix_copy, 0)

    # Set the lower triangular part of the matrix to the negative of the upper triangular part
    for i in range(size):
        for j in range(i + 1, size):
            matrix_copy[j, i] = -matrix_copy[i, j]

    for i in range(size):
        non_diag_indices = [j for j in range(matrix_size) if j != i]
        non_diag_values = matrix_copy[non_diag_indices, i].copy()
        np.random.shuffle(non_diag_values)
        matrix_copy[non_diag_indices, i] = non_diag_values
    
    return matrix_copy

# Function to shuffle elements across rows except for the diagonal elements
def shuffle_row_and_asymmetritisize(matrix, size):
    matrix_copy = matrix.copy()
    
    # Set the diagonal elements to zero
    np.fill_diagonal(matrix_copy, 0)

    # Shuffle each row except for the diagonal elements
    for i in range(size):
        non_diag_indices = [j for j in range(size) if j != i]
        non_diag_values = matrix_copy[i, non_diag_indices].copy()
        np.random.shuffle(non_diag_values)
        matrix_copy[i, non_diag_indices] = non_diag_values
    for i in range(size):
        for j in range(i + 1, size):
            matrix_copy[j, i] = -matrix_copy[i, j]

    return matrix_copy