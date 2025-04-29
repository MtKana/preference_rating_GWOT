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
import itertools
import random

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
    
def load_csv_to_matrix(file_path, response_type, colour_index, matrix_size):
    df = pd.read_csv(file_path)
    df = df[(df['practice_trial'] != 1) & (df['response_type'] == response_type)]

    colour1 = df['colour1']
    colour2 = df['colour2']
    target_preference = df['response']

    matrix = np.zeros((matrix_size, matrix_size))
    for c1, c2, tp in zip(colour1, colour2, target_preference):
        I = colour_index[c1]
        j = colour_index[c2]
        matrix[I, j] = tp

    return matrix.astype(int)

def load_csv_to_matrix_batch(folder_path, response_type, colour_index, matrix_size):
    subject_matrices = []
    files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    )
    
    print("Order of files being loaded:")
    for f in files:
        print(f)
    
    for file in files:
        subject_matrix = load_csv_to_matrix(file, response_type, colour_index, matrix_size)
        subject_matrices.append(subject_matrix)
    
    return subject_matrices

def split_and_average_matrices(matrices):

    def compute_all_splits(matrices):
        """Computes all possible splits of the matrices into two equal-sized groups."""
        num_matrices = len(matrices)
        half_size = num_matrices // 2
        all_possible_splits = list(itertools.combinations(matrices, half_size))
        print(f"Number of possible splits: {len(all_possible_splits)}")
        
        results = []
        checked_splits = set()

        for group1 in all_possible_splits:
            group2 = [m for m in matrices if not any(np.array_equal(m, g) for g in group1)]

            # Convert to frozenset to ensure unique groups
            if frozenset(map(id, group1)) in checked_splits:
                continue
            print("Checked splits")

            checked_splits.add(frozenset(map(id, group1)))

            avg_matrix1 = np.mean(np.array(group1), axis=0)
            print("Averaged group 1")
            avg_matrix2 = np.mean(np.array(group2), axis=0)
            print("Averaged group 2")

            results.append((avg_matrix1, avg_matrix2))
            print("Appended results")

        return results
    num_matrices = len(matrices)
    results = []

    # If the number of matrices is odd, try all possible removals
    if num_matrices % 2 == 1:
        for removed_matrix in matrices:
            remaining_matrices = [m for m in matrices if not np.array_equal(m, removed_matrix)]
            results.extend(compute_all_splits(remaining_matrices))
    else:
        results.extend(compute_all_splits(matrices))

    return results



# def compute_color_preference_distance_batch(matrix_list):
#     """
#     Transforms each 2D numpy array in the input list as follows:
#     1. Applies the given mapping to transform values.
#     2. Replaces the lower triangular values (below the diagonal) with their negative values.
#     3. Computes the average of values across the diagonal and replaces them with their absolute value.
#     4. Takes the absolute value of the diagonal elements.
#     """
#     transformed_matrices = []

#     value_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: -1, 5: -2, 6: -3, 7: -4}
#     vectorized_mapping = np.vectorize(lambda x: value_map.get(x, x))
    
#     for matrix in matrix_list:
#         vectorized_mapping = np.vectorize(lambda x: value_map.get(x, x))
#         transformed_matrix = vectorized_mapping(matrix)

#         lower_triangle_indices = np.tril_indices_from(transformed_matrix, k=-1)
#         transformed_matrix[lower_triangle_indices] *= -1

#         for i in range(transformed_matrix.shape[0]):
#             for j in range(i+1, transformed_matrix.shape[1]): 
#                 avg_value = (transformed_matrix[i, j] + transformed_matrix[j, i]) / 2
#                 transformed_matrix[i, j] = transformed_matrix[j, i] = abs(avg_value)

#         np.fill_diagonal(transformed_matrix, np.abs(np.diagonal(transformed_matrix)))

#         transformed_matrices.append(transformed_matrix)
    
#     return transformed_matrices

def compute_color_preference_raw_batch(matrix_list, value_range_max = 3.5):
    transformed_matrices = []
    if value_range_max == 3.5:
        value_map = {0: 3.5, 1: 2.5, 2: 1.5, 3: 0.5, 4: -0.5, 5: -1.5, 6: -2.5, 7: -3.5}
    if value_range_max == 4:
        value_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: -1, 5: -2, 6: -3, 7: -4}

    vectorized_mapping = np.vectorize(lambda x: value_map.get(x, x))

    for matrix in matrix_list:
        vectorized_mapping = np.vectorize(lambda x: value_map.get(x, x))
        transformed_matrix = vectorized_mapping(matrix)

        transformed_matrices.append(transformed_matrix)
    
    return transformed_matrices


def compute_color_preference_distance_batch(matrix_list, value_range_max=3.5):
    """
    Transforms each 2D numpy array in the input list as follows:
    1. Applies the given mapping to transform values.
    2. Replaces the lower triangular values (below the diagonal) with their negative values.
    3. Computes the average of values across the diagonal and replaces them with their absolute value.
    4. Takes the absolute value of the diagonal elements.
    """
    transformed_matrices = []
    
    if value_range_max == 3.5:
        value_map = {0: 3.5, 1: 2.5, 2: 1.5, 3: 0.5, 4: -0.5, 5: -1.5, 6: -2.5, 7: -3.5}
    if value_range_max == 4:
        value_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: -1, 5: -2, 6: -3, 7: -4}
    
    vectorized_mapping = np.vectorize(lambda x: value_map.get(x, x))
    
    for matrix in matrix_list:
        vectorized_mapping = np.vectorize(lambda x: value_map.get(x, x))
        transformed_matrix = vectorized_mapping(matrix)

        lower_triangle_indices = np.tril_indices_from(transformed_matrix, k=-1)
        transformed_matrix[lower_triangle_indices] *= -1

        for i in range(transformed_matrix.shape[0]):
            for j in range(i+1, transformed_matrix.shape[1]): 
                avg_value = (transformed_matrix[i, j] + transformed_matrix[j, i]) / 2
                transformed_matrix[i, j] = transformed_matrix[j, i] = abs(avg_value)

        np.fill_diagonal(transformed_matrix, np.abs(np.diagonal(transformed_matrix)))
        transformed_matrices.append(transformed_matrix)
    
    return transformed_matrices

def compute_color_similarity_distance_batch(matrix_list):
    """
    Transforms each 2D numpy array in the input list as follows:
    1. Transform the values using the value map.
    """
    transformed_matrices = []

    value_map = {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: 0}

    vectorized_mapping = np.vectorize(lambda x: value_map.get(x, x))
    
    for matrix in matrix_list:
        vectorized_mapping = np.vectorize(lambda x: value_map.get(x, x))
        transformed_matrix = vectorized_mapping(matrix)

        transformed_matrices.append(transformed_matrix)
    
    return transformed_matrices

def show_heatmaps(vmin_val, vmax_val, matrices, titles, nrows, ncols, cmap_name, cbar_label=None, color_labels=None):
    def add_colored_label(ax, x, y, bgcolor, width=1, height=1):
        rect = Rectangle((x, y), width, height, facecolor=bgcolor)
        ax.add_patch(rect)  

    num_plots = len(matrices)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axs = np.array(axs).reshape(-1)  # Flatten the axes array

    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = axs[i]
        
        # Set the divergent colormap here
        im = ax.imshow(matrix, aspect='auto', vmin=vmin_val, vmax=vmax_val, cmap = cmap_name)
        
        ax.set_title(title, fontsize=23)
        ax.set_xlabel("Right", fontsize=23) 
        ax.set_ylabel("Left", fontsize=23)  
        ax.set_xticks([])
        ax.set_yticks([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, fontsize=21)
        cbar.ax.tick_params(labelsize=21)

        # Build tick list including 0, and avoid duplicates
        tick_vals = sorted(set([vmin_val, 0, vmax_val]))
        cbar.set_ticks(tick_vals)
        cbar.ax.set_yticklabels([f"{val:.2f}" for val in tick_vals])

        # Adjust the height of the color bar
        position = cax.get_position()
        new_position = [position.x0, position.y0 + 0.1, position.width, position.height * 0.8]
        cax.set_position(new_position)

        if color_labels is not None:
            for idx, color in enumerate(color_labels):
                add_colored_label(ax, -1.5, idx-0.5, color, width=0.8)
                add_colored_label(ax, idx-0.5, matrix.shape[1] - 0.2, color, height=0.8)

            ax.set_aspect('equal')
            ax.set_xlim(-3.0, matrix.shape[1])
            ax.set_ylim(-1, len(color_labels)+1.4)
            ax.invert_yaxis()

            for spine in ax.spines.values():
                spine.set_visible(False)

    # Hide unused subplots if nrows*ncols > num_plots
    for j in range(num_plots, nrows * ncols):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.show()

    
# def show_heatmaps(vmin_val, vmax_val, matrices, titles, cbar_label=None, color_labels=None):
#     num_plots = len(matrices)
#     fig, axs = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))

#     if num_plots == 1:
#         axs = [axs]

#     for i, (matrix, title) in enumerate(zip(matrices, titles)):
#         ax = axs[i]
        
#         im = ax.imshow(matrix, aspect='auto', vmin=vmin_val, vmax=vmax_val)
#         ax.set_title(title)

#         # Set axis labels
#         ax.set_xlabel("Right")  # Label for x-axis
#         ax.set_ylabel("Left")   # Label for y-axis

#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         cbar = fig.colorbar(im, cax=cax)
#         cbar.set_label(cbar_label)

#         # Adjust the height of the color bar
#         position = cax.get_position()
#         new_position = [position.x0, position.y0 + 0.1, position.width, position.height * 0.8]
#         cax.set_position(new_position)

#         if color_labels is not None:
#             ax.axis('off')
#             for idx, color in enumerate(color_labels):
#                 add_colored_label(ax, -1.5, idx-0.5, color, width=0.8)
#                 add_colored_label(ax, idx-0.5, matrix.shape[1] - 0.2, color, height=0.8)

#             ax.set_aspect('equal')
#             ax.set_xlim(-3.0, matrix.shape[1])
#             ax.set_ylim(-1, len(color_labels)+1.4)
#             ax.invert_yaxis()

#             for spine in ax.spines.values():
#                 spine.set_visible(False)

#     plt.tight_layout()
#     plt.show()

# def show_heatmap(matrix, title, cbar_label=None, color_labels=None):
#     fig, ax = plt.subplots(figsize=(5, 5))

#     im = ax.imshow(matrix, aspect='auto', vmin=0, vmax=7)
#     ax.set_title(title)

#     # Set axis labels
#     ax.set_xlabel("Right")  # Label for x-axis
#     ax.set_ylabel("Left")   # Label for y-axis

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = fig.colorbar(im, cax=cax)
#     cbar.set_label(cbar_label)

#     # Adjust the height of the color bar
#     position = cax.get_position()
#     new_position = [position.x0, position.y0 + 0.1, position.width, position.height * 0.8]
#     cax.set_position(new_position)

#     if color_labels is not None:
#         ax.axis('off')
#         for idx, color in enumerate(color_labels):
#             add_colored_label(ax, -1.5, idx - 0.5, color, width=0.8)
#             add_colored_label(ax, idx - 0.5, matrix.shape[1] - 0.2, color, height=0.8)

#         ax.set_aspect('equal')
#         ax.set_xlim(-3.0, matrix.shape[1])
#         ax.set_ylim(-1, len(color_labels) + 1.4)
#         ax.invert_yaxis()

#         for spine in ax.spines.values():
#             spine.set_visible(False)

#     plt.tight_layout()
#     plt.show()

# def compute_color_preference_distance_matrix(matrix):
#     """
#     Computes the color preference distance matrix from the given matrix.
#     Diagonal values are directly copied from the original matrix.

#     :param matrix: A 12x12 numpy array with integer values ranging from -4 to 4
#     :return: A 12x12 numpy array representing the color preference distance matrix
#     """
#     # Ensure the input is a numpy array
#     matrix = np.array(matrix)

#     # Initialize the distance matrix with zeros
#     distance_matrix = np.zeros_like(matrix, dtype=float)

#     # Iterate over each pair of cells (i, j) and (j, i)
#     for i in range(matrix.shape[0]):
#         for j in range(i, matrix.shape[1]):
#             if i == j:  # Diagonal values
#                 distance_matrix[i, j] = matrix[i, j]
#             else:  # Off-diagonal values
#                 value1 = matrix[i, j]
#                 value2 = matrix[j, i]

#                 # Special case: if values are -1 and 1, set distance to 0
#                 if (value1 == -1 and value2 == 1) or (value1 == 1 and value2 == -1):
#                     distance = 0
#                 else:
#                     # Flip one value if both are positive or both are negative
#                     if (value1 > 0 and value2 > 0) or (value1 < 0 and value2 < 0):
#                         value2 = -value2

#                     # Compute the average of the absolute values
#                     distance = (abs(value1) + abs(value2)) / 2

#                 # Assign the distance to the distance matrix
#                 distance_matrix[i, j] = distance
#                 distance_matrix[j, i] = distance

#     return distance_matrix

# def compute_color_similarity_distance_matrix(matrix):
#     """
#     Computes the color similarity distance matrix from the given matrix.
#     Diagonal values are directly copied from the original matrix.

#     :param matrix: A 12x12 numpy array with integer values ranging from -4 to 4
#     :return: A 12x12 numpy array representing the color similarity distance matrix
#     """
#     # Ensure the input is a numpy array
#     matrix = np.array(matrix)

#     # Initialize the distance matrix with zeros
#     distance_matrix = np.zeros_like(matrix, dtype=float)

#     # Iterate over each pair of cells (i, j) and (j, i)
#     for i in range(matrix.shape[0]):
#         for j in range(i, matrix.shape[1]):
#             if i == j:  # Diagonal values
#                 distance_matrix[i, j] = matrix[i, j]
#             else:  # Off-diagonal values
#                 value1 = matrix[i, j]
#                 value2 = matrix[j, i]

#                 # Compute the average of the absolute values
#                 distance = (value1 + value2) / 2

#                 # Assign the distance to the distance matrix
#                 distance_matrix[i, j] = distance
#                 distance_matrix[j, i] = distance
#     return distance_matrix

def RSA(matrix1, matrix2, method='pearson'):
    flat_1 = matrix1.flatten()
    flat_2 = matrix2.flatten()

    if method == 'pearson':
        corr, _ = pearsonr(flat_1, flat_2)
    elif method == 'spearman':
        corr, _ = spearmanr(flat_1, flat_2)
    else:
        raise ValueError("Invalid method. Choose 'pearson' or 'spearman'.")

    return corr

def compute_correlations(matrices):
    matrix_names = list(matrices.keys())
    correlation_matrix = np.zeros((len(matrix_names), len(matrix_names)))
    
    # Compute correlation matrix
    for i in range(len(matrix_names)):
        for j in range(i, len(matrix_names)):
            mat1, mat2 = matrices[matrix_names[i]], matrices[matrix_names[j]]
            correlation = RSA(mat1, mat2)  
            correlation_matrix[i, j] = correlation_matrix[j, i] = correlation

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        correlation_matrix, 
        annot=True, 
        xticklabels=matrix_names, 
        yticklabels=matrix_names, 
        cmap='coolwarm', 
        fmt=".2f", 
        square=True, 
        linewidths=0.5, 
        annot_kws={"size": 22},  
        vmin=-1, vmax=1,  
        center=0 
    )

    plt.xticks(rotation=45, ha="right", fontsize=22)
    plt.yticks(rotation=0, fontsize=22)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)  
    cbar.set_label('r', fontsize=27)  

    plt.show()
    
    return correlation_matrix, matrix_names


def perform_mds_and_plot(matrices, titles, colour_index, n_rows, n_cols, n_components=2):
    """
    Performs Multidimensional Scaling (MDS) on a list of distance matrices and visualizes the
    resulting 2D embeddings in a grid of scatter plots.

    Parameters:
    ----------
    matrices : list[np.ndarray]
        A list of square distance matrices (numpy arrays). Each matrix should represent
        pairwise dissimilarities between entities.

    titles : list[str]
        A list of titles corresponding to each matrix. Must be the same length as `matrices`.

    colour_index : dict[str, Any]
        A dictionary mapping item labels to colors or indices for plotting.
        The keys of this dictionary are used as labels in the scatter plots.

    n_rows : int
        The number of rows in the subplot grid.

    n_cols : int
        The number of columns in the subplot grid.

    n_components : int, optional (default=2)
        The number of dimensions to reduce the data to using MDS.
        Typically 2 for 2D visualization.

    Returns:
    -------
    None
        Displays a matplotlib plot containing the MDS embeddings.
    """
    assert len(matrices) == len(titles), "Number of matrices must match number of titles"

    mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=42)
    
    num_matrices = len(matrices)
    
    square_size = 5
    fig_width = max(n_cols * square_size, n_rows * square_size)
    fig_height = fig_width

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.array(axes).flatten()

    color_labels = list(colour_index.keys())
    all_embeddings = []

    for matrix in matrices:
        dist_matrix = (matrix + matrix.T) / 2  # Ensure symmetry
        embedding = mds.fit_transform(dist_matrix)
        all_embeddings.append(embedding)

    all_points = np.vstack(all_embeddings)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    axis_limit = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

    for ax, title, embedding in zip(axes[:num_matrices], titles, all_embeddings):
        colors = list(colour_index.keys())
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=400)

        for i, label in enumerate(colors):
            ax.text(embedding[i, 0], embedding[i, 1], label, fontsize=14)

        ax.set_title(title, fontsize=18)
        ax.set_xlabel("MDS Dimension 1", fontsize=16)
        ax.set_ylabel("MDS Dimension 2", fontsize=16)
        ax.tick_params(axis='both', labelsize=15)
        ax.set_aspect('equal')
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)

    for ax in axes[num_matrices:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    
# # Function to shuffle the upper and lower triangular parts of a matrix
# def shuffle_upper_and_lower_triangular(matrix, size):
#     # Create a copy of the matrix
#     matrix_copy = matrix.copy()
    
#     # Set the diagonal elements to zero
#     np.fill_diagonal(matrix_copy, 0)

#     # Shuffle the upper triangular elements
#     upper_tri_indices = np.triu_indices(size, 1)
#     upper_tri_values = matrix_copy[upper_tri_indices].copy()
#     np.random.shuffle(upper_tri_values)
#     matrix_copy[upper_tri_indices] = upper_tri_values
    
#     # Shuffle the lower triangular elements
#     lower_tri_indices = np.tril_indices(size, -1)
#     lower_tri_values = matrix_copy[lower_tri_indices].copy()
#     np.random.shuffle(lower_tri_values)
#     matrix_copy[lower_tri_indices] = lower_tri_values
    
#     # Return the shuffled matrix
#     return matrix_copy

# # Shuffle elements across rows except the diagonal elements
# def shuffle_column_and_asymmetritisize(matrix, size):
#     matrix_copy = matrix.copy()
#     np.fill_diagonal(matrix_copy, 0)

#     # Set the lower triangular part of the matrix to the negative of the upper triangular part
#     for i in range(size):
#         for j in range(i + 1, size):
#             matrix_copy[j, i] = -matrix_copy[i, j]

#     for i in range(size):
#         non_diag_indices = [j for j in range(matrix_size) if j != i]
#         non_diag_values = matrix_copy[non_diag_indices, i].copy()
#         np.random.shuffle(non_diag_values)
#         matrix_copy[non_diag_indices, i] = non_diag_values
    
#     return matrix_copy

# # Function to shuffle elements across rows except for the diagonal elements
# def shuffle_row_and_asymmetritisize(matrix, size):
#     matrix_copy = matrix.copy()
    
#     # Set the diagonal elements to zero
#     np.fill_diagonal(matrix_copy, 0)

#     # Shuffle each row except for the diagonal elements
#     for i in range(size):
#         non_diag_indices = [j for j in range(size) if j != i]
#         non_diag_values = matrix_copy[i, non_diag_indices].copy()
#         np.random.shuffle(non_diag_values)
#         matrix_copy[i, non_diag_indices] = non_diag_values
#     for i in range(size):
#         for j in range(i + 1, size):
#             matrix_copy[j, i] = -matrix_copy[i, j]

#     return matrix_copy

# # Define unique colors
# def getUniqueColours():
#     unique_colours = np.array(['#d2b700', '#db8b08', '#c7512c', '#c13547', '#a03663', '#753a7a', '#4b488e', '#005692', '#006a8b', '#007b75', '#008a52', '#9aa400'])
#     colour_index = {colour: idx for idx, colour in enumerate(unique_colours)}
#     return colour_index