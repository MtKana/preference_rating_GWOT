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
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    for file in files:
        subject_matrix = load_csv_to_matrix(file, response_type, colour_index, matrix_size)
        subject_matrices.append(subject_matrix)
    
    return subject_matrices

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

def compute_color_preference_distance_batch(matrix_list):
    """
    Transforms each 2D numpy array in the input list as follows:
    1. Applies the given mapping to transform values.
    2. Replaces the lower triangular values (below the diagonal) with their negative values.
    3. Computes the average of values across the diagonal and replaces them with their absolute value.
    4. Takes the absolute value of the diagonal elements.
    """
    transformed_matrices = []

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
    1. Applies the given mapping to transform values.
    2. Replaces the lower triangular values (below the diagonal) with their negative values.
    3. Computes the average of values across the diagonal and replaces them with their absolute value.
    4. Takes the absolute value of the diagonal elements.
    """
    transformed_matrices = []

    value_map = {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: 0}
    vectorized_mapping = np.vectorize(lambda x: value_map.get(x, x))
    
    for matrix in matrix_list:
        vectorized_mapping = np.vectorize(lambda x: value_map.get(x, x))
        transformed_matrix = vectorized_mapping(matrix)

        transformed_matrices.append(transformed_matrix)
    
    return transformed_matrices

# Kana's implementation of plotting heatmaps
def add_colored_label(ax, x, y, bgcolor, width=1, height=1):
    rect = Rectangle((x, y), width, height, facecolor=bgcolor)
    ax.add_patch(rect)  

def show_heatmaps(vmin_val, vmax_val, matrices, titles, nrows, ncols, cbar_label=None, color_labels=None):
    def add_colored_label(ax, x, y, bgcolor, width=1, height=1):
        rect = Rectangle((x, y), width, height, facecolor=bgcolor)
        ax.add_patch(rect)  

    num_plots = len(matrices)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axs = np.array(axs).reshape(-1)  # Flatten the axes array
    
    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = axs[i]
        im = ax.imshow(matrix, aspect='auto', vmin=vmin_val, vmax=vmax_val)
        ax.set_title(title, fontsize=17)
        ax.set_xlabel("Right", fontsize=15) 
        ax.set_ylabel("Left", fontsize=15)  
        ax.set_xticks([])
        ax.set_yticks([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, fontsize=15)
        cbar.ax.tick_params(labelsize=15)

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
  upper_tri_1 = matrix1[np.triu_indices(matrix1.shape[0], k=1)]
  upper_tri_2 = matrix2[np.triu_indices(matrix2.shape[0], k=1)]
  if method == 'pearson':
    corr, _ = pearsonr(upper_tri_1, upper_tri_2)
  elif method == 'spearman':
    corr, _ = spearmanr(upper_tri_1, upper_tri_2)

  return corr

# Function to plot the embeddings
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