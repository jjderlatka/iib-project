from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from pathlib import Path
import pickle

import numpy as np


def transform_dictionary(dict_by_sizes):
    merged_timestamps = {}
    for size, timestamps_all_ranks in dict_by_sizes.items():
        for rank, timestamps_this_rank in enumerate(timestamps_all_ranks):
            for label, timing in timestamps_this_rank.items():
                if label not in merged_timestamps:
                    merged_timestamps[label] = {}
                if size not in merged_timestamps[label]:
                    merged_timestamps[label][size] = [None] * size
                merged_timestamps[label][size][rank] = timing

    return merged_timestamps


def extend_to_lists(label_dictionary):
    X, Y = [], []
    for size, times in label_dictionary.items():
        X.extend(times)
        Y.extend([size] * len(times))

    return X, Y


def plot_timings():
    results_folder = Path("results/01")
    filename = 'timing.pkl'

    with open(results_folder/filename, 'rb') as f:
        all_results = pickle.load(f)

    all_results = transform_dictionary(all_results)
    for label, label_data in all_results.items():
        X, Y = extend_to_lists(label_data)
        plt.scatter(X, Y, label=label)
    plt.legend()
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Processes number")
    plt.xlabel("Time [ns]")
    plt.title("216/216 Parameters")
    plt.show()


def save(name):
    results_folder = Path("results")
    file_path = results_folder / name
    plt.savefig(file_path, bbox_inches='tight')

def plot_samples():
    def plot_samples_3d(box_dims, points):
        # Extract box dimensions
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = box_dims

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the box as lines
        box_lines = [
            [(x_min, y_min, z_min), (x_max, y_min, z_min)],
            [(x_min, y_max, z_min), (x_max, y_max, z_min)],
            [(x_min, y_min, z_max), (x_max, y_min, z_max)],
            [(x_min, y_max, z_max), (x_max, y_max, z_max)],
            [(x_min, y_min, z_min), (x_min, y_max, z_min)],
            [(x_max, y_min, z_min), (x_max, y_max, z_min)],
            [(x_min, y_min, z_max), (x_min, y_max, z_max)],
            [(x_max, y_min, z_max), (x_max, y_max, z_max)],
            [(x_min, y_min, z_min), (x_min, y_min, z_max)],
            [(x_max, y_min, z_min), (x_max, y_min, z_max)],
            [(x_min, y_max, z_min), (x_min, y_max, z_max)],
            [(x_max, y_max, z_min), (x_max, y_max, z_max)]
        ]

        for line in box_lines:
            ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]], 'k-')

        # Plot the points
        for point in points:
            ax.scatter(point[0], point[1], point[2], c='r', marker='o')

        # Set plot limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        save('samples.png')


    def plot_samples_projections(box_dims, points):
        # Extract box dimensions
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = box_dims

        # Create a 2D plot for each projection
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # XY Projection
        axs[0].set_title('XY Projection')
        axs[0].set_xlim(x_min, x_max)
        axs[0].set_ylim(y_min, y_max)
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        for point in points:
            axs[0].scatter(point[0], point[1], c='r', marker='o')
        axs[0].plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'k-')

        # XZ Projection
        axs[1].set_title('XZ Projection')
        axs[1].set_xlim(x_min, x_max)
        axs[1].set_ylim(z_min, z_max)
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Z')
        for point in points:
            axs[1].scatter(point[0], point[2], c='r', marker='o')
        axs[1].plot([x_min, x_max, x_max, x_min, x_min], [z_min, z_min, z_max, z_max, z_min], 'k-')

        # YZ Projection
        axs[2].set_title('YZ Projection')
        axs[2].set_xlim(y_min, y_max)
        axs[2].set_ylim(z_min, z_max)
        axs[2].set_xlabel('Y')
        axs[2].set_ylabel('Z')
        for point in points:
            axs[2].scatter(point[1], point[2], c='r', marker='o')
        axs[2].plot([y_min, y_max, y_max, y_min, y_min], [z_min, z_min, z_max, z_max, z_min], 'k-')

        plt.tight_layout()
        save('samples_projected.png')

    with open('results/samples.npy', 'rb') as f:
        ranges = np.load(f)
        global_training_set = np.load(f)

    plot_samples_3d(ranges, global_training_set)
    plot_samples_projections(ranges, global_training_set)

def plot_eigenvalue_decays():
    def plot_eigenvalue_decay(ax, eigenvalues, number_of_modes, title):
        positive_eigenvalues = np.where(eigenvalues > 0., eigenvalues, np.nan)
        singular_values = np.sqrt(positive_eigenvalues)

        xint = list()
        yval = list()

        for x, y in enumerate(eigenvalues[:number_of_modes]):
            yval.append(y)
            xint.append(x+1)

        ax.plot(xint, yval, "^-", color="tab:blue")
        ax.set_xlabel("Eigenvalue number", fontsize=18)
        ax.set_ylabel("Eigenvalue", fontsize=18)
        ax.set_xticks(xint)
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.set_yscale("log")
        ax.set_title(f"{title}", fontsize=24)

    with open('results/eigen.npy', 'rb') as f:
        [eigenvalues_u, eigenvalues_p] = np.load(f)
        [number_of_modes_u, number_of_modes_p] = np.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    plot_eigenvalue_decay(ax1, eigenvalues_u, number_of_modes_u, "Velocity eigenvalues decay")
    plot_eigenvalue_decay(ax2, eigenvalues_p, number_of_modes_p, "Pressure eigenvalues decay")
    
    plt.tight_layout()
    save('eigenvalue_decay.png')

if __name__ == "__main__":
    plot_eigenvalue_decays()