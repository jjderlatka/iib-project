from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import csv
from collections import defaultdict
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
    results_folder = Path("results")
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
    # save('timing.png')


def save(name):
    results_folder = Path("results")
    file_path = results_folder / name
    plt.savefig(file_path, bbox_inches='tight', dpi=600)


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

        # Plot the points
        for point in points:
            ax.scatter(point[0], point[1], point[2], zorder=1, marker='o', edgecolors='darkblue', facecolors='lightblue')

        for line in box_lines:
            ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]], 'k-', zorder=2)

        # Set plot limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Labels
        ax.set_xlabel('$a$')
        ax.set_ylabel('$b$')
        ax.set_zlabel('$\\theta$')

        save('samples.png')


    def plot_samples_projections(box_dims, points):
        # Extract box dimensions
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = box_dims

        # Create a 2D plot for each projection
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # XY Projection
        axs[0].set_title('$ab$ Projection')
        axs[0].set_xlim(x_min, x_max)
        axs[0].set_ylim(y_min, y_max)
        axs[0].set_xlabel('$a$')
        axs[0].set_ylabel('$b$')
        axs[0].grid(True, linestyle='dashed')
        axs[0].set_axisbelow(True)
        for point in points:
            axs[0].scatter(point[0], point[1], marker='o', edgecolors='darkblue', facecolors='lightblue')
        axs[0].plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'k-')

        # XZ Projection
        axs[1].set_title('$a\\theta$ Projection')
        axs[1].set_xlim(x_min, x_max)
        axs[1].set_ylim(z_min, z_max)
        axs[1].set_xlabel('$a$')
        axs[1].set_ylabel('$\\theta$')
        axs[1].grid(True, linestyle='dashed')
        axs[1].set_axisbelow(True)
        for point in points:
            axs[1].scatter(point[0], point[2], marker='o', edgecolors='darkblue', facecolors='lightblue')
        axs[1].plot([x_min, x_max, x_max, x_min, x_min], [z_min, z_min, z_max, z_max, z_min], 'k-')

        # YZ Projection
        axs[2].set_title('$b\\theta$ Projection')
        axs[2].set_xlim(y_min, y_max)
        axs[2].set_ylim(z_min, z_max)
        axs[2].set_xlabel('$b$')
        axs[2].set_ylabel('$\\theta$')
        axs[2].grid(True, linestyle='dashed')
        axs[2].set_axisbelow(True)
        for point in points:
            axs[2].scatter(point[1], point[2], marker='o', edgecolors='darkblue', facecolors='lightblue')
        axs[2].plot([y_min, y_max, y_max, y_min, y_min], [z_min, z_min, z_max, z_max, z_min], 'k-')

        # plt.tight_layout()
        save('samples_projected.png')


    with open('results/samples.npy', 'rb') as f:
        ranges = np.load(f)
        global_training_set = np.load(f)

    plot_samples_3d(ranges, global_training_set)
    plot_samples_projections(ranges, global_training_set)

def plot_eigenvalue_decays():
    def plot_eigenvalue_decay(eigenvalues, number_of_modes, title, filename):
        # fig, ax = plt.subplots(figsize=(8, 10))
        fig, ax = plt.subplots()

        positive_eigenvalues = np.where(eigenvalues > 0., eigenvalues, np.nan)
        xint = range(1, number_of_modes + 1)

        ax.plot(xint, eigenvalues[:number_of_modes], "-", color='darkblue')
        ax.scatter(xint, eigenvalues[:number_of_modes], color='lightblue', edgecolor='darkblue', marker='^', zorder=3)
        
        ax.set_xlabel("Eigenvalue number", fontsize=18)
        ax.set_ylabel("Eigenvalue", fontsize=18)
    
        # Make sure ticks start from 1, not 0
        ax.set_xlim(left=0, right=number_of_modes + 1)  # Slightly adjusted for better visibility of the first tick
        ticks = ax.get_xticks()
        if 1 not in ticks:
            ticks = np.sort(np.append(ticks, 1))
        if 0 in ticks:
            ticks = np.sort(np.delete(ticks, np.argwhere(ticks==0)))
        ax.set_xticks(ticks)

        ax.set_yscale("log")
        ax.grid(True, linestyle='dashed')
        ax.grid(True, which='minor', axis='y', linestyle='dashed', linewidth=0.5)  # Enhance visibility of log scale grid lines
        ax.set_axisbelow(True)
        # ax.set_title(title, fontsize=24)

        plt.tight_layout()
        save(filename)

    
    with open('results/eigen.npy', 'rb') as f:
        [eigenvalues_u, eigenvalues_p] = np.load(f)
        [number_of_modes_u, number_of_modes_p] = np.load(f)

    plot_eigenvalue_decay(eigenvalues_u, number_of_modes_u, "Velocity eigenvalues decay", 'velocity_eigenvalue_decay.png')
    plot_eigenvalue_decay(eigenvalues_p, number_of_modes_p, "Pressure eigenvalues decay", 'pressure_eigenvalue_decay.png')


def plot_error_evolution():
    error_evolution = {}
    with open('results/error_evolution_velocity.npy', 'rb') as f:
        error_evolution["norm_error_deformed_u"] = np.load(f)
        error_evolution["norm_error_u"] = np.load(f)

    with open('results/error_evolution_pressure.npy', 'rb') as f:
        error_evolution["norm_error_deformed_p"] = np.load(f)
        error_evolution["norm_error_p"] = np.load(f)

    fig, axs = plt.subplots(1, 4, figsize=(16, 10))
    for i, (name, values) in enumerate(error_evolution.items()):
        axs[i].plot(values)

    save('error_evolution.png')


def compare_reuse():
    def read_data(filename):
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = defaultdict(list)
            for row in reader:
                data[int(row['N'])].append((float(row['error_u']), float(row['error_p'])))
        return data
    
    def compute_stats(data):
        stats = {}
        for N, values in data.items():
            u_values, p_values = zip(*values)
            stats[N] = {
                'u_mean': np.mean(u_values),
                'u_var': np.var(u_values),
                'u_std': np.std(u_values),
                'p_mean': np.mean(p_values),
                'p_var': np.var(p_values),
                'p_std': np.std(p_values)
            }
        return stats
    
    def plot_comparisons(stats1, stats2, variable):
        Ns = sorted(stats1.keys())
        means1 = [stats1[N][f'{variable}_mean'] for N in Ns]
        means2 = [stats2[N][f'{variable}_mean'] for N in Ns]
        stds1 = [stats1[N][f'{variable}_std'] for N in Ns]
        stds2 = [stats2[N][f'{variable}_std'] for N in Ns]

        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot the means with lines
        ax.plot(Ns, means1, "-", color='darkblue', label=f'Reuse {variable} Means')
        ax.plot(Ns, means2, "-", color='darkorange', label=f'No reuse {variable} Means')
        
        # Plot the error bars
        ax.errorbar(Ns, means1, yerr=stds1, fmt='none', ecolor='lightblue', capsize=5, zorder=2)
        ax.errorbar(Ns, means2, yerr=stds2, fmt='none', ecolor='orange', capsize=5, zorder=2)
        
        # Plot the scatter points
        ax.scatter(Ns, means1, color='lightblue', edgecolor='darkblue', marker='^', zorder=3)
        ax.scatter(Ns, means2, color='orange', edgecolor='darkorange', marker='s', zorder=3)
        
        ax.set_xlabel('N')
        ax.set_ylabel(f'Mean of {variable}')
        ax.set_title(f'Comparison of {variable} Means Across Different N')
        ax.legend()
        
        save(f"compare_reuse_{variable}.png")

    # Read data from both CSV files
    data_reuse = read_data('reuse.csv')
    data_no_reuse = read_data('no_reuse.csv')

    # Compute statistics for both datasets
    stats_reuse = compute_stats(data_reuse)
    stats_no_reuse = compute_stats(data_no_reuse)

    # Plot comparisons
    plot_comparisons(stats_reuse, stats_no_reuse, 'u')
    plot_comparisons(stats_reuse, stats_no_reuse, 'p')


if __name__ == "__main__":
    plot_eigenvalue_decays()