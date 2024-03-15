from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from pathlib import Path
import pickle


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


if __name__ == "__main__":
    plot_timings()