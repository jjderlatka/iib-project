from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

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

    markers = ['o', 'o', 'o', 'o', 'o', '<', '>']  # Different marker shapes
    edge_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'orange', 'gray']  # Edge colors
    fill_colors = ['lightblue', 'navajowhite', 'limegreen', 'salmon', 'mediumpurple', 'gray', 'black']  # Fill colors

    with open(results_folder/filename, 'rb') as f:
        all_results = pickle.load(f)

    all_results = transform_dictionary(all_results)
    for (label, label_data), marker, edge_color, fill_color in zip(all_results.items(), markers, edge_colors, fill_colors):
        if label != 'NN trained' and label != 'Error analysis completed':
            X, Y = extend_to_lists(label_data)
            plt.scatter(X, Y, label=label, marker=marker, edgecolors=edge_color, facecolors=fill_color)
    plt.legend()
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Processes number")
    plt.xlabel("Time [ns]")
    plt.grid(True, linestyle='dashed')
    # plt.title("216/216 Parameters")

    # plt.show()
    save('timing.png')


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

        ax.plot(xint, eigenvalues[:number_of_modes], "-", color='tab:blue')
        ax.scatter(xint, eigenvalues[:number_of_modes], color='lightblue', edgecolor='tab:blue', marker='^', zorder=3)
        
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

    # fig, axs = plt.subplots(1, 4, figsize=(16, 10))
    # for i, (name, values) in enumerate(error_evolution.items()):
    #     axs[i].plot(values)
    xs = [1 * i for i in range(len(error_evolution["norm_error_u"]))]
    plt.plot(xs, error_evolution["norm_error_u"], color='tab:blue')
    plt.scatter(407, error_evolution['norm_error_u'][407], marker='^', edgecolor='tab:blue', color='lightblue')
    plt.xlabel("Epoch number")
    plt.ylabel("Error")
    plt.grid(True, linestyle='dashed', linewidth=0.5)
    save('error_evolution_u.png')

    plt.clf()
    xs = [1 * i for i in range(len(error_evolution["norm_error_p"]))]
    plt.plot(xs, error_evolution["norm_error_p"], color='tab:blue')
    plt.scatter(274, error_evolution['norm_error_p'][274], marker='^', edgecolor='tab:blue', color='lightblue')
    plt.xlabel("Epoch number")
    plt.ylabel("Error")
    plt.grid(True, linestyle='dashed', linewidth=0.5)
    save('error_evolution_p.png')


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


def solvers_benchmark():
    # Provided data
    data = [
        ("preonly", "lu", 1),
        ("cg", "jacobi", -1),
        ("cg", "lu", 1),
        ("cg", "gamg", -1),
        ("cg", "none", -1),
        ("gmres", "none", -1),
        ("gmres", "jacobi", -1),
        ("gmres", "lu", 1),
        ("gmres", "gamg", -1),
        ("bcgs", "none", -1),
        ("bcgs", "jacobi", -1),
        ("bcgs", "lu", 1),
        ("bcgs", "gamg", 0),
        ("richardson", "sor", -1),
        ("chebyshev", "jacobi", -1),
        ("preonly", "jacobi", -1),
        ("preonly", "bjacobi", -1),
        ("preonly", "sor", -1),
        ("preonly", "cholesky", -1)
    ]

    # Updated annotations to replace 1 labels
    annotations = [
        ("preonly", "lu", 11.38),
        ("cg", "lu", 10.88),
        ("gmres", "lu", 11.17),
        ("bcgs", "lu", 11.82),
        ("bcgs", "gamg", 357.44)
    ]

    # Extract unique methods and preconditioners
    methods = sorted(set(item[0] for item in data))
    preconditioners = sorted(set(item[1] for item in data))

    # Create a 2D array for the heatmap data
    heatmap_array = np.full((len(methods), len(preconditioners)), np.nan)

    # Fill the array with values from the provided data
    for method, preconditioner, value in data:
        method_idx = methods.index(method)
        preconditioner_idx = preconditioners.index(preconditioner)
        heatmap_array[method_idx, preconditioner_idx] = value

    # Create a dictionary for quick lookup of annotations
    annotation_dict = {(method, preconditioner): annotation for method, preconditioner, annotation in annotations}

    # Create an annotation array for the heatmap
    annot_array = np.full((len(methods), len(preconditioners)), "", dtype=object)

    # Fill the annotation array with the provided annotations and leave the rest empty
    for method, preconditioner, value in data:
        method_idx = methods.index(method)
        preconditioner_idx = preconditioners.index(preconditioner)
        if (method, preconditioner) in annotation_dict:
            annot_array[method_idx, preconditioner_idx] = annotation_dict[(method, preconditioner)]
        else:
            annot_array[method_idx, preconditioner_idx] = "" if value != 1 else value

    # Define a custom color map with the specified colors
    # cmap = sns.color_palette(["#ee3e31", "#f78837", "#1a8a5a"])
    # cmap = sns.color_palette(["crimson", "orange", "seagreen"])
    cmap = sns.color_palette(["crimson", "#f78837", "#1a8a5a"])

    # Create the heatmap with the specified custom colors, no title, and updated y-axis label
    # plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_array, annot=annot_array, fmt="", cmap=cmap, cbar=False, xticklabels=preconditioners, yticklabels=methods, linewidths=4, linecolor='white')

    # Set labels
    plt.xlabel("Preconditioner")
    plt.ylabel("Krylov subspace method")

    # Rotate the y-axis labels to be horizontal
    plt.yticks(rotation=0)

    save("solvers.png")


def compare_one_v_two():
    data_u = {
300 : {"Single network 37x37" : [0.049174586547913186, 0.06242713469664434, 0.03801643575591195,0.03770449509355723,0.049753405195009044],
        "Two networks 30x30" :
        [0.04321143426150778,
        0.040809111588790116,
        0.04595498537401389,
        0.037267601220187886,
        0.03813266878012012],
        "Single network 60x60" :
        [0.03853068841998876,
        0.053511388521787676,
        0.04224454569753502,
        0.03948659330915595,
        0.04071026589736833]
},

280 : {"Single network 37x37" : 
       [0.04517730565825059,
        0.04593289437152556,
        0.04091168560579145,
        0.06403218318140001,
        0.04435590453460449],
        "Two networks 30x30" :
        [0.04260334148695309,
        0.04386181382936046,
        0.05218434361556656,
        0.051815906145015425,
        0.03517249753541569],
        "Single network 60x60" :
        [0.040009841073843214,
        0.03910720604843337,
        0.04700260256379528,
        0.05012649443324376,
        0.04643129523763315]
},

260 : {"Single network 37x37" : 
       [0.037688131016593296,
        0.044299086745831835,
        0.04345554961966157,
        0.05211305718553723,
        0.039653233231875626],
        "Two networks 30x30" :
        [0.05861853422477406,
        0.04366820683095131,
        0.0468408171681322,
        0.05692004679973833,
        0.05243671913836994],
        "Single network 60x60" :
        [0.03729610293358162,
        0.05780176230625158,
        0.047973912054464406,
        0.04292103162610167,
        0.08129571071917384]
},
240 : {"Single network 37x37" : 
       [0.056606606915895925,
        0.0510768784634713,
        0.08308295893237144,
        0.04994554652245489,
        0.03705169262146919],
        "Two networks 30x30" :
        [0.052334602536455756,
        0.04239618709431495,
        0.057785886917412856,
        0.0540712587245369,
        0.04596089998623446]
},

220 : {"Single network 37x37" : 
       [0.04858443310587238,
        0.08534868153675587,
        0.056909978205144145,
        0.05041531487712147,
        0.08601142479923986],
        "Two networks 30x30":
        [0.05642993398284393,
        0.08649176650899036,
        0.04956519389840059,
        0.05411744399004619,
        0.05857779491353973]
},

200 : {"Single network 37x37" : 
       [0.08440707868427058,
        0.05006811375337682,
        0.05909046406218091,
        0.054298832202442505,
        0.1003921337285539],
        "Two networks 30x30":
        [0.10575072822343208,
        0.06594590139105928,
        0.09703730548929262,
        0.06021775941146988,
        0.04264795516111814]
},

180 : {"Single network 37x37" : 
       [0.08135751090228487,
        0.09360393897549454,
        0.055972906312422306,
        0.06442755840439868,
        0.10205174709659479],
        "Two networks 30x30":
        [0.0631144143485072,
        0.10017558290835159,
        0.06643428528253431,
        0.0816209057215277,
        0.0645475934513584]
}, 
160 : {"Single network 37x37" : 
       [0.09662243187670488,
        0.10898767090241311,
        0.07041668155329306,
        0.09490069382999121,
        0.10447365854286553],
        "Two networks 30x30": 
        [0.11448989061740401,
        0.06755734630880415,
        0.08001140280825839,
        0.10964692905544382,
        0.08239723999194855]
},
140 : {"Single network 37x37" : 
       [0.11195978402217682,
        0.12180846880195845,
        0.11333591651513049,
        0.10440263872985692,
        0.1020764315155116],
        "Two networks 30x30":
        [0.07810821339029751,
        0.10922031396269316,
        0.11025841445677485,
        0.10731334968597152,
        0.10391760276118377]
},

120 : {"Single network 37x37" : 
       [0.11943872566829968,
        0.14579013553872336,
        0.1362273990331813,
        0.11902728124702773,
        0.11827988105180269],
        "Two networks 30x30":
        [0.1461999885561204,
        0.11329753224756971,
        0.10238041724247353,
        0.11675051701062225,
        0.10384239424581404]
},

100 : {"Single network 37x37" : 
       [0.14477994830376614,
        0.14024635504890978,
        0.12919677870847812,
        0.1791479434618045,
        0.13259202384808264],
        "Two networks 30x30":
        [0.1360214347642792,
        0.106020030064668,
        0.11633513807884492,
        0.19427847837692688,
        0.12283307135916158]
},

80 : {"Single network 37x37" : 
      [0.1181974657792918,
        0.2146879330134446,
        0.1926131551486101,
        0.20000892914184304,
        0.13082214224139618],
        "Two networks 30x30" : 
        [0.19069333123289134,
        0.1654740548842043,
        0.1202232162683069,
        0.12469796591954616,
        0.14950775638026295]
},

60 : {"Single network 37x37" : 
      [0.18658136963303487,
        0.24018274539376835,
        0.20541233450301152,
        0.18987434265665423,
        0.17103926446201634],
        "Two networks 30x30" :
        [0.22876807307141633,
        0.20835804369698646,
        0.19355565233405822,
        0.2072764916183718,
        0.13320701845408714]
},

40 : {"Single network 37x37" : 
      [0.4142791110228636,
        0.375889520184361,
        0.2382590920562418,
        0.4086753718937413,
        0.245628823194641],
        "Two networks 30x30":
        [0.26234782377688276,
        0.22861328818480656,
        0.4745917502276433,
        0.22404222066613066,
        0.45534657071927115]
},

20 : {"Single network 37x37" : 
      [0.4367597137779795,
        0.4528267048147725,
        0.23646079564165706,
        0.4224484453882248,
        0.41304996555929446],
        "Two networks 30x30" :
        [0.43380572527820477,
        0.5047469693224543,
        0.47248172438418046,
        0.4482180487386531,
        0.41451854687119255]
}
}

    data_p = {
    300 : { "Single network 37x37" : [0.40096868792561613,
0.6870859651561034,
0.27791970569589886,
0.3593867689997691,
0.46007452497637424],
"Two networks 30x30": [0.6471573706628737,
0.7024693798470578,
0.45237361639183216,
0.5196390468268199,
0.38722897455676536],
"Single network 60x60" :
[0.36671903757740026,
0.46202790464386645,
0.28370508879948403,
0.40125377850859906,
0.4360214326728866]
},


    280 : { "Single network 37x37" : [0.5502410532024107,
0.42742622827706855,
0.3013404158000547,
0.7251731705495457,
0.3311504630400145] , 
"Two networks 30x30": [0.34636777243368255,
0.286254723253532,
0.2673152293020964,
0.653527004398913,
0.6789894655097114],
"Single network 60x60" :
[0.43512139009265466,
0.3332320927844268,
0.5082389898378082,
0.5065159690155033,
0.40579721181580763]
}
,
    260 : { "Single network 37x37" : [0.45803357191283134,
0.3179809026970364,
0.325330746818781,
0.5680292159686399,
0.4557648232544485] ,
"Two networks 30x30": [0.39531145967729026,
0.7590993346945623,
0.5777678399977344,
0.6450834495358402,
0.5162580413768313],
"Single network 60x60" :
[0.30147911683254747,
0.5876859805492052,
0.5421524909014144,
0.30738771038002455,
0.7945566495473968]
}
,
240 : { "Single network 37x37" : [0.4594523039206922,
0.41659329616866514,
0.719290989045649,
0.5085790906740626,
0.31569188268037207],
"Two networks 30x30": [0.6860583457927257,
0.410567993397952,
0.36265030111119845,
0.34732273753839504,
0.7290169004585801] }
,
    220 : { "Single network 37x37" : [0.41683668981056243,
0.697640766001546,
0.5910534131234246,
0.36831675561687205,
0.6628191776971998] ,
"Two networks 30x30": [0.6718138828186336,
0.7274788764089569,
0.4534842900078058,
0.8545999680528941,
0.9343402324096169]}
,
    200 : { "Single network 37x37" : [0.8866275681056782,
0.4140073916664623,
0.45080232357181393,
0.5447760129009617,
0.7571761873181965],
"Two networks 30x30": [0.7345671129367745,
0.6240636000545984,
0.5938064866106498,
0.45017615285132145,
0.8175015992062901] }
,
    180 : { "Single network 37x37" : [0.963529047491965,
0.6265761011061538,
0.6035055317370362,
0.6630289001608529,
0.9744282245925466],
"Two networks 30x30": [0.8482464282660097,
1.0087003056034838,
0.8002942411960379,
0.9286145659757038,
0.34238755450175806] }
,
    160 : { "Single network 37x37" : [0.8175355418479576,
0.7844102319232825,
0.49091304530611296,
0.8341899093289648,
0.9266215344612854],
"Two networks 30x30": [0.8068353737106398,
0.9163971666611664,
0.4288646865170724,
0.9542455231094863,
0.692788206667421] }
,
    140 : { "Single network 37x37" : [0.853395178589635,
0.8780654538765127,
0.8036080051281224,
0.7555367634501328,
0.8657221047632727],
"Two networks 30x30": [0.8428129374359792,
0.8523996873104516,
0.714200106610292,
0.7217404918342181,
1.040830326213599] }
,
    120 : { "Single network 37x37" : [0.7699098333774015,
1.3569847072259262,
0.8674650032296056,
0.8959983686897407,
1.2035752957165298],
"Two networks 30x30": [0.9234733679319534,
1.0735595940929785,
0.7896701895184137,
0.7649703820243803,
0.6598581488748873] }
,
    100 : { "Single network 37x37" : [1.0842391291071454,
1.0195469756826545,
0.8760332727161397,
0.826700771994148,
0.8633598355994179],
"Two networks 30x30": [0.9059647122105751,
0.6154400619641852,
0.7832494626069049,
0.9858502712959057,
0.9247164557043062] }
,

80 : { "Single network 37x37" : [0.8672393932098201,
1.076934071299067,
0.978281514516474,
0.9978625292515573,
1.629525111496979] ,
"Two networks 30x30": [0.8861905298561945,
0.8916924396000615,
0.7882006595881969,
0.5783406820720964,
1.1492725657170584]}

, 60 : { "Single network 37x37" : [0.9017808496793241,
0.994991046861736,
0.8223766296597345,
0.6637855873592381,
0.8867179003311788],
"Two networks 30x30": [1.8151652163247065,
0.808590555479215,
0.5980387356030858,
0.9960444430518878,
1.0389594901747208] }
,
40 : { "Single network 37x37" : [1.5229639929507248,
1.4090859271971843,
1.1130724421689857,
2.3890125470198464,
1.1203229877479313],
"Two networks 30x30": [0.9884802532020592,
5.273414788423884,
1.1710059819970182,
1.0495984318826241,
1.5943843226357783] }
,
20 : { "Single network 37x37" : [1.484028144449216,
2.768852584915636,
1.6407552507873004,
3.0180176036763733,
1.9008845110754464],
"Two networks 30x30": [2.0765658424714557,
2.057096240736191,
2.128075470087268,
3.2470353133653216,
2.0614499328897797] }
}

    def _compare(data_, title):
        # Sort the dataset sizes
        dataset_sizes = sorted(data_.keys(), reverse=True)

        # Prepare data structures
        processes = set(proc for measurements in data_.values() for proc in measurements)
        means = {proc: [] for proc in processes}
        stds = {proc: [] for proc in processes}

        # Populate mean and std lists
        for N in dataset_sizes:
            for process in processes:
                data = data_[N].get(process, [])
                if data:
                    mean = np.mean(data)
                    std_dev = np.std(data)
                else:
                    mean = None
                    std_dev = None
                means[process].append(mean)
                stds[process].append(std_dev)

        # Define colors and markers for potentially many processes
        # colors = ['tab:blue', 'tab:orange', 'tab:green', 'darkred', 'purple']
        # ecolors = ['lightblue', 'navajowhite', 'limegreen', 'red', 'violet']
        colors = ['tab:green', 'tab:blue', 'tab:orange',  'darkred', 'purple']
        ecolors = ['limegreen', 'lightblue', 'navajowhite', 'red', 'violet']
        markers = ['^', 's', 'o', 'x', 'd']
        process_details = {proc: {'color': colors[i % len(colors)], 'ecolor': ecolors[i % len(ecolors)], 'marker': markers[i % len(markers)], 'label': proc} for i, proc in enumerate(processes)}

        # Create the plot
        fig, ax = plt.subplots() # plt.subplots(figsize=(10, 5))

        # Plotting details
        for process in processes:
            valid_indices = [i for i, m in enumerate(means[process]) if m is not None]
            valid_sizes = [dataset_sizes[i] for i in valid_indices]
            valid_means = [means[process][i] for i in valid_indices]
            valid_stds = [stds[process][i] for i in valid_indices]
            
            if valid_means:
                ax.plot(valid_sizes, valid_means, "-", color=process_details[process]['color'], label=process_details[process]["label"])
                ax.errorbar(valid_sizes, valid_means, yerr=valid_stds, fmt='none', ecolor=process_details[process]['ecolor'], capsize=5, zorder=2)
                ax.scatter(valid_sizes, valid_means, color=process_details[process]['ecolor'], edgecolor=process_details[process]['color'], marker=process_details[process]['marker'], zorder=3)

        # Styling
        ax.set_xlabel('Dataset Size')
        ax.set_ylabel('Error')
        # ax.set_title('Comparison of Means Across Different Dataset Sizes')
        ax.set_yscale('log')  # Logarithmic scale
        ax.grid(True, linestyle='dashed')
        ax.grid(True, which='minor', linestyle='dashed', axis='y')
        ax.legend()

        # Show plot
        save(title)

    _compare(data_u, 'one_v_two_u.png')
    _compare(data_p, 'one_v_two_p.png')


def compare_reuse_new():
    new_data_u = {
    20: {"No reusing": [0.19754373291438598, 0.44805356580675054, 0.29333074415232346, 0.15417650899176, 0.197185962558339]},
    40: {"No reusing": [0.12243941182762606, 0.11067983619859266, 0.11870394611379022, 0.12257560488465831, 0.16419074622926114, 0.18254878705479166]},
    60: {"No reusing": [0.109179142905709, 0.08237853727459142, 0.0976228138672222, 0.08410655191256161, 0.11872100603076301]},
    80: {"No reusing": [0.07792825345371795, 0.10796491110003092, 0.11091143548915673, 0.0616593949053978, 0.09256260696485974]},
    100: {"No reusing": [0.09769041771465999, 0.054157506377368506, 0.07493124243952841, 0.1069131800504285, 0.06503892519780498]},
    120: {"No reusing": [0.0581932034941917, 0.06774063889321377, 0.05932645000918143, 0.07616532799676161, 0.05503220322923092]},
    140: {"No reusing": [0.05333694425830754, 0.06630722918942633, 0.04771778964201917, 0.04411007948727043, 0.062210665005547616]},
    160: {"No reusing": [0.044780842744706104, 0.0679382247985487, 0.0633816878647478, 0.050967132539461515, 0.045192277182396975]},
    180: {"No reusing": [0.04719843806687794, 0.05340826777236979, 0.05103585207625167, 0.034890110042030374, 0.04039761503693778]},
    200: {"No reusing": [0.04617832453334244, 0.04457919053306008, 0.04249483869587002, 0.03562924427696575, 0.03857839780197339]}
}
    
    new_data_p = {
    20: {"No reusing": [1.647611661227859, 0.9945798476093074, 1.2714858481407927, 1.1512668639351726, 1.6685800844514973]},
    40: {"No reusing": [1.025885959892209, 0.6310065024294571, 0.6534416740117897, 0.8770832831110361, 0.7870994876194728, 1.2527377794207553]},
    60: {"No reusing": [0.7598651909404884, 0.4996568169528686, 1.1414138577190773, 0.36430693742374687, 0.7112857960061582]},
    80: {"No reusing": [0.8424738300754997, 0.750833007751023, 0.28629290032088034, 1.3291911347492247, 0.8050803190574856]},
    100: {"No reusing": [0.5784278444386173, 0.4803331459305927, 0.703021397165057, 0.27179850010905066, 0.23945463090849842]},
    120: {"No reusing": [0.8370805034062316, 0.6865690659597019, 0.2508292685605493, 0.23743059398645522, 0.9222451462996191]},
    140: {"No reusing": [0.14227174364719422, 0.8368857050990829, 0.2814253297523761, 0.49999677973318424, 0.6298991949236409]},
    160: {"No reusing": [0.18952203986133412, 0.27348878890304285, 0.5726932397666415, 0.23087660769259177, 0.7030203614640707]},
    180: {"No reusing": [0.2716562761169626, 0.18739687748809872, 0.8847593245827476, 0.36852369059556256, 0.5444788589516347]},
    200: {"No reusing": [0.21860564469535812, 0.19509505873762525, 0.9546796554304272, 0.19014452798439746, 0.22346447839694775]}
}
    
    data_u = {
300 : {"Single network 37x37" : [0.049174586547913186, 0.06242713469664434, 0.03801643575591195,0.03770449509355723,0.049753405195009044],
        "Reusing" :
        [0.04321143426150778,
        0.040809111588790116,
        0.04595498537401389,
        0.037267601220187886,
        0.03813266878012012],
        "Single network 60x60" :
        [0.03853068841998876,
        0.053511388521787676,
        0.04224454569753502,
        0.03948659330915595,
        0.04071026589736833]
},

280 : {"Single network 37x37" : 
       [0.04517730565825059,
        0.04593289437152556,
        0.04091168560579145,
        0.06403218318140001,
        0.04435590453460449],
        "Reusing" :
        [0.04260334148695309,
        0.04386181382936046,
        0.05218434361556656,
        0.051815906145015425,
        0.03517249753541569],
        "Single network 60x60" :
        [0.040009841073843214,
        0.03910720604843337,
        0.04700260256379528,
        0.05012649443324376,
        0.04643129523763315]
},

260 : {"Single network 37x37" : 
       [0.037688131016593296,
        0.044299086745831835,
        0.04345554961966157,
        0.05211305718553723,
        0.039653233231875626],
        "Reusing" :
        [0.05861853422477406,
        0.04366820683095131,
        0.0468408171681322,
        0.05692004679973833,
        0.05243671913836994],
        "Single network 60x60" :
        [0.03729610293358162,
        0.05780176230625158,
        0.047973912054464406,
        0.04292103162610167,
        0.08129571071917384]
},
240 : {"Single network 37x37" : 
       [0.056606606915895925,
        0.0510768784634713,
        0.08308295893237144,
        0.04994554652245489,
        0.03705169262146919],
        "Reusing" :
        [0.052334602536455756,
        0.04239618709431495,
        0.057785886917412856,
        0.0540712587245369,
        0.04596089998623446]
},

220 : {"Single network 37x37" : 
       [0.04858443310587238,
        0.08534868153675587,
        0.056909978205144145,
        0.05041531487712147,
        0.08601142479923986],
        "Reusing":
        [0.05642993398284393,
        0.08649176650899036,
        0.04956519389840059,
        0.05411744399004619,
        0.05857779491353973]
},

200 : {"Single network 37x37" : 
       [0.08440707868427058,
        0.05006811375337682,
        0.05909046406218091,
        0.054298832202442505,
        0.1003921337285539],
        "Reusing":
        [0.10575072822343208,
        0.06594590139105928,
        0.09703730548929262,
        0.06021775941146988,
        0.04264795516111814]
},

180 : {"Single network 37x37" : 
       [0.08135751090228487,
        0.09360393897549454,
        0.055972906312422306,
        0.06442755840439868,
        0.10205174709659479],
        "Reusing":
        [0.0631144143485072,
        0.10017558290835159,
        0.06643428528253431,
        0.0816209057215277,
        0.0645475934513584]
}, 
160 : {"Single network 37x37" : 
       [0.09662243187670488,
        0.10898767090241311,
        0.07041668155329306,
        0.09490069382999121,
        0.10447365854286553],
        "Reusing": 
        [0.11448989061740401,
        0.06755734630880415,
        0.08001140280825839,
        0.10964692905544382,
        0.08239723999194855]
},
140 : {"Single network 37x37" : 
       [0.11195978402217682,
        0.12180846880195845,
        0.11333591651513049,
        0.10440263872985692,
        0.1020764315155116],
        "Reusing":
        [0.07810821339029751,
        0.10922031396269316,
        0.11025841445677485,
        0.10731334968597152,
        0.10391760276118377]
},

120 : {"Single network 37x37" : 
       [0.11943872566829968,
        0.14579013553872336,
        0.1362273990331813,
        0.11902728124702773,
        0.11827988105180269],
        "Reusing":
        [0.1461999885561204,
        0.11329753224756971,
        0.10238041724247353,
        0.11675051701062225,
        0.10384239424581404]
},

100 : {"Single network 37x37" : 
       [0.14477994830376614,
        0.14024635504890978,
        0.12919677870847812,
        0.1791479434618045,
        0.13259202384808264],
        "Reusing":
        [0.1360214347642792,
        0.106020030064668,
        0.11633513807884492,
        0.19427847837692688,
        0.12283307135916158]
},

80 : {"Single network 37x37" : 
      [0.1181974657792918,
        0.2146879330134446,
        0.1926131551486101,
        0.20000892914184304,
        0.13082214224139618],
        "Reusing" : 
        [0.19069333123289134,
        0.1654740548842043,
        0.1202232162683069,
        0.12469796591954616,
        0.14950775638026295]
},

60 : {"Single network 37x37" : 
      [0.18658136963303487,
        0.24018274539376835,
        0.20541233450301152,
        0.18987434265665423,
        0.17103926446201634],
        "Reusing" :
        [0.22876807307141633,
        0.20835804369698646,
        0.19355565233405822,
        0.2072764916183718,
        0.13320701845408714]
},

40 : {"Single network 37x37" : 
      [0.4142791110228636,
        0.375889520184361,
        0.2382590920562418,
        0.4086753718937413,
        0.245628823194641],
        "Reusing":
        [0.26234782377688276,
        0.22861328818480656,
        0.4745917502276433,
        0.22404222066613066,
        0.45534657071927115]
},

20 : {"Single network 37x37" : 
      [0.4367597137779795,
        0.4528267048147725,
        0.23646079564165706,
        0.4224484453882248,
        0.41304996555929446],
        "Reusing" :
        [0.43380572527820477,
        0.5047469693224543,
        0.47248172438418046,
        0.4482180487386531,
        0.41451854687119255]
}
}

    data_p = {
    300 : { "Single network 37x37" : [0.40096868792561613,
0.6870859651561034,
0.27791970569589886,
0.3593867689997691,
0.46007452497637424],
"Reusing": [0.6471573706628737,
0.7024693798470578,
0.45237361639183216,
0.5196390468268199,
0.38722897455676536],
"Single network 60x60" :
[0.36671903757740026,
0.46202790464386645,
0.28370508879948403,
0.40125377850859906,
0.4360214326728866]
},


    280 : { "Single network 37x37" : [0.5502410532024107,
0.42742622827706855,
0.3013404158000547,
0.7251731705495457,
0.3311504630400145] , 
"Reusing": [0.34636777243368255,
0.286254723253532,
0.2673152293020964,
0.653527004398913,
0.6789894655097114],
"Single network 60x60" :
[0.43512139009265466,
0.3332320927844268,
0.5082389898378082,
0.5065159690155033,
0.40579721181580763]
}
,
    260 : { "Single network 37x37" : [0.45803357191283134,
0.3179809026970364,
0.325330746818781,
0.5680292159686399,
0.4557648232544485] ,
"Reusing": [0.39531145967729026,
0.7590993346945623,
0.5777678399977344,
0.6450834495358402,
0.5162580413768313],
"Single network 60x60" :
[0.30147911683254747,
0.5876859805492052,
0.5421524909014144,
0.30738771038002455,
0.7945566495473968]
}
,
240 : { "Single network 37x37" : [0.4594523039206922,
0.41659329616866514,
0.719290989045649,
0.5085790906740626,
0.31569188268037207],
"Reusing": [0.6860583457927257,
0.410567993397952,
0.36265030111119845,
0.34732273753839504,
0.7290169004585801] }
,
    220 : { "Single network 37x37" : [0.41683668981056243,
0.697640766001546,
0.5910534131234246,
0.36831675561687205,
0.6628191776971998] ,
"Reusing": [0.6718138828186336,
0.7274788764089569,
0.4534842900078058,
0.8545999680528941,
0.9343402324096169]}
,
    200 : { "Single network 37x37" : [0.8866275681056782,
0.4140073916664623,
0.45080232357181393,
0.5447760129009617,
0.7571761873181965],
"Reusing": [0.7345671129367745,
0.6240636000545984,
0.5938064866106498,
0.45017615285132145,
0.8175015992062901] }
,
    180 : { "Single network 37x37" : [0.963529047491965,
0.6265761011061538,
0.6035055317370362,
0.6630289001608529,
0.9744282245925466],
"Reusing": [0.8482464282660097,
1.0087003056034838,
0.8002942411960379,
0.9286145659757038,
0.34238755450175806] }
,
    160 : { "Single network 37x37" : [0.8175355418479576,
0.7844102319232825,
0.49091304530611296,
0.8341899093289648,
0.9266215344612854],
"Reusing": [0.8068353737106398,
0.9163971666611664,
0.4288646865170724,
0.9542455231094863,
0.692788206667421] }
,
    140 : { "Single network 37x37" : [0.853395178589635,
0.8780654538765127,
0.8036080051281224,
0.7555367634501328,
0.8657221047632727],
"Reusing": [0.8428129374359792,
0.8523996873104516,
0.714200106610292,
0.7217404918342181,
1.040830326213599] }
,
    120 : { "Single network 37x37" : [0.7699098333774015,
1.3569847072259262,
0.8674650032296056,
0.8959983686897407,
1.2035752957165298],
"Reusing": [0.9234733679319534,
1.0735595940929785,
0.7896701895184137,
0.7649703820243803,
0.6598581488748873] }
,
    100 : { "Single network 37x37" : [1.0842391291071454,
1.0195469756826545,
0.8760332727161397,
0.826700771994148,
0.8633598355994179],
"Reusing": [0.9059647122105751,
0.6154400619641852,
0.7832494626069049,
0.9858502712959057,
0.9247164557043062] }
,

80 : { "Single network 37x37" : [0.8672393932098201,
1.076934071299067,
0.978281514516474,
0.9978625292515573,
1.629525111496979] ,
"Reusing": [0.8861905298561945,
0.8916924396000615,
0.7882006595881969,
0.5783406820720964,
1.1492725657170584]}

, 60 : { "Single network 37x37" : [0.9017808496793241,
0.994991046861736,
0.8223766296597345,
0.6637855873592381,
0.8867179003311788],
"Reusing": [1.8151652163247065,
0.808590555479215,
0.5980387356030858,
0.9960444430518878,
1.0389594901747208] }
,
40 : { "Single network 37x37" : [1.5229639929507248,
1.4090859271971843,
1.1130724421689857,
2.3890125470198464,
1.1203229877479313],
"Reusing": [0.9884802532020592,
5.273414788423884,
1.1710059819970182,
1.0495984318826241,
1.5943843226357783] }
,
20 : { "Single network 37x37" : [1.484028144449216,
2.768852584915636,
1.6407552507873004,
3.0180176036763733,
1.9008845110754464],
"Reusing": [2.0765658424714557,
2.057096240736191,
2.128075470087268,
3.2470353133653216,
2.0614499328897797] }
}

    def _compare(data_, title):
        # Sort the dataset sizes
        dataset_sizes = sorted(data_.keys(), reverse=True)

        # Prepare data structures
        processes = set(proc for measurements in data_.values() for proc in measurements)
        means = {proc: [] for proc in processes}
        stds = {proc: [] for proc in processes}

        # Populate mean and std lists
        for N in dataset_sizes:
            for process in processes:
                data = data_[N].get(process, [])
                if data:
                    mean = np.mean(data)
                    std_dev = np.std(data)
                else:
                    mean = None
                    std_dev = None
                means[process].append(mean)
                stds[process].append(std_dev)

        # Define colors and markers for potentially many processes
        # colors = ['tab:blue', 'tab:orange', 'tab:green', 'darkred', 'purple']
        # ecolors = ['lightblue', 'navajowhite', 'limegreen', 'red', 'violet']
        colors = ['tab:blue', 'tab:orange', 'tab:green',  'darkred', 'purple']
        ecolors = ['lightblue', 'navajowhite','limegreen',  'red', 'violet']
        markers = ['^', 's', 'o', 'x', 'd']
        process_details = {proc: {'color': colors[i % len(colors)], 'ecolor': ecolors[i % len(ecolors)], 'marker': markers[i % len(markers)], 'label': proc} for i, proc in enumerate(processes)}

        # Create the plot
        fig, ax = plt.subplots() # plt.subplots(figsize=(10, 5))

        # Plotting details
        for process in processes:
            if process == "No reusing":
                c = 'tab:blue'
                e = 'lightblue'
                m = '^'
            elif process == "Reusing":
                c = 'tab:orange'
                e = 'navajowhite'
                m = '^'
            else:
                continue

            valid_indices = [i for i, m in enumerate(means[process]) if m is not None]
            valid_sizes = [dataset_sizes[i] for i in valid_indices]
            valid_means = [means[process][i] for i in valid_indices]
            valid_stds = [stds[process][i] for i in valid_indices]
            
            if valid_means:
                ax.plot(valid_sizes, valid_means, "-", color=c, label=process_details[process]["label"])
                ax.errorbar(valid_sizes, valid_means, yerr=valid_stds, fmt='none', ecolor=e, capsize=5, zorder=2)
                ax.scatter(valid_sizes, valid_means, color=e, edgecolor=c, marker=m, zorder=3)

        # Styling
        ax.set_xlabel('Dataset Size')
        ax.set_ylabel('Error')
        # ax.set_title('Comparison of Means Across Different Dataset Sizes')
        ax.set_yscale('log')  # Logarithmic scale
        ax.grid(True, linestyle='dashed')
        ax.grid(True, which='minor', linestyle='dashed', axis='y')
        ax.legend()

        # Show plot
        save(title)

    def merge_dicts(d1, d2):
        for key, value in d2.items():
            if key in d1:
                # Access sub-dictionary
                sub_dict = d1[key]
                for sub_key, sub_value in value.items():
                    if sub_key in sub_dict:
                        # Raise an error if the sub-key already exists
                        raise ValueError(f"Error: The sub-key '{sub_key}' already exists in the dictionary for key {key}.")
                    else:
                        sub_dict[sub_key] = sub_value
            else:
                d1[key] = value
        return d1
    
    merged_data_u = merge_dicts(data_u, new_data_u)
    merged_data_p = merge_dicts(data_p, new_data_p)

    _compare(merged_data_u, "reuse_u.png")
    _compare(merged_data_p, "reuse_p.png")


def plot_online_times():
    with open('results/online_time.npy', 'rb') as f:
        FEM_times = np.load(f)
        POD_ANN_times = np.load(f)

    plt.scatter(range(len(FEM_times)), FEM_times, color='lightblue', edgecolor='tab:blue', marker='^', label="FEM")
    plt.scatter(range(len(POD_ANN_times)), POD_ANN_times, color='navajowhite', edgecolor='tab:orange', marker='o', label="POD-ANN online")

    plt.xlabel("Sample")
    plt.ylabel("Time [ns]")

    plt.yscale("log")
    plt.grid(True, linestyle='dashed')
    plt.grid(True, which='minor', axis='y', linestyle='dashed', linewidth=0.5)  # Enhance visibility of log scale grid lines
    plt.legend()
    # plt.set_axisbelow(True)

    save('online_time.png')

if __name__ == "__main__":
    plot_online_times()