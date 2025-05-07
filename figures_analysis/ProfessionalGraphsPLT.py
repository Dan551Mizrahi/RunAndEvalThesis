import math
import random
from math import floor
import matplotlib.pyplot as plt
from random import sample
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from readJSONtoDict import read_json_to_dict, read_whole_folders_data, read_folder_of_folders_to_one_list
import argparse
from FilterData import *
from FindAggFunctionOnData import  calculate_correlation_covariance

def create_data_scatter_plot(X, Y, x_name, y_name, X_avg=None, Y_avg=None, split_name=False, split_values=None, title=None, saving_title=None, p=None, no_title=True):

    # Create the plot
    plt.figure(figsize=(8, 6))
    if no_title:
        pass
    elif title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f'{y_name} against {x_name}', fontsize=14)


    if split_name:
        for i, X_i in enumerate(X):
            Y_i = Y[i]
            split_value = split_values[i]
            # Plot the data points with a colored line and markers add label for legend
            plt.scatter(X_i, Y_i, marker='o', label=f'{split_name}={split_value}')
        # Add legend
        plt.legend()
    else:
        # Plot the data points with a black line and markers
        plt.scatter(X, Y, color='k', marker='o')

    if p:
        plt.plot(X, p(X), "r-")

    if X_avg:
        X_avg, Y_avg = zip(*sorted(zip(X_avg, Y_avg)))
        plt.plot(X_avg, Y_avg, color='red', marker='s', linestyle='--')

    # Set axis labels and font size
    plt.xlabel(x_name, fontsize=12)
    plt.ylabel(y_name, fontsize=12)

    # Set tick parameters for a classic look
    plt.tick_params(direction='in', length=6, width=1, top=True, right=True)

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add a grid for better readability
    plt.grid(linestyle='--', alpha=0.5)

    if saving_title:
        plt.savefig(saving_title)
    else:
        # Save the plot as a PDF for LaTeX inclusion
        plt.savefig(f'figure_{y_name}_{x_name}.pdf')

def create_data_plot(X, Y, x_name, y_name, X_avg=None, Y_avg=None, split_name=False, split_values=None, scatter = False, title=None, saving_title=None, p=None, no_title=True):

    if scatter:
        return create_data_scatter_plot(X, Y, x_name, y_name, split_name=split_name, split_values=split_values, title=title, saving_title=saving_title, p=p, X_avg=X_avg, Y_avg=Y_avg, no_title=no_title)

    # Create the plot
    plt.figure(figsize=(8, 6))
    if no_title:
        pass
    elif title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f'{y_name} against {x_name}', fontsize=14)

    if not split_name:
        # Sort the data points by X
        X, Y = zip(*sorted(zip(X, Y)))

        # Plot the data points with a black line and markers
        plt.plot(X, Y, 'k-', linewidth=2)

    else:

        for i, X_i in enumerate(X):
            Y_i = Y[i]
            split_value = split_values[i]
            # Sort the data points by X
            X_i, Y_i = zip(*sorted(zip(X_i, Y_i)))
            # Plot the data points with a colored line and markers add label for legend
            if X_avg:
                alp = 0.5
            else:
                alp = 1.0
            plt.plot(X_i, Y_i, marker='o', linewidth=2, label=f'{split_name}={split_value}', alpha=alp)
            # Add legend
        plt.legend()

    if X_avg:
        X_avg, Y_avg = zip(*sorted(zip(X_avg, Y_avg)))
        plt.plot(X_avg, Y_avg, color='red', marker='s', linestyle='--')

    # Set axis labels and font size
    plt.xlabel(x_name, fontsize=12)
    plt.ylabel(y_name, fontsize=12)

    # Set tick parameters for a classic look
    plt.tick_params(direction='in', length=6, width=1, top=True, right=True)

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add a grid for better readability
    plt.grid(linestyle='--', alpha=0.5)

    # Save the plot as a PDF for LaTeX inclusion
    plt.savefig(saving_title)

def process_and_plot_data_preprocessing(data):
    """
    Processes a list of dictionaries, filters them based on specified criteria,
    groups them by 'Treewidth', and plots the relationship between 'Treewidth'
    and 'Preprocess Runtime' with error bars.

    Args:
        data: A list of dictionaries.

    Returns:
        None. Plots a graph.
    """

    filtered_data = []
    for d in data:
        if "preprocess_runtime" in d:
            if 45 <= d.get("nodes + hyperedges", 0) <= 70:
                filtered_data.append(d)

    grouped_data = {}
    for d in filtered_data:
        treewidth = d.get("Tree Width")
        if treewidth is not None and treewidth > 0:
            if treewidth not in grouped_data:
                grouped_data[treewidth] = []
            if d["preprocess_runtime"] > 0:
                grouped_data[treewidth].append( (d["preprocess_runtime"], d["nodes + hyperedges"], d["Number of nodes"], d["Number of hyperedges"]))

    treewidths = []
    mean_runtimes = []
    std_runtimes = []
    nm_list = []

    X_reg = []
    Y_reg = []
    min_m = float('inf')
    max_m = float('-inf')
    min_n = float('inf')
    max_n = float('-inf')
    for treewidth, tuples in sorted(grouped_data.items()):
        print("tree width: ", treewidth, " ", len(tuples))
        sample1 = sample(tuples, 20)
        runtimes = [t[0] for t in sample1]
        nm = [t[1] for t in sample1]
        ms = [t[3] for t in sample1]
        ns = [t[2] for t in sample1]
        min_m = min(min_m, min(ms))
        max_m = max(max_m, max(ms))
        min_n = min(min_n, min(ns))
        max_n = max(max_n, max(ns))
        print(f"for treewidth {treewidth} number of runtimes:",len(runtimes), "Mean, std:", np.mean(runtimes), np.std(runtimes))
        if len(runtimes) >= 1: # Ensure we have enough data to calculate the mean and std
            treewidths.append(treewidth)
            mean_runtimes.append(np.mean(runtimes))
            std_runtimes.append(np.std(runtimes))
            nm_list.append(np.mean(nm))
            for nh, rt in zip(nm, runtimes):
                X_reg.append(nh * treewidth * (5 ** (2 * treewidth)))
                Y_reg.append(rt)



    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(treewidths, mean_runtimes, 'k-', marker='o', linewidth=2)

    Model = LinearRegression()

    Model.fit(np.array(X_reg).reshape(-1, 1), np.array(Y_reg).reshape(-1, 1))
    scale_factor = Model.coef_[0][0]

    X = np.linspace(2, 7, 100)
    regression_line_values = [nm_list[min(5, int(floor(x)))] * x * (5 ** (2 * x)) * scale_factor for x in X]

    plt.plot(X, regression_line_values, 'r-', label="Regression Line")

    # Set axis labels and font size
    plt.xlabel(r'Treewidth of $B(\mathcal{H})$', fontsize=12)
    plt.ylabel("Mean Preprocess Runtime (20 samples) [seconds]", fontsize=12)

    # Set tick parameters for a classic look
    plt.tick_params(direction='in', length=6, width=1, top=True, right=True)

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add a grid for better readability
    plt.grid(linestyle='--', alpha=0.5)
    plt.title("Relationship between Treewidth and Preprocess Runtime")

    # Add a text box with the equation of the red line
    equation = f"$y = {scale_factor:.2e} \cdot  (m + n) \cdot w \cdot (5^{{2w}})$"
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    # Add a text box with the range of 'nodes'
    range_text = f"Range of number of nodes: [{int(min_n)}, {int(max_n)}]"
    plt.text(0.05, 0.75, range_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    # Add a text box with the range of 'nodes + hyperedges'
    range_text = f"Range of number of hyperedges: [{int(min_m)}, {int(max_m)}]"
    plt.text(0.05, 0.65, range_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    range_text = r"$||\mathcal{H}|| = n + m$"
    plt.text(0.05, 0.85, range_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    plt.savefig(f'figure_preprocessing1.pdf')

    plt.cla()
    plt.clf()

    plt.figure(figsize=(8, 6))
    plt.scatter(treewidths, mean_runtimes, color="red", marker='x')

    plt.plot(X, regression_line_values, 'k-', label="Regression Line")

    # Set axis labels and font size
    plt.xlabel(r'Treewidth of $B(\mathcal{H})$', fontsize=12)
    plt.ylabel("Mean Preprocess Runtime (20 samples) [seconds]", fontsize=12)

    # Set tick parameters for a classic look
    plt.tick_params(direction='in', length=6, width=1, top=True, right=True)

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add a grid for better readability
    plt.grid(linestyle='--', alpha=0.5)
    plt.title("Relationship between Treewidth and Preprocess Runtime")

    plt.savefig(f'figure_preprocessing2.pdf')

    plt.cla()
    plt.clf()

    plt.figure(figsize=(8, 6))
    plt.plot(treewidths, mean_runtimes, 'k-', marker='o', linewidth=2)

    # Set axis labels and font size
    plt.xlabel(r'Treewidth of $B(\mathcal{H})$', fontsize=12)
    plt.ylabel("Mean Preprocess Runtime (20 samples) [seconds]", fontsize=12)

    # Set tick parameters for a classic look
    plt.tick_params(direction='in', length=6, width=1, top=True, right=True)

    # Add a text box with the range of 'nodes'
    range_text = f"Range of number of nodes: [{int(min_n)}, {int(max_n)}]"
    plt.text(0.05, 0.95, range_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    # Add a text box with the range of 'nodes + hyperedges'
    range_text = f"Range of number of hyperedges: [{int(min_m)}, {int(max_m)}]"
    plt.text(0.05, 0.85, range_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add a grid for better readability
    plt.grid(linestyle='--', alpha=0.5)
    plt.title("Relationship between Treewidth and Preprocess Runtime")

    plt.savefig(f'figure_preprocessing3.pdf')

    plt.cla()
    plt.clf()

    plt.figure(figsize=(8, 6))
    plt.errorbar(treewidths, mean_runtimes, yerr=std_runtimes, fmt='k-', capsize=5, label="Mean Preprocess Runtime")

    # Set axis labels and font size
    plt.xlabel(r'Treewidth of $B(\mathcal{H})$', fontsize=12)
    plt.ylabel("Mean Preprocess Runtime (20 samples) [seconds]", fontsize=12)

    # Set tick parameters for a classic look
    plt.tick_params(direction='in', length=6, width=1, top=True, right=True)

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add a grid for better readability
    plt.grid(linestyle='--', alpha=0.5)
    plt.title("Relationship between Treewidth and Preprocess Runtime")

    plt.savefig(f'figure_preprocessing4.pdf')

def process_and_plot_data_enum(data):
    filtered_data = []
    for d in data:
        filtered_data.append(d)

    grouped_data = {}
    for d in filtered_data:
        treewidth = d.get("Treewidth")
        grouped_data[treewidth] = []
        grouped_data[treewidth].append((d["Delays"], d["(m + n) * tw"]))

    i = 0
    while i<100000:
        X_avg = []
        Y_30 = []
        Y_100 = []
        for treewidth in sorted(grouped_data.keys()):
            t = random.choice(grouped_data[treewidth])
            y, x = t[0], t[1]
            X_avg.append(x)
            Y_30.append(y[:30])
            Y_100.append(y[:100])

        Y_avg_30 = [y[-1]/30 for y in Y_30]
        Y_avg_100 = [y[-1]/100 for y in Y_100]

        if Y_avg_30 == sorted(Y_avg_30):
            break

        i+=1
    if i == 100000:
        print("Could not find a good ordering")

    Y_30 = [[0]+y1 for y1 in Y_30]
    Y_100 = [[0]+y1 for y1 in Y_100]

    X_30 = [list(range(0, len(y)+1)) for y in Y_30]
    create_data_plot(X_30, Y_30, 'The i-th minimal hitting set', 'The cumulative delay (seconds)',
                     title=r'Cumulative delay of the enumeration of the first 30 MHS',
                     saving_title="figure_Enum30.pdf",
                     split_name = "Treewidth", split_values = sorted(list(grouped_data.keys())))

    #transform X_30 and Y_30 to the ranges of the axis for the average plot
    max_avg = max(X_avg)
    min_avg = min(X_avg)
    for X in X_30:
        max_x = max(X)
        min_x = min(X)
        for i in range(len(X)):
            X[i] = ((X[i] - min_x) / (max_x - min_x)) * (max_avg - min_avg) + min_avg
    max_Y_avg = max(Y_avg_30)
    min_Y_avg = min(Y_avg_30)
    for Y in Y_30:
        max_y = max(Y)
        min_y = min(Y)
        for i in range(len(Y)):
            Y[i] = ((Y[i] - min_y) / (max_y - min_y)) * (max_Y_avg - min_Y_avg) + min_Y_avg

    create_data_plot(X_30, Y_30,  r'$(m + n) \cdot TW$', 'The average delay (seconds)',
                     X_avg=X_avg, Y_avg=Y_avg_30,
                     title=r'Average delay of the enumeration of the first 30 MHS',
                     saving_title="figure_Enum30_avg.pdf",
                     split_name="Treewidth", split_values=sorted(list(grouped_data.keys())))
    X_100 = [list(range(0, len(y) + 1)) for y in Y_100]
    create_data_plot(X_100, Y_100, 'The i-th minimal hitting set', 'The cumulative delay (seconds)',
                     title=r'Cumulative delay of the enumeration of the first 100 MHS',
                     saving_title="figure_Enum100.pdf",
                     split_name="Treewidth", split_values=sorted(list(grouped_data.keys())))

    # transform X_30 and Y_30 to the ranges of the axis for the average plot
    max_avg = max(X_avg)
    min_avg = min(X_avg)
    for X in X_100:
        max_x = max(X)
        min_x = min(X)
        for i in range(len(X)):
            X[i] = ((X[i] - min_x) / (max_x - min_x)) * (max_avg - min_avg) + min_avg
    max_Y_avg = max(Y_avg_100)
    min_Y_avg = min(Y_avg_100)
    for Y in Y_100:
        max_y = max(Y)
        min_y = min(Y)
        for i in range(len(Y)):
            Y[i] = ((Y[i] - min_y) / (max_y - min_y)) * (max_Y_avg - min_Y_avg) + min_Y_avg

    create_data_plot(X_100, Y_100, r'$(m + n) \cdot TW$', 'The average delay (seconds)',
                     X_avg=X_avg, Y_avg=Y_avg_100,
                     title=r'Average delay of the enumeration of the first 100 MHS',
                     saving_title="figure_Enum100_avg.pdf",
                     split_name="Treewidth", split_values=sorted(list(grouped_data.keys())))

    create_data_plot(X_avg, Y_avg_100, r'$(m + n) \cdot TW$', 'The average delay (seconds)',
                        title=r'Average delay of the enumeration of the first 100 MHS',
                        saving_title="figure_Enum100_avg_solo.pdf")

def process_and_plot_data_enum_scatter(data):
    filtered_data = []
    for d in data:
        filtered_data.append(d)

    X = []
    Y = []
    treewidths = set()
    for dict1 in filtered_data:
        treewidths.add(dict1['Treewidth'])
        X.append(dict1["(m + n) * tw"])
        Y.append(dict1["Average delay"])

    print("Correlation: ", calculate_correlation_covariance(X, Y)[1], "Covariance: ", calculate_correlation_covariance(X, Y)[0])

    create_data_plot(X, Y, r'$(m + n) \cdot TW$', 'The average delay (seconds)',
                        title=r'Average delay of the enumeration',
                        saving_title="figure_Enum_scatter1.pdf",
                        scatter=True)
    # The same, but divided by treewidth
    X = [[] for _ in range(len(treewidths))]
    Y = [[] for _ in range(len(treewidths))]
    min_treewidth = min(treewidths)
    for dict1 in filtered_data:
        X[int(dict1["Treewidth"]) - min_treewidth].append(dict1["(m + n) * tw"])
        Y[int(dict1["Treewidth"]) - min_treewidth].append(dict1["Average delay"])

    create_data_plot(X, Y, r'$(m + n) \cdot TW$', 'The average delay (seconds)',
                     title=r'Average delay of the enumeration',
                     saving_title="figure_Enum_scatter2.pdf",
                     split_name="Treewidth", split_values=sorted(treewidths),
                     scatter=True)

    X = []
    Y = []
    for dict1 in filtered_data:
        X.append(dict1["(m + n) * tw"])
        Y.append(dict1["Average delay"])

    z = np.polyfit(X, Y, 1)
    p = np.poly1d(z)

    create_data_plot(X, Y, r'$(m + n) \cdot TW$', 'The average delay (seconds)',
                     title=r'Average delay of the enumeration',
                     saving_title="figure_Enum_scatter3.pdf",
                     scatter=True,
                     p=p)

def process_and_plot_data_preprocessing_scatter(data):
    filtered_data = []
    for d in data:
        filtered_data.append(d)

    X = []
    Y = []
    for dict1 in filtered_data:

        X.append(dict1["n + m"]*(dict1["Max Join Node Size"]))
        Y.append(dict1["Preprocessing Runtime"])

    print("Correlation: ", calculate_correlation_covariance(X, Y)[1], "Covariance: ", calculate_correlation_covariance(X, Y)[0])

    create_data_plot(X, Y, r'$(m + n) \cdot 5 ^ ETW$', 'Preprocessing Runtime (seconds)',
                        title=r'Average delay of the enumeration',
                        saving_title="figure_preprocess_important.pdf",
                        scatter=True)

def plot_enum_delays(data, split_property=None, cutoff=None):
    """
    Plots the delays of the enumeration for each of the graphs as plot, and color every graph's line by its split_property value.
    :param data: A list of dictionaries.
    :param split_property: The property to split the data by. If None, we consider only one plot for one graph.
    :param cutoff: The cutoff value for the delays (we take the first cutoff delays). If None, we consider all delays.
    """

    X = []
    Y = []
    split_name = split_property
    split_values = []
    for d in data:
        if cutoff:
            delays = d["Delays"][:cutoff]
        else:
            delays = d["Delays"]
        X.append(list(range(1, len(delays) + 1)))
        Y.append(delays)
        if split_property:
            split_values.append(d[split_property])
        else:
            break

    if split_property is None:
        X = X[0]
        Y = Y[0]

    if cutoff:
        title = f'Cumulative delay of the enumeration of the first {cutoff} MHS'
    else:
        title = f'Cumulative delay of the enumeration of MHS'

    create_data_plot(X, Y, "The i-th hitting set", 'Time (seconds)',
                     title=title,
                     saving_title=f"enumeration_cumulative_time_split_{split_property}_cutoff_{cutoff}.pdf",
                     split_name=split_property, split_values=split_values,)

def two_properties_graph(data, x_property, y_property, scatter=False, trend_line=False, log_scaled_x_axis=False):
    """
    Plots the relationship between two properties of the data.
    :param data: list of dictionaries.
    :param x_property: The property to plot on the x-axis.
    :param y_property: The property to plot on the y-axis.
    """
    X = []
    Y = []

    for d in data:
        if x_property in d and y_property in d:
            X.append(d[x_property])
            Y.append(d[y_property])
    if log_scaled_x_axis:
        X = [math.log(x) for x in X]
    if scatter and trend_line:
        Y_avg_lists = [[] for _ in set(X)]
        X_avg = sorted(set(X))
        for i, y in enumerate(Y):
            Y_avg_lists[X_avg.index(X[i])].append(y)
        Y_avg = [sum(l) / len(l) for l in Y_avg_lists]
    print("Correlation: ", calculate_correlation_covariance(X, Y)[1], "Covariance: ", calculate_correlation_covariance(X, Y)[0])
    if scatter and trend_line:
        create_data_plot(X, Y, x_property, y_property + " (seconds)",
                         X_avg=X_avg, Y_avg=Y_avg,
                         title=f'{y_property} against {x_property}',
                         saving_title=f'{y_property}_against_{x_property}_Trend.pdf',
                         scatter=scatter)
    else:
        create_data_plot(X, Y, x_property, y_property + " (seconds)",
                         title=f'{y_property} against {x_property}',
                         saving_title=f'{y_property}_against_{x_property}.pdf',
                         scatter=scatter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and plot data from JSON files.')
    parser.add_argument('--path_to_data', type=str, default="/path/to/defualt/data")
    args = parser.parse_args()
    path_to_data = args.path_to_data
    total_list = read_folder_of_folders_to_one_list(path_to_data)
    print("Number of items in data: ", len(total_list))

    # total_list = filter_by_tw(total_list, 11)
    # two_properties_graph(total_list, "n + m", "Preprocessing Runtime", scatter=True, trend_line=False)
    # process_and_plot_data_enum(total_list)
    plot_enum_delays(total_list)

    # two_properties_graph(total_list, "Special Join Measure","Preprocessing Runtime", scatter=True, trend_line=False)
    # total_list = filter_by_max_join(total_list, 17)
    # two_properties_graph(total_list, "Special Join Measure", "Preprocessing Runtime", scatter=True, log_scaled_x_axis=True)
    # process_and_plot_data_enum(total_list)
    # process_and_plot_data_enum_scatter(total_list)
    # two_properties_graph(total_list, "Number of Minimal Hitting Sets","Average delay", scatter=True, trend_line=False, )
    # two_properties_graph(total_list, "Number of Minimal Hitting Sets", "Preprocessing Runtime", scatter=True, trend_line=False)
    # process_and_plot_data_preprocessing_scatter(total_list)