import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import glob


def find_files(path): return glob.glob(path)


chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', \
         't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', "'", '.', ',', '\n']


def extract_statistics(choose_type, to_analyze_first, to_analyze_second, plot=False):
    num_of_data = len(find_files("list_data/" + choose_type + "/char/*.txt"))
    history_first, history_second = [], []

    for a in range(num_of_data):  # for each person
        print("\t Processing", (a + 1), "th data...")
        with open("list_data/" + choose_type + "/char/char" + str(a)+".txt", "rb") as fp:
            char = pickle.load(fp)

        with open("list_data/" + choose_type + "/x_data/x_data" + str(a)+".txt", "rb") as fp:
            x_data = pickle.load(fp)

        with open("list_data/" + choose_type + "/y_data/y_data" + str(a)+".txt", "rb") as fp:
            y_data = pickle.load(fp)

        x_dic, y_dic, x_count = {}, {}, {}
        space, dot = [], []

        for i in range(len(char)):  # for each sentence
            x_one_line, y_one_line = {}, {}

            for j, k in enumerate(char[i]):  # for one sentence
                x, y = x_data[i][j], y_data[i][j]
                if k in x_one_line.keys():
                    x_one_line[k].append(x)
                    y_one_line[k].append(y)
                else:
                    x_one_line[k], y_one_line[k] = [x], [y]

            if to_analyze_first in x_one_line.keys():
                # the base point to_analyze_first (' ')
                x0, y0 = np.mean(x_one_line[to_analyze_first]), np.mean(y_one_line[to_analyze_first])

                # to scale the points based on to_analyze_first (' ') and to_analyze_second ('p')
                if to_analyze_second in x_one_line.keys():
                    x1, y1 = np.mean(x_one_line[to_analyze_second]), np.mean(y_one_line[to_analyze_second])

                    space.append([x0, y0])
                    dot.append([x1, y1])

                    for l in x_one_line:
                        for list_x in x_one_line[l]:
                            if type(list_x) == int:
                                list_x = [list_x]
                            for x_original in list_x:
                                x = (x_original - x0)
                                if l in x_dic.keys():
                                    x_dic[l].append(x)
                                    x_count[l] = x_count[l] + 1
                                else:
                                    x_dic[l] = [x]
                                    x_count[l] = 0
                    for m in y_one_line:
                        for list_y in y_one_line[m]:
                            if type(list_y) == int:
                                list_y = [list_y]
                            for y_original in list_y:
                                y = (y0 - y_original)
                                if m in y_dic.keys():
                                    y_dic[m].append(y)
                                else:
                                    y_dic[m] = [y]
        history_first.append(space)
        history_second.append(dot)

        if plot:
            fig, ax = plt.subplots()
            # for all the points
            for c in x_dic.keys():
                plt.scatter(x_dic[c], y_dic[c], s= 2)
                pylab.ylim([-100, 550])

            for d in x_dic.keys():
                plt.scatter(np.mean(x_dic[d]), np.mean(y_dic[d]), s=10, c='black')

            for char in x_dic.keys():
                plt.text(np.mean(x_dic[char]) + 10, np.mean(y_dic[char]) + 10, char, fontsize=24)

            if not os.path.exists("figs"):
                os.makedirs("figs")
            fig.savefig('figs/figure_' + choose_type + '_' + str(a))

            plt.show()
            plt.close(fig)

    return history_first, history_second


if __name__ == "__main__":
    # Reading total data
    to_analyze_first, to_analyze_second = ' ', 'p'
    plot = True
    print("Processing palm detached...")
    o_space, o_char_p = extract_statistics("o", to_analyze_first, to_analyze_second, plot)
    print("Processing palm attached...")
    x_space, x_char_p = extract_statistics("x", to_analyze_first, to_analyze_second, plot)

    # preprosess data: convert to numpy array
    space = o_space + x_space
    dot = o_char_p + x_char_p
    history_len = [len(l) for l in space]
    num_to_iter = min(history_len)
    space = np.array([l[:num_to_iter] for l in space])
    dot = np.array([l[:num_to_iter] for l in dot])

    # extract statistics
    # 1) scale
    epsilon = 0.0001
    scale = np.abs(dot - space)
    scale_x, scale_y = scale[:, :, 0], scale[:, :, 1]
    scale_x /= (scale_x[:, 0].reshape((scale_x.shape[0], 1)) + epsilon)
    scale_y /= (scale_y[:, 0].reshape((scale_y.shape[0], 1)) + epsilon)

    # 2) offset
    offset = np.copy(space)
    offset_x, offset_y = offset[:, :, 0], offset[:, :, 1]
    offset_x -= offset_x[:, 0].reshape((offset_x.shape[0], 1))
    offset_y -= offset_y[:, 0].reshape((offset_y.shape[0], 1))

    # 3) size
    pixel_size = 0.2652
    scale_factor_x, scale_factor_y = (14.5 / 5), (5.0 / 3.0)
    size = np.abs(dot - space) * pixel_size
    size_x, size_y = size[:, :, 0] * scale_factor_x, size[:, :, 1] * scale_factor_y

    # plot the result
    print("Analysis Result:")

    if not os.path.exists("figs"):
        os.makedirs("figs")

    for var, name, color in zip([scale_x, scale_y, offset_x, offset_y, size_x, size_y],
                                ["Scale: Horizontal", "Scale: Vertical", "Offset: Horizontal", "Offset: Vertical", "size_x", "size_y"],
                                ["orange", "green", "crimson", "navy", "goldenrod", "indigo"]):
        x = np.arange(1, num_to_iter + 1, 1)
        mean, std = np.mean(var, axis=0), np.std(var, axis=0)
        print('[' + name + ']')
        print('\t mean:', np.mean(mean))
        print('\t std:', np.std(var))
        print('\t min:', np.min(mean))
        print('\t max:', np.max(mean))
        fig, ax = plt.subplots()
        plt.errorbar(x, mean, std, fmt='-o', color=color, ecolor='lightgray', elinewidth=3, capsize=0)
        # plt.xlabel("Time Step", fontsize=12)
        # plt.ylabel(name, fontsize=12)
        # plt.title("Variation of " + name + " over time")
        ratio = 0.3
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        # the abs method is used to make sure that all numbers are positive
        # because x and y axis of an axes maybe inverted.
        ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
        fig.savefig('figs/figure_' + name)
        plt.show()
        plt.close(fig)
