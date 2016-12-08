import matplotlib.pyplot as plt
import constants as c
import json
import numpy as np
import math


def plot_combined(filename, normalization=False):

    #Load appropriate data
    DIR_GIST = c.PATH_RES_GIST_TRAIN
    DIR_COH = c.PATH_RES_COH_TRAIN
    DIR_ORB = c.PATH_RES_ORB_TRAIN
    if normalization:
        filename += "_norm"
    filename += "_results_"
    ext = ".json"
    file_gist = json.load(open(DIR_GIST + filename + 'gist'+ext, "r"))
    file_coh = json.load(open(DIR_COH + filename + 'coh' + ext, "r"))
    file_orb = json.load(open(DIR_ORB + filename + 'orb' + ext, "r"))
    print file_gist
    print file_coh
    print file_orb

    # Plotting now
    fig, ax = plt.subplots()
    if normalization:
        label_chng = ' with normalization'
    else:
        label_chng = ' without normalization'

    x_loc = np.arange(len(file_gist))
    width = .25
    list_label = []
    mean_gist, mean_coh, mean_orb = [], [], []
    std_gist, std_coh, std_orb = [], [], []
    for name, value in file_gist.items():
        list_label.append(name)
        mean_gist.append(value[0])
        std_gist.append(value[1])
    for name, value in file_coh.items():
        mean_coh.append(value[0])
        std_coh.append(value[1])
    for name, value in file_orb.items():
        mean_orb.append(value[0])
        std_orb.append(value[1])

    rects1 = plt.bar(x_loc, mean_gist, width=width, color='r', yerr=std_gist)
    rects2 = plt.bar(x_loc + width, mean_coh, width=width, color='b', yerr=std_coh)
    rects3 = plt.bar(x_loc + 2*width, mean_orb, width=width, color='g', yerr=std_orb)
    plt.xticks(x_loc + (1.5 * width), list_label)
    plt.tick_params(axis='x')
    plt.xlabel('Classifier Type')
    plt.margins(0.02)
    plt.legend((rects1[0], rects2[0], rects3[0]), ('GIST', 'COH', 'ORB'))
    print type(mean_gist)
    comb_list = mean_gist + mean_coh + mean_orb
    low = min(comb_list)
    high = max(comb_list)
    plt.ylim([math.ceil(low - 0.5 * (high - low)), math.ceil(high + 0.5 * (high - low))])
    if filename.__contains__("accuracy"):
        plt.ylabel('Accuracy')
    elif filename.__contains__("traintime"):
        plt.ylabel('Training Time')
    else:
        plt.ylabel('Prediction Time')
    plt.title('Test Accuracy of Classifiers' + label_chng)
    print "\n Plotting ", filename , " ..."
    plt.savefig(c.PATH_FIGURES + filename + 'Default.png')
    print c.PATH_FIGURES + filename + "Default.png saved.\n"
    plt.gcf().clear()
