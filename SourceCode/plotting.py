import matplotlib.pyplot as plt
import constants as c
import json
import numpy as np
import math
from scipy.ndimage import imread
from PIL import Image
from PIL import ImageOps
#import cv2

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

def plot_gist_svm(grid):
    kernels = ['rbf', 'linear']
    Cs = [10 ** x for x in range(-2, 2)]
    Gammas = [10 ** x for x in range(-2, 2)]
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(kernels), len(Cs), len(Gammas))

    for ind, i in enumerate(Cs):
        plt.plot(Gammas, scores[0][ind], label='kernel: rbf,  C: ' + str(i))
        plt.plot(Gammas, scores[1][ind], label='kernel: linear,  C: ' + str(i))
    plt.legend()
    plt.xlabel('Gamma')
    plt.ylabel('Mean Score')
    plt.title('SVM Tuning on GIST features')
    plt.savefig(c.PATH_FIGURES+'SVM_tuning_GIST.png')


def plot_gist_nn(grid):
    hidden_layer_sizes = [25, 50, 100, 150]
    print grid
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores)
    plt.plot(np.array(hidden_layer_sizes), scores)
    plt.legend()
    plt.xlabel('Hidden Layer Sizes')
    plt.ylabel('Mean Score')
    plt.title('Multi-layer perceptron tuning on GIST features')
    plt.savefig(c.PATH_FIGURES+'NN_tuning_GIST.png')

def plot_coh_rf(grid):
    n_estimators = range(200,300,20)
    max_features = range(1,5)
    max_depth = range(10, 40, 10)
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(n_estimators), len(max_features), len(max_depth))

    for ind, i in enumerate(max_features):
        for n in range(len(n_estimators)):
            plt.plot(max_depth, scores[n][ind], label='n_estimators: ' + str(n_estimators[n]) + ' C: ' + str(i))
    #plt.legend()
    plt.xlabel('max_depth')
    plt.ylabel('Mean Score')
    plt.title('Random forest tuning on COH features')
    plt.savefig(c.PATH_FIGURES+'RF_tuning_COH.png')


def plot_coh_ab(grid):
    n_estimators = range(10,150,10)
    print grid
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores)
    plt.plot(np.array(n_estimators), scores)
    plt.legend()
    plt.xlabel('Number of estimators')
    plt.ylabel('Mean Score')
    plt.title('Ada Boost Classifier on COH features')
    plt.savefig(c.PATH_FIGURES+'AB_tuning_COH.png')


def plot_film_strip(file, subdir, pred_heading, correct_heading):
    # This function displays two images 6 images side-by-side
    # Plot the clean and noisy images
    plt.figure(0, figsize=(18, 3))
    plt.title('Correct estimate')
    DIR = c.PATH_IMAGES_GSV + subdir
    #filename = DIR + 'church_%d.jpg'
    filename = DIR + 'shopfront_%d.jpg'

    #util.plot_pair(img_clean, img_noisy, "Clean", "Noisy")
    #plt.savefig('../../Figures/CleanVsNoisy.png')
    # Extra Credit
    # plt.savefig('../../Figures/SVD_CleanVsNoisy.png')

    for i in range(1, 7, 1):
        Title = (i-1)*60
        imageName = filename % Title
        #img = imread(imageName, flatten=False) / 255.0
        img = Image.open(imageName)
        if Title == 240:
            img = ImageOps.expand(img, border=20, fill='red')
        if Title == 300:
            img = ImageOps.expand(img, border=20, fill='blue')
    #    if Title == 180:
    #        img = ImageOps.expand(img, border=20, fill='#32cd32')
        plt.subplot(1, 6, i)
        plt.imshow(img, interpolation="nearest")
        plt.xticks(())
        plt.yticks(())
        plt.title(str(Title), fontsize= 25)
    plt.tight_layout()
    plt.savefig(c.PATH_FIGURES+file)

#plot_film_strip('correct_filmstrip.png', '42.3742566x-72.5220319/', 180, 180)
plot_film_strip('incorrect_filmstrip.png', '42.3747951x-72.5204444/', 180, 180)