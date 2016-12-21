import constants as c
import json

normalization = True
filename = "accuracy"

print "Hello!"

# Load appropriate data
DIR_GIST = c.PATH_RES_GIST_TRAIN
DIR_COH = c.PATH_RES_COH_TRAIN
DIR_ORB = c.PATH_RES_ORB_TRAIN
if normalization:
    filename += "_norm"
filename += "_results_"
ext = ".json"
file_gist = json.load(open(DIR_GIST + filename + 'gist' + ext, "r"))
file_coh = json.load(open(DIR_COH + filename + 'coh' + ext, "r"))
file_orb = json.load(open(DIR_ORB + filename + 'orb' + ext, "r"))
print file_gist
print file_coh
print file_orb

