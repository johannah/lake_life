import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
from glob import glob
import pandas as pd
from collections import Counter
from imageio import imread
from IPython import embed

random_state = np.random.RandomState(394)

"""
#####################
######################
## UVP
######################

images are color, but are the same across all channels (grayscale)
bottom ~20 pixels in y axis are the scale measurement
they all seem to have "1mm" at the bottom - are there any that are at a different zoom scale
 other to check is a combo of cladocera and copepoda

Some copepoda are big, am I cutting them off?
multiple class doesn't matter- doesnt matter
in the horizontal transects, seaweed is very big

#####################
######################
## Zooscan
#####################
## othertocheck 1873
## Holopediidae 778
## Chaoboridae 71
## Insecta 7
## multiple<other 51
## bubble 14
## Volvox 46
## nauplii<Copepoda 84
## Cladocera 4513
## detritus 5287
## Copepoda 9310
## badfocus<artefact 5035
## Arachnida 5
## Unknown 74
## Rotifera 575
## Leptodora 10
## nauplii Copepoda the name for baby copepods:
##   For copepods, the egg hatches into a nauplius form, with a head and a tail
##   but no true thorax or abdomen. The larva molts several times until it resembles
##   the adult and then, after more molts, achieves adult development.
##   The nauplius form is so different from the adult form that it was once thought to be a separate species.
#

TODO - make dataset of just Holepedidae
"""


class_rules = {
        'Arachnida': '', #29 spider!
        'Chaoboridae': '', #1459 important tactile predator that can eat fast (mosquito larve)
        'Chironomidae': '', #17
        'Cladocera': '', #73077 important to see where they move
        'Copepoda': '',  #37088 important
        'Copepoda X': 'Copepoda', #234 TODO ask RL about this
        'Holopediidae': '', #15797 jello capsule which is gross to fish
        'Notonecta': '', #20, TODO ask RL
        'Rotifera': '', #7137,  really tiny zooplankton in a colony - don't really care about this class
        'Unknown': 'none', #77,
        'Volvox': '', #47
        'badfocus<artefact':'', # 39021,
        'daphnia like': 'daphnia', #567
        'detritus': '', # 3301,
        'fiber<detritus': 'detritus', #12,
        'living': 'none', # 504  RL - don't use this!
        'multiple<Copepoda': 'multiple<other', #73,
        'multiple<other': '', #341,
        'not-living': 'none', #4,
        'other<living': '', #41111,
        'other<plastic': 'none', # 11
        'othertocheck': '', #80488,
        'part<Copepoda': 'part<other', #293,
        'part<other': '',
        'seaweed': '', #207,
        'volvoxlike': 'Volvox', #1936
        }


def load_tsv_files(data_dir, summary_path="data/UVP_data_folder/summary_path.tsv"):
    """
    create (or load) a summary tsv file that encapsulates the entire dataset
    follow the rules in class_rules to combine or trash particular classes
    """
    data_labels = sorted(glob(os.path.join(data_dir, '**', '*.tsv'), recursive=True))
    print('found %s tsv files' %len(data_labels))
    cat_path = summary_path.replace('.tsv', '_counts.tsv')
    if not os.path.exists(summary_path):
        file_categories_dict = {'all':{}}
        for i in range(len(data_labels)):
            print('loading', data_labels[i])
            # row 1 (zero indexed) is parameter which tells datatype
            try:
                fp = pd.read_csv(data_labels[i], sep='\t', skiprows=[1])
            except UnicodeDecodeError as e:
                print('error loading %s - trying different encoding'%data_labels[i])
                fp = pd.read_csv(data_labels[i], sep='\t', skiprows=[1], encoding = "ISO-8859-1")
            base = os.path.split(os.path.split(data_labels[i])[0])[0]
            export = os.path.split(base)[1]
            profile = os.path.split(os.path.split(base)[0])[1]
            tsv_name = os.path.split(data_labels[i])[1]
            quick_name = profile+'__'+export+'__'+tsv_name.replace('.tsv','')
            fp.loc[:,'sub_exp_name'] = os.path.split(os.path.split(data_labels[i])[0])[1]
            fp.loc[:,'exp_name'] = os.path.split(os.path.split(os.path.split(data_labels[i])[0])[0])[1]
            fp.loc[:,'tsv_name'] = tsv_name
            fp.loc[:,'quick_name'] = quick_name
            fp.loc[:,'dir_name'] = os.path.split(data_labels[i])[0]
            # get counts for each type of label in this tsv file

            object_classes = fp['object_annotation_category']
            refined_object_classes = []
            for name in object_classes:
                if class_rules[name] != '':
                    # use rules to rename
                    refined_object_classes.append(class_rules[name])
                else:
                    refined_object_classes.append(name)

            fp['class'] = refined_object_classes
            output = fp['class'].value_counts()
            file_categories_dict[quick_name] =  dict(output)
            for name, val in output.items():
                if name not in file_categories_dict['all'].keys():
                    print('adding', name)
                    file_categories_dict['all'][name] = val
                else:
                    file_categories_dict['all'][name]+=val
            if not i:
              dd = fp
            else:
              dd = dd.append(fp)
        dd.loc[:,'num'] = range(dd.shape[0])
        dd.to_csv(summary_path, sep=',', header=True)
        file_categories = pd.DataFrame.from_dict(file_categories_dict, orient='index')
        file_categories.to_csv(cat_path, sep=',', header=True)
        # this returns wrong, but load is fine
    else:
        dd = pd.read_csv(summary_path, sep=',')
        file_categories = pd.read_csv(cat_path, sep=',', index_col=0)
    print(file_categories.loc['all'])
    return dd, file_categories


def write_data_file(file_path, rows):
    fdir = os.path.split(file_path)[0]
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    f = open(file_path, 'w')
    [f.write(row+'\n') for row in rows]
    f.close()

def make_train_test_split(df, exp_name):
    many_inds = np.array(df.index)
    random_state.shuffle(many_inds)
    n_valid = int(many_inds.shape[0]*.12)
    valid_class_count_dict = {}
    valid_inds = many_inds[:n_valid]
    train_rows = []
    valid_rows = []
    overall_dir = os.path.join('experiments', exp_name)
    if not os.path.exists(overall_dir):
        os.makedirs(overall_dir)
    for cnt in many_inds:
        file_path = os.path.abspath(os.path.join(df.loc[cnt, 'dir_name'], df.loc[cnt, 'img_file_name']))
        if not os.path.exists(file_path):
            print("could not find image file:", file_path)
            embed()
        dclass = df.loc[cnt, 'class']
        line = '%d,%s,%s'%(cnt, dclass, file_path)
        # we need at least one example of each
        if dclass not in valid_class_count_dict.keys():
            valid_class_count_dict[dclass] = 1
            valid_rows.append(line)
        elif cnt in valid_inds:
            valid_rows.append(line)
            valid_class_count_dict[dclass] += 1
        else:
            train_rows.append(line)

    write_data_file(os.path.join(exp_dir, 'valid.csv'), valid_rows)
    write_data_file(os.path.join(exp_dir, 'train.csv'), train_rows)

if __name__ == '__main__':
    # each UVP folder has a subdir then another dir w/ tsv file
    from config import exp_dir, data_dir
    summary_path = os.path.join(data_dir, 'data_summary.tsv')
    summary, all_categories = load_tsv_files(data_dir, summary_path)
    summary, all_categories = load_tsv_files(data_dir, summary_path)
    make_train_test_split(summary, exp_dir)
