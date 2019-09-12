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
find out which files were used in tsv so we know the label
full image path,target,label
"""


def load_dataset_files(datadir):
    # each UVP folder has a subdir then another dir w/ tsv file
    summary_file = os.path.join(datadir, 'data_summary.tsv')
    if not os.path.exists(summary_file):
        data_labels = glob(os.path.join(datadir, '*', '*', '*.tsv*'))
        for i in range(len(data_labels)):
            print('loading', data_labels[i])
            fp = pd.read_csv(data_labels[i], sep='\t')
            tsv_name = os.path.split(data_labels[i])[1]
            fp.loc[:,'sub_exp_name'] = os.path.split(os.path.split(data_labels[i])[0])[1]
            fp.loc[:,'exp_name'] = os.path.split(os.path.split(os.path.split(data_labels[i])[0])[0])[1]
            fp.loc[:,'tsv_name'] = tsv_name
            fp.loc[:,'dir_name'] = data_labels[i]
            if not i:
                dd = fp
            else:
                dd = dd.append(fp)
        dd.to_csv(summary_file, sep=',', header=True)
    else:
        dd = pd.read_csv(summary_file, sep=',')
    dd.loc[:,'num'] = range(dd.shape[0])
    return dd

def load_training_files(exp_name):
    # load train and test csvs and combine
    train = pd.read_csv(os.path.join('experiments', exp_name, 'train.csv'), names=['file', 'class_label', 'label'])
    train['phase'] = 'train'
    valid = pd.read_csv(os.path.join('experiments', exp_name, 'train.csv'), names=['file', 'class_label', 'label'])
    valid['phase'] = 'valid'
    all_data = train.append(valid)
    all_data.index = np.arange(all_data.shape[0])
    return all_data

def get_images(datadir, labeled_data, exp_data):
    output_csv = os.path.join(datadir, 'output_summary_class.csv')
    if not os.path.exists(output_csv):
        # we know the label for these images
        labeled_data.index = np.arange(labeled_data.shape[0])
        labeled_jpgs = [os.path.join(datadir, labeled_data.loc[i,'exp_name'], labeled_data.loc[i, 'sub_exp_name'], labeled_data.loc[i, 'img_file_name']) for i in range(labeled_data.shape[0])]

        exp_files = list(exp_data.loc[:,'file'])
        jpgs = sorted(glob(os.path.join(datadir, '*/*/*.jpg')))
        output = []
        ll_cnt = 0
        exp_cnt = 0
        ul_cnt = 0
        for jpg in jpgs:
            if jpg in labeled_jpgs:
                index = labeled_jpgs.index(jpg)
                label = labeled_data.loc[index, 'object_annotation_category'].lower()
                # dependent on the data that the network was trained on
                if jpg in exp_files:
                    exp_index = exp_files.index(jpg)
                    exp_class = exp_data.loc[exp_index,'class_label'].lower()
                    exp_files.pop(exp_index)
                    exp_cnt +=1
                else:
                    exp_index = -1
                    exp_class = 'none'
                row = labeled_data.loc[index]
                # TODO - add class label
                output.append([jpg,  exp_class, label, exp_index, index])
                ll_cnt +=1
                # reduce search 
                labeled_jpgs.pop(index)
            else:
                ul_cnt +=1
                output.append([jpg, 'unk', 'unk', 'unk', 'unk']) 

        print(exp_cnt, ll_cnt)
        outpd = pd.DataFrame(np.array(output), columns=['file', 'class_label', 'label', 'experiment_index', 'summary_index'])
        outpd.to_csv(output_csv, sep=',')
        embed()
    else:
        outpd = pd.read_csv(output_csv)
    return outpd

if __name__ == '__main__':
    datadir = './data/UVP_data_folder/'
    exp_name = 'uvp_big_1000small_noliving_rotate_bg0_trim_combine'
    exp_data = load_training_files(exp_name)
    labeled_data = load_dataset_files(datadir)
    all_image_paths = get_images(datadir, labeled_data, exp_data)

