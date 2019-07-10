import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
from glob import glob
import pandas
from collections import Counter
from imageio import imread
from IPython import embed

random_state = np.random.RandomState(394)
"""
images are color, but are the same across all channels (grayscale)
bottom ~20 pixels in y axis are the scale measurement
they all seem to have "1mm" at the bottom - are there any that are at a different zoom scale"""
datadir = './data'
data_labels = glob(os.path.join(datadir, '*', '*.tsv*'))
for i in range(len(data_labels)):
  print('loading', data_labels[i])
  fp = pandas.read_csv(data_labels[i], sep='\t')
  tsv_name = os.path.split(data_labels[i])[1]
  fp.loc[:,'tsv_name'] = tsv_name
  fp.loc[:,'file_path'] = os.path.join(datadir, tsv_name[-8:-4])
  if not i:
    dd = fp
  else:
    dd = dd.append(fp)
dd.loc[:,'num'] = range(dd.shape[0])
# object_annotation_category
# img_file_name
# object_annotation_status
# object_lat
# object_lon
# object_date
# Cladocera, Copepoda, Rotifera, Holopediidae

unique = list(set(dd['object_annotation_category']))
dont_use = ['unknown', 'othertocheck', 'multiple<other']
labels_to_use = []
for ztype in unique:
    count = np.sum(pandas.Series(dd['object_annotation_category']).str.count(ztype))
    if ztype.lower() not in dont_use:
      labels_to_use.append(ztype)
      print(ztype, count)



# othertocheck 1873
# Holopediidae 778
# Chaoboridae 71
# Insecta 7
# multiple<other 51
# bubble 14
# Volvox 46
# nauplii<Copepoda 84
# Cladocera 4513
# detritus 5287
# Copepoda 9310
# badfocus<artefact 5035
# Arachnida 5
# Unknown 74
# Rotifera 575
# Leptodora 10
# nauplii Copepoda the name for baby copepods:
#   For copepods, the egg hatches into a nauplius form, with a head and a tail
#   but no true thorax or abdomen. The larva molts several times until it resembles
#   the adult and then, after more molts, achieves adult development.
#   The nauplius form is so different from the adult form that it was once thought to be a separate species.

for xc,ml in enumerate(labels_to_use):
  this_dd = dd[dd.loc[:,'object_annotation_category']==ml]
  if not xc:
    many_dd = this_dd
  else:
    many_dd = many_dd.append(this_dd)
many_dd = many_dd.set_index(np.arange(many_dd.shape[0]))

# make train test split
def write_data_file(dataframe, row_inds, data_type, base_dir):
    fname = os.path.join(base_dir, data_type+'.csv')
    print('writing', fname)
    f = open(fname, 'w')

    for i in row_inds:
        name = os.path.join(dataframe.loc[i,'file_path'], dataframe.loc[i,'img_file_name'])
        label = dataframe.loc[i, 'object_annotation_category']
        if label in ['badfocus<artefact', 'detritus', 'bubble']:
            label = 'not_useful'
        f.write("%s,%s\n"%(name, label))
    f.close()

def make_train_test_split(df, exp_name):
    many_inds = np.array(df.index)
    train_rows = []
    valid_rows = []
    for label in labels_to_use:
        this_label_inds = np.array(df[df.loc[many_inds,'object_annotation_category'] == label].index)
        random_state.shuffle(this_label_inds)
        this_row_size = this_label_inds.shape[0]
        n_val = int(this_row_size*.2)
        if n_val < 1:
            n_val+=1
        n_tr = this_row_size-n_val
        print('adding %s valid %s train for %s'%(n_val, n_tr, label))
        valid_rows.extend(list(this_label_inds[:n_val]))
        train_rows.extend(list(this_label_inds[n_val:]))

    overall_dir = os.path.join('experiments', exp_name)
    if not os.path.exists(overall_dir):
        os.makedirs(overall_dir)
    write_data_file(df, valid_rows, 'valid',  overall_dir)
    write_data_file(df, train_rows, 'train',  overall_dir)

exp_name = 'most_merged'
make_train_test_split(many_dd, exp_name)
