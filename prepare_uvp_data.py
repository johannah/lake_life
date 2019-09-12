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
datadir = './data/UVP_data_folder/'

"""
images are color, but are the same across all channels (grayscale)
bottom ~20 pixels in y axis are the scale measurement
they all seem to have "1mm" at the bottom - are there any that are at a different zoom scale"""

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
    dd.loc[:,'num'] = range(dd.shape[0])
    dd.to_csv(summary_file, sep=',', header=True)
else:
    dd = pd.read_csv(summary_file, sep=',')

# object_annotation_category
# img_file_name
# object_annotation_status
# object_lat
# object_lon
# object_date
# Cladocera, Copepoda, Rotifera, Holopediida

unique = list(set(dd['object_annotation_category']))
dont_use = ['unknown', '[t]',
            'living', 'not-living', 'other<living',
            'part<other',
            'part<copepoda', 'othertocheck']

labels_to_use = []
class_count = []
for ztype in unique:
    count = np.sum(pd.Series(dd['object_annotation_category']).str.count(ztype))
    if ztype.lower() not in dont_use:
      labels_to_use.append(ztype)
      print(ztype, count)
      class_count.append(count)

print(labels_to_use)
print(class_count)
#individual_labels = ['cladocera', 'copepoda', 'holopediidae',
#                      'badfocus<artefact', 'othertocheck',
#                      'chaoboridae', 'detritus', 'volvoxlike', 'rotifera']
#
#individual_labels = ['cladocera', 'copepoda', 'holopediidae',
#                      'badfocus<artefact',
#                      'chaoboridae', 'detritus', 'volvoxlike', 'rotifera']
"""
to check -
 the variants of living are not really classes
what is part<copepoda look like
[t]
not-living

other<plastic
multiple<Copepoda
multiple<other
"""
# most important
# cladocera
# copepoda
# cxxx
# holopedidae
# other to check is a combo of cladocera and copepoda
# Some copepoda are big, am I cutting them off?
# multiple class doesn't matter- doesnt matter
# in the horizontal transects, seaweed is very big
# plot ('part<Copepoda', 238)
#####################
#('[t]', 166755)
#('living', 18017)
#('not-living', 4)
#('other<living', 17509)

#('badfocus<artefact', 25635)
#('othertocheck', 44064)
#('Chaoboridae', 1352)

#('detritus', 1646)
#('volvoxlike', 1567)
#('Rotifera', 5516)

#('Notonecta', 20)
#('Arachnida', 29)
#('Chironomidae', 17)
#('Volvox', 38)
#('seaweed', 81)
#('part<other', 8)
#('Unknown', 52)
#('part<Copepoda', 238)
#('fiber<detritus', 3)
#('multiple<Copepoda', 55) -- > how to handle
#('multiple<other', 292) ---> how to handle
#('other<plastic', 11)

####################
#####################
# Zooscan
####################
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
    print('adding', ml, this_dd.shape)
    if not xc:
        many_dd = this_dd
    else:
        many_dd = many_dd.append(this_dd)
many_dd = many_dd.set_index(np.arange(many_dd.shape[0]))

# make train test split
def write_data_file(dataframe, row_inds, data_type, base_dir):
    fname = os.path.join(base_dir, data_type+'.csv')
    labels = []
    print('writing', fname)
    f = open(fname, 'w')
    for i in row_inds:
        label = dataframe.loc[i, 'object_annotation_category']
        file_path = os.path.join(datadir, dataframe.loc[i,'exp_name'], dataframe.loc[i, 'sub_exp_name'], dataframe.loc[i, 'img_file_name'])
        if label in ['badfocus<artefact', 'detritus']:
            class_label = 'not_useful'
        elif class_count[labels_to_use.index(label)] < 1800:
            class_label = 'small_class'
        else:
            class_label = label.lower()
        if class_label not in labels:
            labels.append(class_label)
        f.write("%s,%s,%s\n"%(file_path,class_label,label))
    f.close()
    print("CLASSES INCLUDE", labels)

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

#exp_name = 'uvp_big_1000small_noliving_norotate_other_bg0'
exp_name = 'uvp_big_1000small_noliving_rotate_bg0_trim_combine_new'
make_train_test_split(many_dd, exp_name)
