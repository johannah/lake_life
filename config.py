import os
from glob import glob

exp_name = 'uvp_adaptive'
data_type = 'uvp'
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/adaptive.html#AdaptiveLogSoftmaxWithLoss
# in train class - these are the counts
# loaded classes
# (40173, 'othertocheck')
# (31712, 'Cladocera') cutoff at 2
# (23967, 'badfocus<artefact')
# (22823, 'Copepoda')
# (15666, 'other<living')
# (10732, 'Holopediidae') -- cutoff at 5
# (5002, 'Rotifera')
# (1468, 'detritus')
# (1417, 'Volvox')
# (1284, 'Chaoboridae') -- cutoff at 9
# (366, 'multiple<other')
# (221, 'part<other') -- cutoff at 11
# (69, 'seaweed')
# (24, 'Arachnida')
# (19, 'Notonecta')
# (17, 'Chironomidae')
# (14, 'not-living') -- cutoff at 16 is default

adaptive_sm_cutoffs = [2, 5, 9, 11]
# make dependent dirs
exp_dir = os.path.join('experiments', exp_name)

base_data_dir = 'data'
uvp_data_dir = 'UVP_data_folder'
data_dir = os.path.join(base_data_dir, uvp_data_dir)
test_data_dir = os.path.join(data_dir, 'test_data')
train_data_dir = os.path.join(data_dir, 'train_data')

mkdirs = [data_dir, exp_dir]
for mdir in mkdirs:
    if not os.path.exists(mdir):
        os.makedirs(mdir)



# make unique checkpoints dir
def get_checkpoints_dir(checkpoints_dir=''):
    if checkpoints_dir != '':
        if not os.path.exists(checkpoints_dir):
           os.makedirs(checkpoints_dir)
        return checkpoints_dir
    else:
        cnt = 0
        checkpoints_dir = os.path.join(exp_dir, 'checkpoints_%02d'%cnt)
        while True:
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)
                break
            elif not len(glob(os.path.join(checkpoints_dir, '*.pt'))):
                # dir is there, but there are not models in it
                break
            else:
                cnt+=1
                checkpoints_dir = os.path.join(exp_dir, 'checkpoints_%02d'%cnt)
        return checkpoints_dir
