import os
from glob import glob

exp_name = 'uvp_warmup'
data_type = 'uvp'
# make dependent dirs
exp_dir = os.path.join('experiments', exp_name)

base_data_dir = 'data'
uvp_data_dir = 'UVP_data_folder'
data_dir = os.path.join(base_data_dir, uvp_data_dir)
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
