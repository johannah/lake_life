import os
from glob import glob

zip_files = glob('*.zip*')
for zip_file in zip_files:
    dfile = zip_file.replace('.zip', '')
    os.system('mkdir %s'%dfile)
    os.system('unzip %s -d %s'%(zip_file, dfile))

