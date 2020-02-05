from glob import glob
import os
import sys
"""
download UVP_data_folder into data as data/UVP_data_folder/

"""

def unzip_all(data_path):
    search = os.path.join(data_path,'**', '*.zip')
    zips = sorted(glob(search, recursive=True))
    print(search, zips)
    for zip_file in zips:
        zip_dir = zip_file.replace('.zip', '')
        if not os.path.exists(zip_dir):
            print('making dzip_dir', zip_dir)
            os.makedirs(zip_dir)
            cmd = 'unzip %s -d %s'%(zip_file, zip_dir)
            print('calling', cmd)
            os.system(cmd)

if __name__ == '__main__':
    data_path = 'data/UVP_data_folder'
    unzip_all(data_path)
