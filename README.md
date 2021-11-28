# lake_life

Scripts for training zooplankton class detection. 

## Install
Requirements: 

We suggest using python package manager, preferably [Anaconda](https://www.anaconda.com/distribution/)

Once Anaconda is installed, navigate to the directory that you want to work in then:

`git clone github.com/johannah/lake_life'  

`cd lake_life`  

'conda create -n lakelife -f requirements.txt`

# Data

`mkdir -p data/UVP_data_folder`  

`cd data/UVP_data_folder`  

Download images and tsv files from private data source into UVP_data_folder. Zipfiles should be unzipped using unzip_data.py

# Train

'python train_resnet.py`
