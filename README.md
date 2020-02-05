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

# 

### TODO
double check loading  
fix labels according to RL input
make global config file  
combine model to output  
write tsv file  
figure out what ecotaxa is doing  
add in vertical profiles  
sampling issue?   
add top-1 and top-5 errors
try CB Focal and 320x320 size, zoom, no-zoom
do hierarchical, so even imbalanced classes will come together
try focal-loss
look at other datasets

