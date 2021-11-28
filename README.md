# lake_life

This repository contains scripts for training image classification for Canadian freshwater zooplankton. Data, labeling, and guidance was provided by [Riwan Leroux](https://scholar.google.ca/citations?user=NU6iGfEAAAAJ&hl=fr) of UQTR as part of the [GRIL](https://oraprdnt.uqtr.uquebec.ca/pls/public/gscw031?owa_no_site=543) research effort in limnology.  

The purpose of this research is to accurately classify zooplankton species and reduce the labeling burden.  

## Install

Requirements: 

We suggest using python package manager, preferably [Anaconda](https://www.anaconda.com/distribution/)

Once Anaconda is installed, navigate to the directory that you want to work in then:

`git clone github.com/johannah/lake_life'  

`cd lake_life`  

'conda create -n lakelife -f requirements.txt`

## Data

`mkdir -p data/UVP_data_folder`  

`cd data/UVP_data_folder`  

Download images and tsv files from private data source into UVP_data_folder. Zipfiles should be unzipped using unzip_data.py

## Train Classifier

Use a hierarchical loss with a pre-trained resnet to train this unbalanced dataset.

`python train_resnet.py`

Confusion matrix from this model: 

![alt text](https://github.com/johannah/lake_life/blob/master/ckptwt_eval00160_train_normalized_confusion.png)

## Train ACN to Enable Easy Labeling 

Train an [Associative Compressive Network (ACN)](https://arxiv.org/abs/1804.02476) to find similiar zooplankton from unlabeled data to speed up the labeling process. 

`python train_acn.py` 

Given image on the left, the model finds the most similiar images from unlabeled data on the right:

![alt text](https://github.com/johannah/lake_life/blob/master/000066.png)

![alt text](https://github.com/johannah/lake_life/blob/master/000091.png)
