# Real-bogus classification

This repository represents the research described in the article “Neural network architecture for artifacts detection in ZTF survey“ that has been accepted for publication in “Systems and Means of Informatics” scientific journal.


## Introduction

The goal of this work is the development of an algorithm to predict whether a light curve from the Zwicky Transient Facility data releases has a bogus nature or not, based on the sequence of frames. A labeled dataset provided by experts from [SNAD](https://snad.space/) team was utilized, comprising 2230 frames series. Due to  substantial size of the frame sequences, the application of a variational autoencoder was deemed necessary for mapping the images into lower-dimensional vectors. For the task of binary classification based on sequences of compressed frame vectors, a recurrent neural network was employed. Several neural network models were considered, and the quality metrics were assessed using k-fold cross-validation. The final performance metrics, including $\rm{ROC-AUC}=0.86 \pm 0.01$ and $\rm{Accuracy}=0.80 \pm 0.02$, suggest that the model has practical utility.

The figure below shows a diagram of how the real-bogus classificator works:

![](https://github.com/semtim/RB_ZTF/blob/master/readme_images/scheme_eng.png)

The work consisted of several steps:
- Download data
- Train variational autoencoder (VAE)
- Save embeddings of frames
- Train recurrent neural network (RNN)
- Validate final models


## Data

Details about data type and preprocessing are written in Section 2 of the article.

The labeled objects list is contained in the file `akb.ztf.snad.space.json`. Script `download_and_cut_fits/download_and_cut_fits.py` contains code for downloading all full-sized fits for object with certain OID, cutting these fits to 28x28pix image, whose center corresponds to the coordinates of the object and saving cuted fits. This script was not used because it took too long to execute. Instead of this, `download_cuted_fits/download_cuted_fits.py` was used. It contains code, which download already cuted fits from [IPAC](https://irsa.ipac.caltech.edu/docs/program_interface/ztf_api.html). `data/` directory containing sequences of object frames cannot be uploaded to GitHub because it takes up too much space. However, the embeddings for these images are preserved in `embeddings`.

`datasets.py` implements frame normalization functions as well as dataset classes for training VAE and RNN.

## VAE

## RNN

## Validation
