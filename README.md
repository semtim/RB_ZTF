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

Details about the architecture of the variational autoencoder are written in Section 3.1 of the article. `notebooks/train_vae.ipynb` was used to train several models, which are saved in `trained_models/vae` directory as well as loss values during training. Script `scripts/vae.py` implements encoder and decoder classes.

Loss values during training different models:
<img src="https://github.com/semtim/RB_ZTF/blob/master/readme_images/vae_loss.png" alt="drawing" width="700"/>

## RNN

Details about the architectures of the considered recurrent neural networks are written in Section 3.2 of the article. `notebooks/train_rnn_kfold.ipynb` was used to train several models, which are saved in `trained_models/rnn` directory as well as additional information during training. During the RNN training, k-fold cross-validation (k=5) was employed. The dataset was randomly divided into 5 folds, and then 5 models were trained, each having a specific fold as the test set and the union of the remaining 4 folds as the training set. Thus, the data split into training/test sets was unique for each model. The final quality metrics for each considered model were calculated as the average values across 5 models from the k-fold cross-validation split. Script `scripts/rnn.py` implements recurrent neural network class.

Loss values during training different models:
<img src="https://github.com/semtim/RB_ZTF/blob/master/readme_images/rnn_losses.png" alt="drawing" width="700"/>

## Validation and visualization

`notebooks/validate_models.ipynb` $-$ this notebook is designed to validate trained rnn models. It calculates quality metrics using kfold cross-validation method, and also plot ROC curves. The results given in this notebook correspond to those given in the article.

Final ROC curves for different models:

<img src="https://github.com/semtim/RB_ZTF/blob/master/readme_images/roc.png" alt="drawing" width="700"/>

Figures (1-3) from article constructed by `notebooks/data_visualization.ipynb`

(**NOTE** To plot figures correctly, you must have installed texlive. You can do this by running the following command in the terminal: `sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super`)
