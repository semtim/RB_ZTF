## Data

`download_and_cut_fits/` contains code for downloading all full-sized fits for object with certain OID, cutting these fits to 28x28pix image, whose center corresponds to the coordinates of the object and saving cuted fits.

`download_cuted_fits/` contains code, which download already cuted fits from IPAC.

`datasets.py` implements frame normalization functions as well as dataset classes for training VAE and RNN.

## Models
The files `vae.py` and `rnn.py` contain architectures for neural networks and necessary functions for training the models.

VAE training was performed using a notebook `train_vae.ipynb`. `train_rnn_kfold.ipynb` was used for training the RNN, which implements k-fold cross-validation. The trained models are saved in a `trained_models/`.
