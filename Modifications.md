# Modifications and Notes for STC

Christian Ritter

* Setup for kaggle_carvana_segmentation task


## Comments

* Full training implementation
    * Train functions including loss function are implemented for TernausNet
    * Cyclical learning rates implemented

* Can easily switch to TernausNetV2 in unet_models.py
* prepare_folds.py creates folds of training data. Modified prepare_folds to just use one fold for testing.


