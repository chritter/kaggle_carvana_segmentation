# Modifications and Notes for STC

Christian Ritter

* Setup for kaggle_carvana_segmentation task


## Comments

* Full training implementation
    * Train functions including loss function are implemented for TernausNet
    * Learning Rate combination of Adam and cyclical exponential-decaying learning rate. (see notebook). Differetn from exp in Smith17?

* Can easily switch to TernausNetV2 in unet_models.py
* prepare_folds.py creates folds of training data. Modified prepare_folds to just use one fold for testing.
* Uses combination of cross entropy and dice loss
* Once the model is restarted tqdm does not show the correct step number as it is not updated when step number is set at model checkpoint load

