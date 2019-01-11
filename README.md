# Introduction

This project combines few-shot learning method with pruning technique, aiming to produce a few-shot adjustable and small scale model that can run a mobile devices and use only little data to fast master new tasks. 

# Dependencies

python == 3.6

pytorch == 1.0

matplotlib == 2.2.3



# Train and Prune

## 1. Prepare the Dataset

This project is on a handwriting images dataset called [Omniglot](www.omniglot.com/). It consists of 20 instances of 1623 characters from 50 different alphabets. To use the dataset, simply set `Download=True` in `Omniglot.py` and the program will automatically download and process the dataset.

## 2. Train the model

To train the model, execute `python ./train_prune.py` in command line, and the training process. There are several parameters that one can play with, here are descriptions about some important ones:

1. **n_way** : Number of classes in one task, default is 5.
2. **k_spt** : Number of instances of each class during one epoch, default is 1.
3. **meta_lr** : The outer learning rate of meta learner's optimizer.
4. **update_lr** :Task-level inner update learning rate.

## 3. Prune

The pruning process is right after the training process, when training is done, the pruning process will start. Here are some important parameters in the pruning process. 

1. **prune_iteration** : Number of pruning epochs.
2. **finetune_epoch** : Number of finetuning epochs after each pruning epochs.

