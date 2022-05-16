# EnsembleDistillation

The experiments are performed in three phases:

##### 1) File mt_cnn_exp.py trains an ensemble of MtCNNs with different random seeds.
##### 2) File make_softlabel.py creates and stores soft-labels using the ensemble predictions.
##### 3) File mt_cnn_soft.py trains a single model with the stored soft labels from the previous step. 

Notes:
 #####  - Data is not included. A loadData method is included but not defined. 
