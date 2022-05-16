from models import keras_mt_shared_cnn
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import f1_score
import math
import pickle5 as pickle
import numpy as np
import os
from mpi4py import MPI
from NewDataLoader import loadAllTaskNoID
import argparse
import pandas as pd
#from models import evaluationMetrics

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

rank = rank+29
'''
Datasets Sizes: 
Registry:  0  Size: 85789
Registry:  1  Size: 577094
Registry:  2  Size: 137135
Registry:  4  Size: 441732
Registry:  5  Size: 360375
Registry:  6  Size: 365152
'''

model_name = "MtCNN"
# parser = argparse.ArgumentParser()
# parser.add_argument("r_target", help="Target Registry [0,1,2,3]", type=int)
# args = parser.parse_args()
#r_target = args.r_target
r_target=4 #use the bigger dataset as the OOD
#TASK = "Histology"
NUM_CLASES_TASK = [70, 327, 7, 645, 4]
num_classes = NUM_CLASES_TASK
CLASSES_TASK = {"Site":0, "Subsite":1, "Laterality":2, "Histology":3, "Behavior": 4}

model_dir = "//gpfs/alpine/med107/proj-shared/kevindeangeli/interRegistrySavedModels/destilationModels/DataWithID/R"+str(r_target)+"_Ensemble/"
scores_dir = "results/" + "R" + str(r_target) + "/"

#dirs = 'checkpoints'
if rank == 0:
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
if rank == 0:
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)


def run_cnn(
        maxlen = 1500,
        wv_len = 300,
        epochs = 100,
        filter_k1 = 3,
        filter_k2 = 3,
        num_filters = 300,
        concat_dropout_prob=0.5,
        fc_layers=0,
        n_fc=0,
        fc_dropout=0.0,
        batch_size= 128,
        learning_rate= 1.0
    ):

    filter_sizes = []
    n_filters = []

    for k in range( filter_k1, filter_k1 + filter_k2 ):
        filter_sizes.append( k )
        n_filters.append( num_filters )

    train_x, train_y, val_x, val_y, test_x, test_y, unseen_x, unseen_y = loadAllTaskNoID(r_target,print_shapes=True)


    # prop = 0.1
    # propX = int(prop * len(train_x))
    # propXT = int(prop * len(test_x))
    # propXV = int(prop * len(val_x))
    #
    # train_x = train_x[0:propX]
    # val_x = val_x[0:propXV]
    # test_x = test_x[0:propXT]
    # train_y = train_y[0:propX]
    # val_y = val_y[0:propXV]
    # test_y = test_y[0:propXT]


    wv_mat = np.random.uniform( low= -0.05, high= 0.05, size= ( 200000, num_filters ) )

    import tensorflow as tf
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        cnn = keras_mt_shared_cnn.init_export_network(
            num_classes= num_classes,
            in_seq_len= 1500,
            vocab_size= len( wv_mat ),
            wv_space= len( wv_mat[ 0 ] ),
            filter_sizes= filter_sizes,
            num_filters= n_filters,
            concat_dropout_prob = concat_dropout_prob,
            fc_layers = fc_layers,
            n_fc = n_fc,
            fc_dropout= fc_dropout,
            optimizer= 'adam',
            learning_rate= learning_rate )

        stopper = EarlyStopping( monitor= 'val_loss', min_delta= 0, patience= 5, verbose= 0, mode= 'auto', restore_best_weights= True )

    #print( cnn.summary() )
    validation_data = ({'Input': np.array(val_x)},
                           {'Dense0': val_y[:,0],
                            'Dense1': val_y[:,1],
                            'Dense2': val_y[:,2],
                            'Dense3': val_y[:,3],
                            'Dense4': val_y[:,4],
                            #'Dense5': val_y[:, 5],
                            })


    ret = cnn.fit( x= np.array( train_x ),
                 y= [
                     np.array(train_y[:,0]),
                     np.array(train_y[:,1]),
                     np.array(train_y[:,2]),
                     np.array(train_y[:,3]),
                     np.array(train_y[:,4]),
                     #np.array(train_y[:,5]),
                 ],
                 batch_size= batch_size, epochs= 100,
                    verbose= 2,
                    validation_data= validation_data, callbacks= [ stopper ]
                 )


    cnn.save( model_dir + '/model.' + str( rank ) + '.h5' )

    train_pred = cnn.predict( train_x )
    val_pred = cnn.predict( val_x )
    test_pred = cnn.predict( test_x )
    unseen_pred = cnn.predict(unseen_x)

    with open( model_dir + '/preds_' + str( rank ) + '.pickle', 'wb' ) as f:
        pickle.dump( train_pred, f, protocol= pickle.HIGHEST_PROTOCOL )
        pickle.dump( val_pred, f, protocol= pickle.HIGHEST_PROTOCOL )
        pickle.dump( test_pred, f, protocol= pickle.HIGHEST_PROTOCOL )
        pickle.dump( unseen_pred, f, protocol= pickle.HIGHEST_PROTOCOL )


    micMac= []
    preds_probs = test_pred
    for t in range(len(num_classes)):
        preds = [ np.argmax( x ) for x in preds_probs[ t ] ]
        micro = f1_score( test_y[ :, t ], preds, average= 'micro' )
        macro = f1_score( test_y[ :, t ], preds, average= 'macro' )
        micMac.append(micro)
        micMac.append(macro)

    data = np.zeros(shape=(1,  len(num_classes)*2))
    data = np.vstack((data, micMac))

    #Calculate unseen:
    micMac= []
    preds_probs = unseen_pred
    for t in range(len(num_classes)):
        preds = [ np.argmax( x ) for x in preds_probs[ t ] ]
        micro = f1_score( unseen_y[ :, t ], preds, average= 'micro' )
        macro = f1_score( unseen_y[ :, t ], preds, average= 'macro' )
        micMac.append(micro)
        micMac.append(macro)

    #CLASSES_TASK = {"Site":0, "Subsite":1, "Laterality":2, "Histology":3, "Behavior": 4}
    data = np.vstack((data, micMac))
    df0 = pd.DataFrame(data,
                       columns=['Sit_Mic', 'Sit_Mac', 'Sub_Mic', 'Sub_Mac', 'Lat_Mic', 'Lat_Mac', 'His_Mic',
                                'His_Mac', 'Be_Mic', 'Be_Mac'])

    df0.to_csv(scores_dir + str(rank) + "_test.csv")

    '''
    #############################################################################
    Save loss at each epoch:
    #############################################################################
    '''
    numEpochs = len(ret.history['val_Dense0_loss'])
    Valloss_data = []
    for i in range(5):
        Valloss_data.append(ret.history['val_Dense' + str(i) + '_loss'])

    #Store Validation Loss
    Valloss_data = np.reshape(Valloss_data, (5, numEpochs))
    df0 = pd.DataFrame(Valloss_data.T,
                       columns=['Sit_ValLoss', 'Sub_ValLoss', 'Lat_ValLoss', 'His_ValLoss',\
                                'Be_ValLoss'])
    df0.to_csv(scores_dir + str(rank) + "_loss.csv")
    #CLASSES_TASK = {"Site":0, "Subsite":1, "Laterality":2, "Histology":3, "Behavior": 4}


    # pred = [val_pred, test_pred, unseen_pred]
    # true = [val_y, test_y, unseen_y]
    # evaluationMetrics.find97AccThresholdTASK(pred, true, saveDir=model_name)
    # evaluationMetrics.makeSitePredHistogram(unseen_pred, saveDir=model_name)





if __name__ == "__main__":
    # main()
    run_cnn()

