from models import keras_mt_shared_cnn
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import f1_score
import math
import pickle
from NewDataLoader import loadAllTaskNoID
import numpy as np
import os
from mpi4py import MPI
import argparse
from models import evaluationMetrics
import pandas as pd
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# parser = argparse.ArgumentParser()
# parser.add_argument("r_target", help="Target Registry [0,1,2,3]", type=int)
# args = parser.parse_args()
# r_target = args.r_target
r_target = 4
EnsembleSize = 1000
#model_dir = "//gpfs/alpine/med107/proj-shared/kevindeangeli/interRegistrySavedModels/destilationModels/DataWithID/R4_Ensemble/"
model_dir = "//gpfs/alpine/med107/proj-shared/kevindeangeli/interRegistrySavedModels/destilationModels/DataWithID/R4_Ensemble/"
scores_dir = "results/" + "R" + str(r_target) + "/student/"
model_name = "Student" + str(EnsembleSize) + str(rank)
load_model_name = "ensemble1000"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if rank == 0:
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)
#
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

#scores_dir = "results/"


#student_models = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 14]

if rank == 0:
    rank=8
elif rank == 1:
    rank = 10
elif rank == 2:
    rank=13
else:
    rank = rank+12


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



    #train_x, train_y, val_x, val_y, test_x, test_y, unseen_x, unseen_y = loadAllTasks(r_target,print_shapes=True)
    train_x, train_y, val_x, val_y, test_x, test_y, unseen_x, unseen_y = loadAllTaskNoID(r_target)

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

    # binarize labels
    NUM_CLASES_TASK = [70, 327, 7, 645, 4]
    num_classes = NUM_CLASES_TASK
    #num_classes = [ 4, 639, 7, 70, 326 ]

    val_bin_y = []
    for t in range( len( num_classes ) ):
        arr = np.zeros( ( len( val_y[ :, t ] ), num_classes[ t ] ), dtype= 'int32' )
        for k in range( len( val_y[ :, t ] ) ):
            arr[ k, int( val_y[ k, t ] ) ] = 1
        val_bin_y.append( arr )

    test_bin_y = []
    for t in range( len( num_classes ) ):
        arr = np.zeros( ( len( test_y[ :, t ] ), num_classes[ t ] ), dtype= 'int32' )
        for k in range( len( test_y[ :, t ] ) ):
            arr[ k, int( test_y[ k, t ] ) ] = 1
        test_bin_y.append( arr )


    # load soft labels
    with open( model_dir +"softlabels/" + load_model_name + '_softy.pickle', 'rb' ) as f:
        train_soft = pickle.load( f )
        val_soft = pickle.load( f )
        test_soft = pickle.load( f )

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

        stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
                                restore_best_weights=True)

    cnn.compile( loss= "categorical_crossentropy", optimizer= 'adam', metrics=[ "acc" ] )

    print( cnn.summary() )

    validation_data = ({'Input': np.array(val_x)},
                           {'Dense0': val_bin_y[0],
                            'Dense1': val_bin_y[1],
                            'Dense2': val_bin_y[2],
                            'Dense3': val_bin_y[3],
                            'Dense4': val_bin_y[4],
                            })




    ret = cnn.fit( x= np.array( train_x ),
                 y= train_soft,
                 batch_size= batch_size, epochs= 100,
                 verbose= 2,
                 #class_weight= class_weight,
                 validation_data= validation_data, callbacks= [ stopper ]
                 )

    #Save Student model:
    student_model_dir =  model_dir + '/studentModel/'
    if not os.path.exists(student_model_dir):
        os.makedirs(student_model_dir)
    cnn.save( student_model_dir + '/' + model_name + '_' + str( rank ) + '.h5' )


    train_pred = cnn.predict( train_x )
    val_pred = cnn.predict( val_x )
    test_pred = cnn.predict( test_x )
    unseen_pred = cnn.predict( unseen_x )


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

    data = np.vstack((data, micMac))
    df0 = pd.DataFrame(data,
                       columns=['Sit_Mic', 'Sit_Mac', 'Sub_Mic', 'Sub_Mac', 'Lat_Mic', 'Lat_Mac', 'His_Mic',
                                'His_Mac', 'Be_Mic', 'Be_Mac'])

    df0.to_csv(scores_dir + model_name + "_" + str(rank) + "Test.csv")


    with open( student_model_dir + '/'+ model_name + '_preds_' + str( rank ) + '.pickle', 'wb' ) as f:
        pickle.dump( train_pred, f, protocol= pickle.HIGHEST_PROTOCOL )
        pickle.dump( val_pred, f, protocol= pickle.HIGHEST_PROTOCOL )
        pickle.dump( test_pred, f, protocol= pickle.HIGHEST_PROTOCOL )
        pickle.dump( unseen_pred, f, protocol= pickle.HIGHEST_PROTOCOL )


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
    df0.to_csv(scores_dir + model_name + str(rank) + "_loss.csv")

    pred = [val_pred, test_pred, unseen_pred]
    true = [val_y, test_y, unseen_y]
    evaluationMetrics.find97AccThresholdTASK(pred, true, saveDir=model_name)
    #evaluationMetrics.makeSitePredHistogram(unseen_pred, saveDir=model_name)




    
    


if __name__ == "__main__":
    # main()
    run_cnn()

