'''
Created by: Kevin De Angeli
Email: Kevindeangeli@utk.edu
Date: 4/12/21

Files were saved with format:
    with open( 'underspec/data.reg.' + str( train_reg ) + '.pickle', 'wb' ) as f:
        pickle.dump( train_x_tokens, f, protocol= pickle.HIGHEST_PROTOCOL )
        pickle.dump( train_y, f, protocol= pickle.HIGHEST_PROTOCOL )
        pickle.dump( val_x_tokens, f, protocol= pickle.HIGHEST_PROTOCOL )
        pickle.dump( val_y, f, protocol= pickle.HIGHEST_PROTOCOL )
        pickle.dump( test_x_arr, f, protocol= pickle.HIGHEST_PROTOCOL )
        pickle.dump( test_y_arr, f, protocol= pickle.HIGHEST_PROTOCOL )
        pickle.dump( le_array, f, protocol= pickle.HIGHEST_PROTOCOL )


Data format: data.reg.0.pickle<-- train/val comes from R0. But Test comes the combination of all others.
This data is meant to train on Registry A, and test on Registry B,C,D, ...


RawDataPath = "/gpfs/alpine/proj-shared/med107/NCI_Data/hjy/sevenreg/underspec/underspec_raw_data.pickle"
data0 = "/gpfs/alpine/proj-shared/med107/NCI_Data/hjy/sevenreg/underspec/data.reg.0.pickle"
Class Order: [ site, subsite, laterality, histology, behavior ]

'''
import numpy as np
import pickle5 as pickle
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA = "/gpfs/alpine/proj-shared/med107/NCI_Data/hjy/sevenreg/underspec/dataWithDocID/data.reg." #NewDatawithIDs
TASK_DIC = {"Site":0, "Subsite":1, "Laterality":2, "Histology":3, "Behavior": 4}

def getEnconder(target_reg):
    filename = DATA + str(target_reg) + '.pickle'
    with open(filename, 'rb') as f:
        _ = pickle.load(f)
        _ = pickle.load(f)
        _ = pickle.load(f)
        _ = pickle.load(f)
        _ = pickle.load(f)
        _ = pickle.load(f)
        _ = pickle.load(f)
        _ = pickle.load(f)
        le = pickle.load(f)
    #Le is a list containing one encoder per task
    return le






def loadAllTaskNoID( target_reg, print_shapes=True ):
    #The last value in the Ys is the Doc ID.

    filename = DATA + str( target_reg ) + '.pickle'
    with open( filename, 'rb' ) as f:
        train_x = pickle.load( f )
        train_y = pickle.load( f )
        val_x = pickle.load( f )
        val_y = pickle.load( f )
        test_x = pickle.load( f )
        test_y = pickle.load( f )
        unseen_x = pickle.load( f )
        unseen_y = pickle.load( f )

    #Remove DocId
    train_y = train_y[:,0:-1]
    val_y = val_y[:,0:-1]
    test_y = test_y[:,0:-1]
    unseen_y = unseen_y[:,0:-1]

    train_x = pad_sequences( train_x, maxlen= 1500, padding= 'pre', truncating= 'pre' )
    val_x = pad_sequences( val_x, maxlen= 1500, padding= 'pre', truncating= 'pre' )
    test_x = pad_sequences( test_x, maxlen= 1500, padding= 'pre', truncating= 'pre' )
    unseen_x = pad_sequences(unseen_x, maxlen=1500, padding='pre', truncating='pre')


    if print_shapes == True:
        print("Train: ", np.array(train_x).shape)
        print("Train_y: ", np.array(train_y).shape)
        print("Test: ", np.array(test_x).shape)
        print("Test_y: ", np.array(test_y).shape)
        print("Val: ", np.array(val_x).shape)
        print("Val_y: ", np.array(val_y).shape)
        print("Unseen : ", np.array(unseen_x).shape)
        print("Unseen_y: ", np.array(unseen_y).shape)

    return train_x, train_y, val_x, val_y, test_x, test_y, unseen_x, unseen_y


def loadAllTaskWithID( target_reg, print_shapes=True ):
    filename = DATA + str( target_reg ) + '.pickle'
    with open( filename, 'rb' ) as f:
        train_x = pickle.load( f )
        train_y = pickle.load( f )
        val_x = pickle.load( f )
        val_y = pickle.load( f )
        test_x = pickle.load( f )
        test_y = pickle.load( f )
        unseen_x = pickle.load( f )
        unseen_y = pickle.load( f )

    train_x = pad_sequences( train_x, maxlen= 1500, padding= 'pre', truncating= 'pre' )
    val_x = pad_sequences( val_x, maxlen= 1500, padding= 'pre', truncating= 'pre' )
    test_x = pad_sequences( test_x, maxlen= 1500, padding= 'pre', truncating= 'pre' )
    unseen_x = pad_sequences(unseen_x, maxlen=1500, padding='pre', truncating='pre')

    # train_y = train_y[:, TASK_DIC[task]]
    # val_y = val_y[:, TASK_DIC[task]]
    # test_y = test_y[:, TASK_DIC[task]]
    # unseen_y = unseen_y[:, TASK_DIC[task]]

    ### hjy for debug purpose
    '''
    train_x = train_x[ 0 : 10000 ]
    train_y = train_y[ 0 : 10000 ]
    val_x = val_x[ 0 : 1000 ]
    val_y = val_y[ 0 : 1000 ]
    test_x = test_x[ 0 : 1000 ]
    test_y = test_y[ 0 : 1000 ]
    '''

    if print_shapes == True:
        print("Train: ", np.array(train_x).shape)
        print("Train_y: ", np.array(train_y).shape)

        print("Test: ", np.array(test_x).shape)
        print("Test_y: ", np.array(test_y).shape)

        print("Val: ", np.array(val_x).shape)
        print("Val_y: ", np.array(val_y).shape)

        print("Unseen : ", np.array(unseen_x).shape)
        print("Unseen_y: ", np.array(unseen_y).shape)

    return train_x, train_y, val_x, val_y, test_x, test_y, unseen_x, unseen_y


def loadTaskNoID( target_reg,TASK, print_shapes=True ):
    CLASSES_TASK = {"Site": 0, "Subsite": 1, "Laterality": 2, "Histology": 3, "Behavior": 4}

    filename = DATA + str( target_reg ) + '.pickle'
    with open( filename, 'rb' ) as f:
        train_x = pickle.load( f )
        train_y = pickle.load( f )
        val_x = pickle.load( f )
        val_y = pickle.load( f )
        test_x = pickle.load( f )
        test_y = pickle.load( f )
        unseen_x = pickle.load( f )
        unseen_y = pickle.load( f )

    train_y = train_y[:,CLASSES_TASK[TASK]]
    test_y = test_y[:,CLASSES_TASK[TASK]]
    unseen_y = unseen_y[:,CLASSES_TASK[TASK]]
    val_y = val_y[:,CLASSES_TASK[TASK]]



    train_x = pad_sequences( train_x, maxlen= 1500, padding= 'pre', truncating= 'pre' )
    val_x = pad_sequences( val_x, maxlen= 1500, padding= 'pre', truncating= 'pre' )
    test_x = pad_sequences( test_x, maxlen= 1500, padding= 'pre', truncating= 'pre' )
    unseen_x = pad_sequences(unseen_x, maxlen=1500, padding='pre', truncating='pre')


    if print_shapes == True:
        print("Train: ", np.array(train_x).shape)
        print("Train_y: ", np.array(train_y).shape)

        print("Test: ", np.array(test_x).shape)
        print("Test_y: ", np.array(test_y).shape)

        print("Val: ", np.array(val_x).shape)
        print("Val_y: ", np.array(val_y).shape)

        print("Unseen : ", np.array(unseen_x).shape)
        print("Unseen_y: ", np.array(unseen_y).shape)

    return train_x, train_y, val_x, val_y, test_x, test_y, unseen_x, unseen_y

