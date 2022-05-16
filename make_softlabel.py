import numpy as np
import pickle5 as pickle
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
#from loaddata6reg import loadAllTasks
#from NewDataLoader import loadAllTaskNoID
from models import NewDataLoader
import os
import argparse
from models import evaluationMetrics

model_name = "ensemble6"

r_target=4
#EnsembleSize = 276
EnsembleSize = 1000
#model_dir = "//gpfs/alpine/med107/proj-shared/kevindeangeli/interRegistrySavedModels/destilationModels/DataWithID/R4_Ensemble/"
model_dir = "//gpfs/alpine/med107/proj-shared/kevindeangeli/interRegistrySavedModels/destilationModels/DataWithID/R4_Ensemble/"
scores_dir = "results/" + "R" + str(r_target) + "/ensembleResult/"
model_name = "ensemble" + str(EnsembleSize)
if not os.path.exists(scores_dir):
    os.makedirs(scores_dir)
train_x, train_y, val_x, val_y, test_x, test_y, unseen_x, unseen_y = NewDataLoader.loadAllTaskNoID(r_target, print_shapes=True)




for k in range(EnsembleSize):
    with open( model_dir+'preds_' + str( k ) + '.pickle', 'rb' ) as f :
        train_preds = pickle.load( f )
        val_preds = pickle.load( f )
        test_preds = pickle.load( f )
        unseen_preds = pickle.load( f )

    if k == 0:
        train_soft = train_preds
        val_soft = val_preds
        test_soft = test_preds
        unseen_soft = unseen_preds
    else:
        for t in range( len( train_preds ) ):
            train_soft[ t ] += train_preds[ t ]
            val_soft[ t ] += val_preds[ t ] #I expect this to be out of bounds.
            test_soft[ t ] += test_preds[ t ]
            unseen_soft[ t ] += unseen_preds[ t ]


for t in range( len( train_soft ) ):
    train_soft[ t ] = normalize( train_soft[ t ], axis= 1, norm= 'l1' )
    val_soft[ t ] = normalize( val_soft[ t ], axis= 1, norm= 'l1' )
    test_soft[ t ] = normalize( test_soft[ t ], axis= 1, norm= 'l1' )
    unseen_soft[ t ] = normalize( unseen_soft[ t ], axis= 1, norm= 'l1' )


#Save softlabels:
sosftLabDir = model_dir + 'softlabels/'
if not os.path.exists(sosftLabDir):
    os.makedirs(sosftLabDir)

with open( sosftLabDir + model_name + '_softy.pickle', 'wb' ) as f:
    pickle.dump( train_soft, f, protocol= pickle.HIGHEST_PROTOCOL )
    pickle.dump( val_soft, f, protocol= pickle.HIGHEST_PROTOCOL )
    pickle.dump( test_soft, f, protocol= pickle.HIGHEST_PROTOCOL )
    pickle.dump( unseen_soft, f, protocol= pickle.HIGHEST_PROTOCOL )


print("Test Score --------------------------")
for t in range( len( test_y[ 0, : ] ) ):
    preds = [ np.argmax( x ) for x in test_soft[ t ] ]
    print( 'Task', t, 'Micro F1', f1_score( test_y[ :, t ], preds, average= 'micro' ) )
    print( 'Task', t, 'Macro F1', f1_score( test_y[ :, t ], preds, average= 'macro' ) )

print("Unseen Score --------------------------")
for t in range( len( unseen_y[ 0, : ] ) ):
    preds = [ np.argmax( x ) for x in unseen_soft[ t ] ]
    print( 'Task', t, 'Micro F1', f1_score( unseen_y[ :, t ], preds, average= 'micro' ) )
    print( 'Task', t, 'Macro F1', f1_score( unseen_y[ :, t ], preds, average= 'macro' ) )

import pandas as pd
micMac=[]
for t in range( len( test_y[ 0, : ] ) ):
    preds = [ np.argmax( x ) for x in test_soft[ t ] ]
    micro = f1_score( test_y[ :, t ], preds, average= 'micro' )
    macro = f1_score( test_y[ :, t ], preds, average= 'macro' )
    micMac.append(micro)
    micMac.append(macro)
data = np.zeros(shape=(1,   len( test_y[ 0, : ] ) *2))
data = np.vstack((data, micMac))

#Calculate unseen:
micMac= []
for t in range( len( unseen_y[ 0, : ] ) ):
    preds = [ np.argmax( x ) for x in unseen_soft[ t ] ]
    micro = f1_score( unseen_y[ :, t ], preds, average= 'micro' )
    macro = f1_score( unseen_y[ :, t ], preds, average= 'macro' )
    micMac.append(micro)
    micMac.append(macro)

data = np.vstack((data, micMac))
df0 = pd.DataFrame(data,
                   columns=['Sit_Mic', 'Sit_Mac', 'Sub_Mic', 'Sub_Mac', 'Lat_Mic', 'Lat_Mac', 'His_Mic',
                            'His_Mac', 'Be_Mic', 'Be_Mac'])

df0.to_csv(scores_dir + model_name +"Ensemble_" +"Test.csv")

pred = [val_soft, test_soft, unseen_soft]
true = [val_y,test_y,unseen_y]
evaluationMetrics.find97AccThresholdTASK(pred, true,saveDir =model_name)
evaluationMetrics.makeSitePredHistogram(unseen_soft, saveDir=model_name)





