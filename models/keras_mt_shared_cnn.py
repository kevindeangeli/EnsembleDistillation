import numpy as np
#np.random.seed(1337)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Embedding
from tensorflow.keras.layers import GlobalMaxPooling1D, Convolution1D
from tensorflow.keras.layers import Concatenate



def init_export_network(num_classes,
                        in_seq_len,
                        vocab_size,
                        wv_space,
                        filter_sizes,
                        num_filters,
                        concat_dropout_prob,
                        #emb_l2,
                        #w_l2,
                        fc_layers,
                        n_fc,
                        fc_dropout,
                        #fc_l2,
                        optimizer,
                        learning_rate,
                        ):


    # define network layers ----------------------------------------------------
    input_shape = tuple([in_seq_len])
    model_input = Input(shape=input_shape, name= "Input")
    # embedding lookup
    emb_lookup = Embedding(vocab_size,
                           wv_space,
                           input_length=in_seq_len,
                           name="embedding",
                           #embeddings_initializer=RandomUniform,
                           #embeddings_regularizer=l2(emb_l2)
                           )(model_input)
    # convolutional layer and dropout
    conv_blocks = []
    for ith_filter,sz in enumerate(filter_sizes):
        conv = Convolution1D(filters=num_filters[ ith_filter ],
                             kernel_size=sz,
                             padding="same",
                             activation="relu",
                             strides=1,
                             #kernel_regularizer= l2( w_l2 ),
                             # kernel_initializer ='lecun_uniform,
                             name=str(ith_filter) + "_thfilter")(emb_lookup)
        conv_blocks.append(GlobalMaxPooling1D()(conv))
    concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    concat_drop = Dropout(concat_dropout_prob)(concat)

    # different dense layer per tasks
    FC_models = []
    for i in range(len(num_classes)):
        ts_layer = []
        ts_layer.append( concat_drop )
        for k in range( fc_layers ):
            lay = Dense( n_fc, name= 'TS'+str(i)+str(k), activation= 'relu', 
                #kernel_regularizer= l2( fc_l2 ) 
                )( ts_layer[ -1 ] )
            ts_layer.append( lay )
            lay = Dropout( fc_dropout )( ts_layer[ -1 ] )
            ts_layer.append( lay )
        outlayer = Dense(num_classes[i], name= "Dense"+str(i), activation='softmax')( ts_layer[ -1 ] )#, kernel_regularizer=l2(0.01))( concat_drop )
        FC_models.append(outlayer)


    # the multitsk model
    model = Model(inputs=model_input, outputs = FC_models)

    embedding_layer = model.get_layer("embedding")
    wv_matrix = np.random.uniform( -0.05, 0.05, size= ( vocab_size, wv_space ) )
    embedding_layer.set_weights([wv_matrix])

    model.compile( loss= "sparse_categorical_crossentropy", optimizer= optimizer, metrics=[ "acc" ] )

    #opt = Lamb()
    #model.compile( loss= 'sparse_categorical_crossentropy', optimizer= opt, metrics= [ 'acc' ] )


    return model






if __name__ == "__main__":
    main()


