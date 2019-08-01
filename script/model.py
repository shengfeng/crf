import pickle
from process import load_data
import tensorflow as tf
import os
from keras import backend as K
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras.layers import Dropout, TimeDistributed, Dense
from keras_contrib.layers import CRF


EMBEDDING_OUT_DIM = 128
TIME_STAMPS = 100
HIDDEN_UNITS = 200
DROPOUT_RATE = 0.3

def crf_model(VOCAB_SIZE, TAGS_NUMS):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, output_dim=EMBEDDING_OUT_DIM, mask_zero=True, input_length=TIME_STAMPS))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(TimeDistributed(Dense(TAGS_NUMS)))
    crf = CRF(TAGS_NUMS, sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
    return model



def save_model_to_serving(model, export_version, export_path='ner_serving'):
    print(model.input, model.output)
    signature = tf.saved_model.signature_def_utils.predict_signature_def( 
        inputs={'input_ids': model.input}, outputs={'outputs': model.output})
    export_path = os.path.join(
        tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(export_version)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),                                                                                                                    
        tags=[tf.saved_model.tag_constants.SERVING],                                                                                             
        signature_def_map={                                                                                                                      
            'ner': signature,                                                                                                                     
        },
        legacy_init_op=legacy_init_op)
    builder.save()

    