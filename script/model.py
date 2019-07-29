from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
import pickle
from process import load_data
import tensorflow as tf
import os
from keras import backend as K

EMBED_DIM = 200
BIRNN_UNITS = 200

def crf_model(VOCAB_SIZE, TAGS_NUMS):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True))
    model.add(Bidirectional(LSTM(BIRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(TAGS_NUMS, sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    return model



def save_model_to_serving(model, export_version, export_path='ner_serving'):
    print(model.input, model.output)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(                                                                        
        inputs={'inputs': model.input}, outputs={'outputs': model.output})
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

    