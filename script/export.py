import tensorflow as tf
import pickle
import model
from keras_contrib.utils import save_load_utils
from keras import backend as K

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)

with open('models/config.pkl', 'rb') as inp:
    (vocab, chunk_tags) = pickle.load(inp)

VOCAB_SIZE = len(vocab)
TAGS_NUMS = len(chunk_tags)

predict_model = model.crf_model(VOCAB_SIZE, TAGS_NUMS)
# save_load_utils.load_all_weights(predict_model, 'models/crf.h5', include_optimizer=False)
# export_path = 'ner_serving'

# Fetch the keras session and save the model
# The signature definition is defined by the input and output tensor
# And stored with the default serving key
# model.save_model_to_serving(predict_model, "1")

predict_model.load_weights('models/crf.h5')
legacy_init_op = tf.group(tf.tables_initializer())

with K.get_session() as sess:
    export_path = 'ner_serving/3'
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    signature_inputs = {
        'input_ids': predict_model.input
    }

    signature_outputs = {
        'outputs': predict_model.output
    }

    signature_def = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs=signature_inputs,
        outputs=signature_outputs)

    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'ner': signature_def
        },
        legacy_init_op=legacy_init_op
    )

    builder.save()