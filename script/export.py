import tensorflow as tf
import pickle
import model
from keras_contrib.utils import save_load_utils

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)

with open('models/config.pkl', 'rb') as inp:
    (vocab, chunk_tags) = pickle.load(inp)

VOCAB_SIZE = len(vocab)
TAGS_NUMS = len(chunk_tags)

predict_model = model.crf_model(VOCAB_SIZE, TAGS_NUMS)
save_load_utils.load_all_weights(predict_model, 'models/crf.h5', include_optimizer=False)
export_path = 'ner_serving'

# Fetch the keras session and save the model
# The signature definition is defined by the input and output tensor
# And stored with the default serving key
model.save_model_to_serving(predict_model, "1")
