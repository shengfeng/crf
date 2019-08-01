import model
from process import load_data
from keras_contrib.utils import save_load_utils

EPOCH = 10

(train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = load_data()

VOCAB_SIZE = len(vocab)
TAGS_NUMS = len(chunk_tags)

model = model.crf_model(VOCAB_SIZE, TAGS_NUMS)
model.fit(train_x, train_y, batch_size=128, epochs=EPOCH, validation_data=[test_x, test_y])

# save_load_utils.save_all_weights(model, 'models/crf.h5')
model.save('models/crf.h5')