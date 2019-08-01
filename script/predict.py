import model
from process import process_data
from keras_contrib.utils import save_load_utils
import numpy as np
import pickle

with open('models/config.pkl', 'rb') as inp:
    (vocab, chunk_tags) = pickle.load(inp)

VOCAB_SIZE = len(vocab)
TAGS_NUMS = len(chunk_tags)

predict_model = model.crf_model(VOCAB_SIZE, TAGS_NUMS)
save_load_utils.load_all_weights(predict_model, 'models/crf.h5', include_optimizer=False)

predict_text = '中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
# predict_text = '我留言板球球群你们加下找我要回关'
text, length = process_data(predict_text, vocab)
print(text)

raw = predict_model.predict(text)[0][-length:]
result = [np.argmax(row) for row in raw]
result_tags = [chunk_tags[i] for i in result]

per, loc, org = '', '', ''

for s, t in zip(predict_text, result_tags):
    if t in ('B-PER', 'I-PER'):
        per += ' ' + s if (t == 'B-PER') else s
    if t in ('B-ORG', 'I-ORG'):
        org += ' ' + s if (t == 'B-ORG') else s
    if t in ('B-LOC', 'I-LOC'):
        loc += ' ' + s if (t == 'B-LOC') else s

print(['person:' + per, 'location:' + loc, 'organzation:' + org])