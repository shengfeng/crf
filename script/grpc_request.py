
import grpc
import tensorflow as tf

import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
import pickle
from process import process_data

channel = grpc.insecure_channel('192.168.115.26:9000')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'ner-serving'
request.model_spec.signature_name = 'ner'

with open('models/config.pkl', 'rb') as inp:
    (vocab, chunk_tags) = pickle.load(inp)

VOCAB_SIZE = len(vocab)
TAGS_NUMS = len(chunk_tags)
predict_text = '中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
# predict_text = '我留言板球球群你们加下找我要回关'
text, length = process_data(predict_text, vocab)

data = np.array(text[0], dtype=float).tolist()

request.inputs['input_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(data, shape=[1, 100]))

import time
begin = time.time()
result = stub.Predict(request, 10.0)
end = time.time() - begin
print('time {}'.format(end))
print('length of probabilities:{}'.format(len(result.outputs['outputs'].float_val)))

output = result.outputs['outputs'].float_val
output = np.array(output).reshape(1, 100, 7)

raw = output[0][-length:]

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
