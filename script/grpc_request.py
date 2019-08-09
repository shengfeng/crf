
import grpc
import tensorflow as tf

import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2

channel = grpc.insecure_channel('192.168.115.26:9000')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'ner-serving'
request.model_spec.signature_name = 'ner'

request.inputs['input_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(np.zeros((0),   dtype=int).tolist(), shape=[1, 100]))
# request.inputs['input_mask'].CopyFrom(tf.contrib.util.make_tensor_proto(np.zeros((60),   dtype=int).tolist(), shape=[1, 60]))
# request.inputs['label_ids'].CopyFrom  (tf.contrib.util.make_tensor_proto([0], shape=[1, 1]))
# request.inputs['segment_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(np.zeros((60),   dtype=int).tolist(), shape=[1, 60]))

import time
begin = time.time()
# for i in range(0, 100):
#     result = stub.Predict(request, 10.0)  # 10 secs timeout
result = stub.Predict(request, 10.0)
end = time.time() - begin
print('time {}'.format(end))
print(result.outputs['outputs'].float_val)
print('length of probabilities:{}'.format(len(result.outputs['outputs'].float_val)))