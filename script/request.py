from process import process_data
import pickle
import json
import requests

with open('models/config.pkl', 'rb') as inp:
    (vocab, chunk_tags) = pickle.load(inp)

predict_text = '中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
# predict_text = '我留言板球球群你们加下找我要回关'
text, length = process_data(predict_text, vocab)

test_data = dict()
test_data['inputs'] = [0] * 100


data = json.dumps({"signature_name": "ner", "instances": [test_data]})
print(data)

headers = {"content-type": "application/json"}
json_response = requests.post('http://192.168.115.26:9001/v1/models/ner-serving:predict', data=data, headers=headers)

print(json_response.text)

predictions = json.loads(json_response.text)

print(predictions)