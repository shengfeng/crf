# 命名实体识别


## requirement

- keras >= 2.14

- keras contribute 2.0.8 (https://github.com/keras-team/keras-contrib)

- h5py

- pickle


## framework

- process: 数据预处理

- model: BiLSTM + CRF模型架构

- train: 模型训练

- grpc_request: 使用grpc连接tensorflow-serving进行预测

- request: 使用http连接tensorflow-serving进行预测


构建的模型是BiLSTM+CRF架构，具体模型搭建如下:

``` python
crf_model(VOCAB_SIZE, TAGS_NUMS)
```

导出训练模型：
```python
save_model_to_serving(model, export_version, export_path)
```

## demo

input:
```
中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚
```

output:
```
['person: 周恩来 陈毅, 王东', 'location: 埃塞俄比亚 非洲 阿尔巴尼亚', 'organzation: 中华人民共和国国务院 外交部']
```
