# 命名实体识别

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

