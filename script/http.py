#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json
import time

url = "http://120.92.188.233:8889"
# url = "http://127.0.0.1:8889"

def http_post(url, message):
    body = { 
        "appid": "server_test", 
        "opid": 10001, 
        "uid": 1049760434, 
        "msgid": 10001, 
        "sender_name": "巴音布鲁克",  
        "channel_id": 0, 
        "channel_type": "public", 
        "message": message, 
        "time": int(time.time()) 
        }

    content = "message="+ json.dumps(body)
    headers = {'content-type': "application/x-www-form-urlencoded"}
    response = requests.post(url, data = content, headers = headers)

    print(response.text)


data_set = []
with open("data/test_data.txt") as f:
    array = f.readlines()
    for arr in array:
        line = arr.replace('\n','')
        data_set.append(line)

count = 0
for i in range(0, 10000):
    for data in data_set:
        count += 1
        print("count = %d" % count)
        http_post(url, data)
        time.sleep(0.1)