#!/usr/bin/env python
# encoding: utf-8

import requests 
import re, json

url     = "http://127.0.0.1:8000/downthrpt"
data    = {'features': [201.124649],'model': 'lstm_mid'}   #features: vazão em KB/s; model: modelo de predição
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
r       = requests.post(url, data=json.dumps(data), headers=headers)
print r.text
