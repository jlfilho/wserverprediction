#!/usr/bin/env python
# encoding: utf-8

import requests 
import re, json

'''
O parâmetro features é a vazão do tempo corrente (t) em Kb/s 
O parâmetro model é o modelo de predição (lstm_onetomany, lstm_onetoone,rf,...)
O serviço retorna ou uma lista com 5 valores (t+1,...,t+5) para o modelo lstm_onetomany e um valor para os demais modelos
retorna uma lista vazia se o modelo não existir. 
'''

url     = "http://127.0.0.1:8000/downthrpt"
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

#Requisição para modelo gru_mid_threetofive
data    = {'features': [1756.157952,2634.943488,1403.456512],'model': 'gru_mid_threetofive'}
r       = requests.post(url, data=json.dumps(data), headers=headers)

print("Modelo gru_mid_threetofive:")
print r.text

#Requisição para modelo gru_mid_threetoone
data    = {'features': [1756.157952,2634.943488,1403.456512],'model': 'gru_mid_threetoone'}
r       = requests.post(url, data=json.dumps(data), headers=headers)
print("Modelo gru_mid_threetoone:")
print r.text
