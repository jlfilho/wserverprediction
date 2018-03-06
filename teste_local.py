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
data    = {'features': [0.24224877136486545,0.6388405511221528,1.5440887892700914],'model': 'gru_mid_threetofive'}
r       = requests.post(url, data=json.dumps(data), headers=headers)

print("Modelo gru_mid_threetofive:")
print(r.text)

#Requisição para modelo gru_mid_fivetoone
data    = {'features': [0.24224877136486545,0.6388405511221528,1.5440887892700914,2.0210784622897293,0.5829505337960819],'model': 'gru_mid_fivetoone'}
r       = requests.post(url, data=json.dumps(data), headers=headers)
print("Modelo gru_mid_fivetoone:")
print(r.text)
