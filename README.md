# wserverprediction

# Ubuntu 16.04 - Instalação de virutalenv e pacotes necessários

## Python 2.7
sudo apt-get update
sudo apt-get -y install python-pip python-dev python-virtualenv
mkdir -p ./tensorflow
python -m virtualenv --system-site-packages ./tensorflow
source ./tensorflow/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow keras matplotlib scikit-learn pandas bottle numpy h5py


## Python 3.x
sudo apt-get update
sudo apt-get install python3-pip python3-dev python-virtualenv
mkdir -p ./tensorflow
virtualenv --system-site-packages -p python3 ./tensorflow/
source ./tensorflow/bin/activate
pip3 install --upgrade tensorflow keras matplotlib jupyter scikit-learn pandas bottle numpy h5py

# Executar o serviço de predição

source ./tensorflow/bin/activate
./webservice.py


# Fazer predição
## Modelo GRU com 5 lags e predição para 1 time
### Requisição para modelo gru_mid_fivetoone

A requisição deve ser via método post do http com envio de um json com as medições da vazão dos últimos 5 segmentos baixados, o serviço retorna  a predição da vazão para o próximo segmento.
As medições de vazão devem estar em Mbps. 

data    = {'features': [0.24224877136486545,0.6388405511221528,1.5440887892700914,2.0210784622897293,0.5829505337960819],'model': 'gru_mid_fivetoone'}
r       = requests.post(url, data=json.dumps(data), headers=headers)
print(r.text)


### Requisição para modelo gru_mid_threetofive
data    = {'features': [0.24224877136486545,0.6388405511221528,1.5440887892700914],'model': 'gru_mid_threetofive'}
r       = requests.post(url, data=json.dumps(data), headers=headers)



Ver arquivo teste_local.py com modelo de requisição.
