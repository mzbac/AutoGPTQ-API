
## Create a new conda environment
```
conda create -n autogptq python=3.10.9
conda activate autogptq
```
## Install Pytorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

```
## Install AutoGPTQ
### Quick Installation
```
pip install auto-gptq
```
### Install from source
```
mkdir repositories
cd repositories

git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ

pip install .
```

## Install dependencies
```
pip install -r requirements.txt
```

## Create a self-signed certificate
```
openssl req -x509 -out cert.pem -keyout key.pem \
  -newkey rsa:2048 -nodes -sha256 \
  -subj '/CN=localhost' -extensions EXT -config <( \
   printf "[dn]\nCN=localhost\n[req]\ndistinguished_name = dn\n[EXT]\nsubjectAltName=DNS:localhost\nkeyUsage=digitalSignature\nextendedKeyUsage=serverAuth")
```
## Usage

1. Blocking api, update the model name and model weight path in blocking_api.py and run.
```
python blocking_api.py

```
The server will start on localhost port 5000.

To generate text, send a POST request to the /api/v1/generate endpoint. The request body should be a JSON object with the following keys:
prompt: The input prompt (required).
min_length: The minimum length of the sequence to be generated (optional, default is 0).
max_length: The maximum length of the sequence to be generated (optional, default is 50).
top_p: The nucleus sampling probability (optional, default is 0.95).
temperature: The temperature for sampling (optional, default is 0.6). For example, you can use curl to send a request
```
curl -k -s -X POST https://localhost:5000/api/v1/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request\n### Instruction: write a for loop in typescript\n### Response:", "max_length": 1000, "temperature": 0.7}'
```