# Multi LM
## Installation (conda)

```
conda create -n lm python=3.9
conda activate lm
pip install --upgrade pip
pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 torchaudio==0.10.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r reqs/pip-train.requirements.txt
pip install setuptools==59.5.0
```

## Installation (Docker)
```SHELL
   make up
```
