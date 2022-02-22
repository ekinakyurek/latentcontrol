# Multi LM

## Installation
```SHELL
    python3 -m venv .multilm
    source activate .multilm/bin/activate
    pip install -r requirements.txt
```

## Running
```SHELL
    python main.py
```
or
```SHELL
    python -m scripts.data
    python -m scripts.datastats
```

```SHELL
    python minGPT/train.py --n_layer 4 --n_head 4 --n_embd 128 --gpus 1 --precision 32 --batch_size 1 --block_size 8 --strategy deepspeed_stage_2 
    python minGPT/evaluate.py --gpus 1 --precision 32 --batch_size 1  --block_size 8 --strategy deepspeed_stage_2 
```