import os
from typing import List, Mapping
import numpy as np
from src.utils import set_seed
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(file: str = "datasets/SimpleKB/corpus.txt") -> List[List[str]]:
    data = []
    with open(file) as f:
        for line in f:
            words = line.strip().split()
            words = ["[s]"] + words + ["[e]"]
            data.append(words)
    return data

def get_cx(data: List[List[str]]) -> Counter:
    cx = Counter()
    for d in data:
        for w in d:
            cx[w] += 1
    return cx


def get_cxy(data: List[List[str]]) -> Counter:
    cxy = Counter()
    for d in data:
        for i in range(len(d)-1):
            cxy[(d[i], d[i+1])] += 1
    return cxy

def get_py_x(cx: Mapping, cxy: Mapping) -> Mapping:
    cxy = dict(cxy)
    py_x = {}
    for k, v in cxy.items():
        x, y = k
        if y not in py_x:
            py_x[y] = {}
        py_x[y][x] = v / cx[x]
    return py_x


if __name__ == "__main__":
    sns.set_theme()
    plt.style.use(".mplstyle")
    os.makedirs("plots", exist_ok=True)
    
    set_seed(0)
    
    data = read_data()
    cx = get_cx(data)
    cxy = get_cxy(data)
    py_x = get_py_x(cx, cxy)
    
    vocab = list(cx.keys())
    N = len(vocab)
    G = np.zeros((N, N), dtype=np.float32)
    
    for y, v in py_x.items():
        j = vocab.index(y)
        for x, p in v.items():
            i = vocab.index(x)
            G[i,j] = p
    
    ax = sns.heatmap(G, 
                     xticklabels=vocab, 
                     yticklabels=vocab, 
                     vmin=0, 
                     vmax=1.0, 
                     annot=True, 
                     fmt=".1f")
    
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0)
    ax.set_title("P(X|Y)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('plots/bigram.png', dpi=300)
    
        
    

        
        